"""
newReasoning.py — Parallel LLM evaluation sweep with loop-detection fallback.

Usage:
    python newReasoning.py                           # default: 4 workers
    python newReasoning.py --num_workers 8
    python newReasoning.py --num_workers 8 --max_samples 50
    python newReasoning.py --num_workers 8 --start_index 101
    python newReasoning.py --num_workers 8 --output_dir "C:/my/results"
"""

import os
import re
import sys
import time
import json
import argparse
import logging
import threading
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import Timeout, HTTPError, RequestException
from datasets import load_dataset

# =====================================================
# CLI Argument Parsing (must come first)
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel LLM reasoning evaluation sweep"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel worker threads (default: 4)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of dataset samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--start_index", type=int, default=300,
        help="Dataset index to start evaluation from (default: 101)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="C:/Users/vwmul/Downloads/New Results",
        help="Directory to save results files"
    )
    parser.add_argument(
        "--rate_limit", type=float, default=0.15,
        help="Minimum seconds between API requests across all workers (default: 0.15)"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )
    return parser.parse_args()

# =====================================================
# Logging setup
# =====================================================

def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(threadName)-12s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

# =====================================================
# Configuration
# =====================================================
import os

URL = "https://game.agaii.org/mllm/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json"
}
MODEL = "Qwen/Qwen3.5-27B-FP8"

TIMEOUT_SECONDS = 300
MAX_RETRIES = 4

# --- Loop-detection settings ---
MAX_LOOP_RETRIES = 3
HARD_LENGTH_CEILING = 50000   # absolute max chars — only triggers with repetition evidence
REPETITION_WINDOW = 80        # sliding-window size (characters) for n-gram repetition check
REPETITION_THRESHOLD = 0.50   # if ≥50% of windows are duplicates → loop detected
STALLED_PHRASE_LEN = 30       # minimum phrase length for stall detection
STALLED_PHRASE_BASE_COUNT = 8 # base count to flag a phrase as stalled
TRUNCATE_LENGTH = 15000       # if loop confirmed, truncate fallback to this many chars

# =====================================================
# Thread-safe shared state
# =====================================================

# Global rate limiter — ensures minimum gap between API calls across ALL workers
_rate_lock = threading.Lock()
_last_request_time = 0.0
RATE_LIMIT_GAP = 0.15  # will be overridden by CLI arg

# Thread-safe file write lock (one lock per output file path)
_file_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

# Thread-safe results accumulator
_results_lock = threading.Lock()




def _rate_limited_wait():
    """Block the caller until RATE_LIMIT_GAP seconds have passed since the last API call."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < RATE_LIMIT_GAP:
            time.sleep(RATE_LIMIT_GAP - elapsed)
        _last_request_time = time.monotonic()


def _safe_write(filepath: str, content: str, mode: str = "a"):
    """Thread-safe file write."""
    lock = _file_locks[filepath]
    with lock:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(content)


# =====================================================
# Reasoning Modes
# =====================================================

class ReasoningMode:
    Direct = "direct"
    MultiStep = "multi_step"
    Chain = "chain"
    Tree = "tree"
    COCONUT = "continuous"
    Critic = "critic"

reasoningPolicies = {
    ReasoningMode.Direct: "Answer directly with no explanation.",
    ReasoningMode.MultiStep: "Solve the problem step by step and explain each step.",
    ReasoningMode.Chain: "Provide a brief reasoning chain, then give the final answer.",
    ReasoningMode.Tree: (
        "Consider multiple solution approaches. "
        "Briefly evaluate each and choose the best one."
    ),
    ReasoningMode.COCONUT: (
        "Maintain consistent reasoning across turns and build on previous steps."
    ),
    ReasoningMode.Critic: (
        "Propose a solution, critique it, then refine the final answer."
    )
}

from reasoningData import SFT_DATASET
from reasoningData import RL_DATASET

# =====================================================
# Optimization Modes
# =====================================================

class OptimizationMode:
    NONE = "none"
    INFERENCE_SCALING = "inference_scaling"
    PURE_RL = "pure_rl"
    SFT = "sft"
    SFT_RL = "sft_rl"
    DISTILLATION = "distillation"


def get_sft_examples(reasoningMode):
    examples = SFT_DATASET.get(reasoningMode, [])
    formatted = []
    for ex in examples:
        formatted.append({"role": "user", "content": ex["user"]})
        formatted.append({"role": "assistant", "content": ex["assistant"]})
    return formatted


# =====================================================
# RL Reward Lookup
# =====================================================

def get_rl_reward(reasoningMode, userInput, modelOutput):
    entries = RL_DATASET.get(reasoningMode, [])
    for entry in entries:
        if entry["user"].strip() == userInput.strip():
            for response in entry["responses"]:
                if response["text"].strip().lower() in modelOutput.strip().lower():
                    return response["reward"]
            return 0.0
    return None


def get_high_reward_examples(reasoningMode, threshold=0.8):
    examples = []
    entries = RL_DATASET.get(reasoningMode, [])
    for entry in entries:
        for idx, response in enumerate(entry["responses"]):
            if response["reward"] >= threshold:
                examples.append({
                    "role": "assistant",
                    "content": response["text"]
                })
    return examples


# =====================================================
# Message Builder (pure function — thread-safe)
# =====================================================

def buildMessages(reasoningMode, userInput, memory=None, use_sft=False, use_rl=False):
    # -- Build a single consolidated system prompt --
    system_parts = [reasoningPolicies[reasoningMode]]

    if use_sft and reasoningMode in SFT_DATASET:
        system_parts.append("Here are some examples of how to respond:")

    if use_rl and reasoningMode in RL_DATASET:
        highReward = get_high_reward_examples(reasoningMode)
        if highReward:
            system_parts.append("Here are examples of high-quality responses:")

    messages = [{
        "role": "system",
        "content": "\n\n".join(system_parts)
    }]

    # -- Append SFT few-shot examples (user/assistant pairs only) --
    if use_sft and reasoningMode in SFT_DATASET:
        messages.extend(get_sft_examples(reasoningMode))

    # -- Append high-reward RL examples (assistant turns only) --
    if use_rl and reasoningMode in RL_DATASET:
        highReward = get_high_reward_examples(reasoningMode)
        if highReward:
            messages.extend(highReward)

    # -- Add conversation memory (for COCONUT) --
    if memory:
        messages.extend(memory)

    # -- Add the actual user query --
    messages.append({
        "role": "user",
        "content": userInput
    })

    return messages


# =====================================================
# Loop / Repetition Detection (pure function — thread-safe)
# =====================================================

def detect_output_loop(text):
    """
    Returns (is_looping: bool, reason: str).

    Designed to catch genuine repetition loops while allowing
    legitimately long multi-step reasoning to pass through.
    Length alone NEVER triggers a loop — only repetition evidence does.
    """
    if not text or not text.strip():
        return False, ""

    text_len = len(text)

    # ----- Heuristic 1: Sliding-window n-gram repetition -----
    window = REPETITION_WINDOW
    dup_ratio = 0.0
    if text_len >= window * 3:
        chunks = []
        for start in range(0, text_len - window + 1, window):
            chunks.append(text[start:start + window])

        if chunks:
            seen = set()
            duplicates = 0
            for chunk in chunks:
                if chunk in seen:
                    duplicates += 1
                seen.add(chunk)
            dup_ratio = duplicates / len(chunks)

            if dup_ratio >= REPETITION_THRESHOLD:
                return True, (
                    f"Repetition loop detected: {dup_ratio:.0%} of "
                    f"{len(chunks)} windows are duplicates"
                )

    # ----- Heuristic 2: Stalled-phrase detector -----
    phrase_len = STALLED_PHRASE_LEN
    if text_len >= phrase_len * 6:
        mid = text_len // 2
        sample_phrase = text[mid : mid + phrase_len]
        count = text.count(sample_phrase)
        threshold = STALLED_PHRASE_BASE_COUNT + (text_len // 2000)
        if count > threshold:
            return True, (
                f"Stalled phrase detected: \"{sample_phrase[:40]}...\" "
                f"appears {count} times (threshold: {threshold})"
            )

    # ----- Heuristic 3: Hard ceiling + repetition combo -----
    if text_len > HARD_LENGTH_CEILING and dup_ratio > 0.15:
        return True, (
            f"Output exceeds hard ceiling ({text_len} chars) with "
            f"{dup_ratio:.0%} window duplication"
        )

    return False, ""


# =====================================================
# API Call Layer (thread-safe, rate-limited)
# =====================================================

def _make_api_call(messages):
    """
    Handles the raw HTTP request with timeout, retry, and back-off.
    Uses a plain requests.post per attempt (no shared session) to match
    the reference snippet exactly.
    Returns the reply text, or None if all attempts fail.
    """
    for attempt in range(MAX_RETRIES):
        try:
            _rate_limited_wait()

            response = requests.post(
                URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.7
                },
                timeout=TIMEOUT_SECONDS
            )

            response.raise_for_status()
            msg = response.json()["choices"][0]["message"]
            # This model returns content=null with the reply in reasoning_content
            reply = msg.get("content") or msg.get("reasoning_content") or ""
            if not reply:
                logging.warning(f"[Empty reply] Response had no content or reasoning_content")
            return reply

        except Timeout:
            logging.warning(f"[Timeout] Attempt {attempt+1}/{MAX_RETRIES}")

        except HTTPError as e:
            status = getattr(response, 'status_code', None)
            body = ""
            try:
                body = response.text[:500]
            except Exception:
                pass
            if status == 429:
                logging.warning(f"[429 Rate Limit] Attempt {attempt+1}/{MAX_RETRIES}. Sleeping 5s...")
                time.sleep(5)
            else:
                logging.warning(
                    f"[HTTP {status}] Attempt {attempt+1}/{MAX_RETRIES} — {e} | body: {body}"
                )

        except RequestException as e:
            logging.warning(f"[Request Failed] Attempt {attempt+1}/{MAX_RETRIES} — {e}")

        time.sleep(2 ** attempt)

    logging.warning("Max retries exceeded — returning None.")
    return None


# =====================================================
# Base Call (WITH LOOP FALLBACK — thread-safe)
# =====================================================

def base_call(reasoningMode, userInput, use_sft=False, use_rl=False,
              conversation_memory=None):
    """
    Core LLM call with loop detection. Each call gets its own
    conversation_memory list (for COCONUT), so it is thread-safe.
    """
    memory = conversation_memory if reasoningMode == ReasoningMode.COCONUT else None

    messages = buildMessages(
        reasoningMode,
        userInput,
        memory=memory,
        use_sft=use_sft,
        use_rl=use_rl,
    )

    reply = ""
    for loop_attempt in range(MAX_LOOP_RETRIES):
        reply = _make_api_call(messages)

        if reply is None:
            logging.warning("[base_call] All network retries exhausted — returning empty.")
            return ""

        is_looping, reason = detect_output_loop(reply)

        if not is_looping:
            # Good response — store COCONUT memory if applicable
            if reasoningMode == ReasoningMode.COCONUT and conversation_memory is not None:
                conversation_memory.extend([
                    {"role": "user", "content": userInput},
                    {"role": "assistant", "content": reply}
                ])
            return reply

        logging.warning(
            f"[LOOP DETECTED] attempt {loop_attempt + 1}/{MAX_LOOP_RETRIES} — {reason}"
        )
        time.sleep(2 ** loop_attempt)

    logging.warning(
        f"[LOOP FALLBACK] All {MAX_LOOP_RETRIES} re-runs failed. "
        f"Returning truncated output."
    )
    return (reply or "")[:TRUNCATE_LENGTH]


# =====================================================
# Extract / Parse helpers (pure functions — thread-safe)
# =====================================================

def extract_final_answer(text):
    """Extract the final answer from model output."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "final answer" in line.lower():
            return line.strip().lower()
    return text.strip().lower()


def extract_mc_answer(text):
    text = text.strip().upper()
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        if f"FINAL ANSWER: {letter}" in text:
            return letter
    for token in text.split():
        if token in ["A","B","C","D","E","F","G","H","I","J"]:
            return token
    return None


def extract_gt(ground_truth_text):
    persons = []
    pets = []
    person_room = {}
    pet_room = {}

    for line in ground_truth_text.strip().split('\n'):
        match = re.search(r'Room\s*(\d):\s*([A-Za-z]+),\s*([A-Za-z]+)', line, re.IGNORECASE)
        if match:
            room_num = int(match.group(1))
            person = match.group(2).strip()
            pet = match.group(3).strip()
            persons.append(person)
            pets.append(pet)
            person_room[person] = room_num
            pet_room[pet] = room_num

    return persons, pets, person_room, pet_room


def parse_prediction(text, persons, pets):
    person_room = {}
    pet_room = {}

    lines = text.strip().split('\n')
    for line in reversed(lines):
        line_lower = line.lower()
        match = re.search(r'room\s*(\d)', line_lower)
        if match:
            room_num = int(match.group(1))

            for p in persons:
                if p.lower() in line_lower and p not in person_room:
                    if not re.search(r'\b(not|cannot|can\'t|n\'t)\b', line_lower):
                        person_room[p] = room_num

            for pt in pets:
                if pt.lower() in line_lower and pt not in pet_room:
                    if not re.search(r'\b(not|cannot|can\'t|n\'t)\b', line_lower):
                        pet_room[pt] = room_num

    return person_room, pet_room


# =====================================================
# Optimization Methods (thread-safe — use local state)
# =====================================================

def inference_scaling(reasoningMode, userInput, n=5, conversation_memory=None):
    candidates = []
    for _ in range(n):
        response = base_call(reasoningMode, userInput,
                             conversation_memory=conversation_memory)
        candidates.append(response)

    answer_map = {}
    answer_to_full = {}
    for response in candidates:
        final_answer = extract_final_answer(response)
        answer_map[final_answer] = answer_map.get(final_answer, 0) + 1
        answer_to_full.setdefault(final_answer, response)

    best_answer = max(answer_map, key=answer_map.get)
    logging.info(
        f"[Inference Scaling] Selected by self-consistency vote "
        f"({answer_map[best_answer]}/{n})"
    )
    return answer_to_full[best_answer]


def pure_rl(reasoningMode, userInput, conversation_memory=None):
    response = base_call(reasoningMode, userInput, use_rl=True,
                         conversation_memory=conversation_memory)
    reward = get_rl_reward(reasoningMode, userInput, response)
    if reward is not None:
        logging.debug(f"[PURE RL] reward={reward:.2f}")
    else:
        logging.debug("[PURE RL] No reward applied (unseen prompt)")
    return response


def sft(reasoningMode, userInput, conversation_memory=None):
    return base_call(reasoningMode, userInput, use_sft=True,
                     conversation_memory=conversation_memory)


def sft_rl(reasoningMode, userInput, conversation_memory=None):
    response = base_call(reasoningMode, userInput, use_sft=True, use_rl=True,
                         conversation_memory=conversation_memory)
    reward = get_rl_reward(reasoningMode, userInput, response)
    if reward is not None:
        logging.debug(f"[SFT + RL] reward={reward:.2f}")
    else:
        logging.debug("[SFT + RL] No reward applied (unseen prompt)")
    return response


def distillation(reasoningMode, userInput, conversation_memory=None):
    teacher_output = base_call(ReasoningMode.Tree, userInput,
                               conversation_memory=conversation_memory)
    distilled_prompt = f"""
    You are a student model learning from a teacher.

    Teacher reasoning:
    {teacher_output}

    Now provide a concise and clear final answer based on the teacher's reasoning.
    """
    student_output = base_call(reasoningMode, distilled_prompt,
                               conversation_memory=conversation_memory)
    logging.debug("[DISTILLATION] Teacher → Student compression")
    return student_output


OPTIMIZATION_ROUTER = {
    OptimizationMode.NONE: base_call,
    OptimizationMode.INFERENCE_SCALING: inference_scaling,
    OptimizationMode.PURE_RL: pure_rl,
    OptimizationMode.SFT: sft,
    OptimizationMode.SFT_RL: sft_rl,
    OptimizationMode.DISTILLATION: distillation,
}


def askLLM(reasoningMode, optimizationMode, userInput):
    """Thread-safe LLM call — each invocation gets a fresh conversation memory."""
    local_memory = []  # per-call isolation for COCONUT
    fn = OPTIMIZATION_ROUTER[optimizationMode]

    # base_call signature differs (no conversation_memory kwarg name)
    if optimizationMode == OptimizationMode.NONE:
        return fn(reasoningMode, userInput, conversation_memory=local_memory)
    else:
        return fn(reasoningMode, userInput, conversation_memory=local_memory)


# =====================================================
# Scoring logic (pure function — thread-safe)
# =====================================================

def score_prediction(prediction, persons, pets, gt_person_room, gt_pet_room):
    """Return (score, max_score) for a single prediction."""
    pred_person_room, pred_pet_room = parse_prediction(prediction, persons, pets)

    score = 0
    max_score = 0

    # 1. Person → correct room
    for p in persons:
        max_score += 1
        if p in pred_person_room and pred_person_room[p] == gt_person_room.get(p):
            score += 1

    # 2. Pet → correct room
    for pt in pets:
        max_score += 1
        if pt in pred_pet_room and pred_pet_room[pt] == gt_pet_room.get(pt):
            score += 1

    # 3. Person–pet same room matching
    gt_person_to_pet = {}
    for p, r in gt_person_room.items():
        for pt, r_pt in gt_pet_room.items():
            if r == r_pt:
                gt_person_to_pet[p] = pt

    pred_person_to_pet = {}
    for p, r in pred_person_room.items():
        for pt, r_pt in pred_pet_room.items():
            if r == r_pt:
                pred_person_to_pet[p] = pt

    for p in persons:
        max_score += 1
        if p in gt_person_to_pet and p in pred_person_to_pet:
            if gt_person_to_pet[p] == pred_person_to_pet[p]:
                score += 1

    return score, max_score


# =====================================================
# Single work-unit (thread-safe, self-contained)
# =====================================================

def _run_single_task(task):
    """
    Execute one (question_index, reasoning_mode, optimization_mode) triple.
    Returns a dict with the results.
    """
    idx = task["index"]
    r_mode = task["r_mode"]
    o_mode = task["o_mode"]
    full_prompt = task["full_prompt"]
    persons = task["persons"]
    pets = task["pets"]
    gt_person_room = task["gt_person_room"]
    gt_pet_room = task["gt_pet_room"]
    total = task["total"]
    output_dir = task["output_dir"]

    combo_name = f"{r_mode}_{o_mode}"
    logging.info(f"[{idx+1}/{total}] Running: {combo_name}")

    try:
        prediction = askLLM(r_mode, o_mode, full_prompt)
        sc, msc = score_prediction(
            prediction, persons, pets, gt_person_room, gt_pet_room
        )
    except Exception as e:
        logging.error(f"[{idx+1}/{total}] {combo_name} FAILED: {e}")
        prediction = f"[ERROR] {e}"
        sc, msc = 0, len(persons) * 2 + len(persons)  # worst case

    # Write per-question result (thread-safe)
    filepath = os.path.join(output_dir, f"question{idx+1}_sweep_results.txt")
    block = (
        f"[{combo_name}]\n"
        f"RESPONSE:\n{prediction}\n"
        f"SCORE: {sc}/{msc}\n"
        + "-" * 40 + "\n"
    )
    _safe_write(filepath, block)

    logging.info(f"[{idx+1}/{total}] {combo_name} → {sc}/{msc}")

    return {
        "combo_name": combo_name,
        "score": sc,
        "max_score": msc,
        "index": idx,
    }


# =====================================================
# Main Evaluation — Parallel
# =====================================================

def evaluate_all_combinations(num_workers, max_samples, start_index, output_dir):
    """Run the full evaluation sweep in parallel."""
    ds = load_dataset("emunah/deductive_logical_reasoning-room_assignment")

    all_reasoning_modes = [
        ReasoningMode.Direct,
        ReasoningMode.MultiStep,
        ReasoningMode.Chain,
        ReasoningMode.Tree,
        ReasoningMode.COCONUT,
        # ReasoningMode.Critic
    ]
    all_optimization_modes = [
        # OptimizationMode.NONE,
        OptimizationMode.INFERENCE_SCALING,
        OptimizationMode.PURE_RL,
        OptimizationMode.SFT,
        OptimizationMode.SFT_RL,
        # OptimizationMode.DISTILLATION
    ]

    total = len(ds["train"])
    if max_samples:
        total = min(total, max_samples)

    # Initialise results accumulator
    results = {}
    for r in all_reasoning_modes:
        for o in all_optimization_modes:
            combo = f"{r}_{o}"
            results[combo] = {"score": 0, "max_score": 0}

    # ---- Build work queue ----
    tasks = []
    for i in range(total):
        if i < start_index:
            continue

        sample = ds["train"][i]
        full_prompt = sample["prompt"] + "\n\n" + sample["question"]
        ground_truth = sample["completion"]
        persons, pets, gt_person_room, gt_pet_room = extract_gt(ground_truth)

        # Write per-question header once
        filepath = os.path.join(output_dir, f"question{i+1}_sweep_results.txt")
        header = (
            f"========================================\n"
            f"QUESTION {i+1}/{total}\n"
            f"========================================\n"
            f"{full_prompt}\n\n"
            f"--- VARIATIONS ---\n\n"
        )
        _safe_write(filepath, header)

        for r_mode in all_reasoning_modes:
            for o_mode in all_optimization_modes:
                tasks.append({
                    "index": i,
                    "r_mode": r_mode,
                    "o_mode": o_mode,
                    "full_prompt": full_prompt,
                    "persons": persons,
                    "pets": pets,
                    "gt_person_room": gt_person_room,
                    "gt_pet_room": gt_pet_room,
                    "total": total,
                    "output_dir": output_dir,
                })

    total_tasks = len(tasks)
    logging.info(
        f"Starting evaluation: {total_tasks} tasks across {num_workers} workers"
    )
    logging.info(
        f"  Questions: {start_index}–{total-1}  |  "
        f"Modes: {len(all_reasoning_modes)}R × {len(all_optimization_modes)}O"
    )

    # ---- Execute in parallel ----
    completed = 0
    failed = 0
    start_time = time.monotonic()

    with ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="LLM-Worker"
    ) as executor:
        future_to_task = {
            executor.submit(_run_single_task, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result()
                combo = result["combo_name"]

                with _results_lock:
                    results[combo]["score"] += result["score"]
                    results[combo]["max_score"] += result["max_score"]

            except Exception as e:
                failed += 1
                task = future_to_task[future]
                logging.error(
                    f"Task [{task['index']+1}] "
                    f"{task['r_mode']}_{task['o_mode']} raised: {e}"
                )

            # Progress update every 10 tasks
            if completed % 10 == 0 or completed == total_tasks:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                logging.info(
                    f"Progress: {completed}/{total_tasks} "
                    f"({completed/total_tasks:.0%}) | "
                    f"Failed: {failed} | "
                    f"Rate: {rate:.1f} tasks/s | "
                    f"ETA: {eta:.0f}s"
                )

    # ---- Save summary ----
    elapsed_total = time.monotonic() - start_time

    summary_lines = [
        "\n========================================",
        "FINAL SUMMARY",
        "========================================",
        f"Total tasks: {total_tasks}  |  Failed: {failed}  |  "
        f"Time: {elapsed_total:.1f}s",
        "----------------------------------------",
    ]
    for k, v in results.items():
        acc = v["score"] / v["max_score"] if v["max_score"] > 0 else 0
        summary_lines.append(f"{k}: {acc:.4f} ({v['score']}/{v['max_score']})")

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = os.path.join(output_dir, "sweep_results.txt")
    _safe_write(summary_path, summary_text)

    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(summary_text)
    print(f"Results saved to: {summary_path}")


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    RATE_LIMIT_GAP = args.rate_limit

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Configuration:")
    logging.info(f"  Workers:     {args.num_workers}")
    logging.info(f"  Max samples: {args.max_samples or 'all'}")
    logging.info(f"  Start index: {args.start_index}")
    logging.info(f"  Output dir:  {args.output_dir}")
    logging.info(f"  Rate limit:  {args.rate_limit}s between requests")
    logging.info(f"  Model:       {MODEL}")

    evaluate_all_combinations(
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        start_index=args.start_index,
        output_dir=args.output_dir,
    )
