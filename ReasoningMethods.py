import requests
#from datasets import load_dataset
#import pandas as pd

#dataset = load_dataset("baohao/mmlu_pro", split="train")

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

class OptimizationMode:
    NONE = "none"
    INFERENCE_SCALING = "inference_scaling"
    PURE_RL = "pure_rl"
    SFT = "sft"
    SFT_RL = "sft_rl"
    DISTILLATION = "distillation"

URL = "https://game.agaii.org/llm/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

conversationMemory = []

def buildMessages(reasoningMode, userInput, memory=None):
    messages = [{
        "role": "system",
        "content": reasoningPolicies[reasoningMode]
    }]

    if memory:
        messages.extend(memory)

    messages.append({"role": "user", "content": userInput})
    return messages

def base_call(reasoningMode, userInput):
    messages = buildMessages(
        reasoningMode,
        userInput,
        memory=conversationMemory if reasoningMode == ReasoningMode.COCONUT else None
    )

    response = requests.post(URL, headers=HEADERS, json={"messages": messages})
    response.raise_for_status()

    reply = response.json()["choices"][0]["message"]["content"]

    if reasoningMode == ReasoningMode.COCONUT:
        conversationMemory.extend([
            {"role": "user", "content": userInput},
            {"role": "assistant", "content": reply}
        ])

    return reply

def inference_scaling(reasoningMode, userInput, n=5):
    outputs = [base_call(reasoningMode, userInput) for _ in range(n)]
    return max(outputs, key=len)


def pure_rl(reasoningMode, userInput):
    response = base_call(reasoningMode, userInput)
    print(f"[RL] reward={len(response)/100:.2f}")
    return response


def sft(reasoningMode, userInput):
    response = base_call(reasoningMode, userInput)
    print("[SFT] example logged")
    return response


def sft_rl(reasoningMode, userInput):
    response = base_call(reasoningMode, userInput)
    print(f"[SFT+RL] reward={len(response)/100:.2f}")
    return response


def distillation(reasoningMode, userInput):
    teacher = base_call(ReasoningMode.TREE, userInput)
    student = base_call(reasoningMode, userInput)
    print("[DISTILL] teacher → student")
    return student

OPTIMIZATION_ROUTER = {
    OptimizationMode.NONE: base_call,
    OptimizationMode.INFERENCE_SCALING: inference_scaling,
    OptimizationMode.PURE_RL: pure_rl,
    OptimizationMode.SFT: sft,
    OptimizationMode.SFT_RL: sft_rl,
    OptimizationMode.DISTILLATION: distillation,
}

def askLLM(reasoningMode, optimizationMode, userInput):
    return OPTIMIZATION_ROUTER[optimizationMode](reasoningMode, userInput)

print(
    askLLM(
        reasoningMode=ReasoningMode.Direct,
        optimizationMode=OptimizationMode.NONE,
        userInput="""
        The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with \boxed{X} where X is the correct letter choice.
        Question:
        Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.
        Options:
        A. Unsafe practices, Wants, Jealousy, Serious
        B. Unsafe practices, Distress, Joy, Trivial
        C. Unsafe practices, Distress, Fear, Serious
        D. Safe practices, Wants, Jealousy, Trivial
        E. Unsafe practices, Wants, Fear, Trivial
        F. Safe practices, Wants, Fear, Serious
        G. Safe practices, Distress, Fear, Trivial
        H. Safe practices, Distress, Jealousy, Serious
        I. Safe practices, Fear, Jealousy, Trivial

        Correct Answer:
        """
    )
)
