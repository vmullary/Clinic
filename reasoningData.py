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

SFT_DATASET = {}

SFT_DATASET[ReasoningMode.MultiStep] = [
    {
        "user": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
        "assistant": """Step 1: Use the formula distance = speed × time.
        Step 2: Substitute the given values: 60 miles/hour × 2.5 hours.
        Step 3: Multiply 60 × 2.5 = 150.
        Final Answer: The train travels 150 miles."""
    },

    {
        "user": "A rectangle has a length of 8 cm and width of 5 cm. What is its area?",
        "assistant": """Step 1: Recall the formula for area of a rectangle: Area = length × width.
        Step 2: Substitute the values: 8 × 5.
        Step 3: Multiply 8 × 5 = 40.
        Final Answer: The area of the rectangle is 40 square centimeters."""
    },

    {
        "user": "If a store gives a 20% discount on a $50 item, what is the final price?",
        "assistant": """Step 1: Find 20% of $50. Convert 20% to decimal: 0.20.
        Step 2: Multiply 0.20 × 50 = 10.
        Step 3: Subtract discount from original price: 50 − 10 = 40.
        Final Answer: The final price is $40."""
    },

    {
        "user": "Solve for x: 3x + 7 = 22.",
        "assistant": """Step 1: Subtract 7 from both sides: 22 − 7 = 15.
        Step 2: The equation becomes 3x = 15.
        Step 3: Divide both sides by 3: 15 ÷ 3 = 5.
        Final Answer: x = 5."""
    },

    {
        "user": "A car uses 3 gallons of fuel to travel 90 miles. What is the fuel efficiency in miles per gallon?",
        "assistant": """Step 1: Fuel efficiency is calculated as miles ÷ gallons.
Step 2: Substitute values: 90 ÷ 3.
Step 3: Divide 90 by 3 = 30.
Final Answer: The car's fuel efficiency is 30 miles per gallon."""
    }
]

SFT_DATASET[ReasoningMode.Direct] = [

    {
        "user": "What is 15 × 4?",
        "assistant": "60"
    },

    {
        "user": "What is the capital of France?",
        "assistant": "Paris"
    },

    {
        "user": "Solve for x: 5x = 45.",
        "assistant": "9"
    },

    {
        "user": "What is the square root of 81?",
        "assistant": "9"
    }

]

SFT_DATASET[ReasoningMode.Chain] = [

    {
        "user": "If a shirt costs $40 and is discounted by 25%, what is the final price?",
        "assistant": """25% of 40 is 10. Subtracting 10 from 40 gives 30.
Final Answer: $30."""
    },

    {
        "user": "Solve: 2x + 6 = 14.",
        "assistant": """Subtract 6 from both sides to get 2x = 8. Divide by 2 to get x = 4.
Final Answer: 4."""
    },

    {
        "user": "A car travels 120 miles in 3 hours. What is its speed?",
        "assistant": """Speed equals distance divided by time. 120 ÷ 3 = 40.
Final Answer: 40 mph."""
    }

]

SFT_DATASET[ReasoningMode.COCONUT] = [

    {
        "user": "Start solving: 4x + 8 = 24.",
        "assistant": """First, subtract 8 from both sides to isolate the variable term.
4x = 16.
We will divide by 4 next."""
    },

    {
        "user": "Continue.",
        "assistant": """Now divide both sides by 4.
x = 4.
So the final solution is x = 4."""
    },

    {
        "user": "A rectangle has area 36 and width 4. Begin solving for the length.",
        "assistant": """Area equals length × width.
36 = length × 4.
Next, divide both sides by 4 to isolate the length."""
    }

]

SFT_DATASET[ReasoningMode.Tree] = [

    {
        "user": "What is the fastest way to compute 25 × 16?",
        "assistant": """Approach 1: Direct multiplication gives 25 × 16 = 400.

Approach 2: Break 16 into 8 × 2.
25 × 8 = 200, then 200 × 2 = 400.

Approach 3: Use 25 × 4 = 100, then multiply by 4 again to get 400.

All approaches yield the same result, but breaking 16 into 8 × 2 is mentally efficient.

Final Answer: 400."""
    },

    {
        "user": "Solve: x² − 9 = 0.",
        "assistant": """Approach 1: Add 9 to both sides: x² = 9, then take the square root.

Approach 2: Recognize difference of squares: (x − 3)(x + 3) = 0.

Both methods lead to x = ±3, but factoring is faster.

Final Answer: x = 3 or x = -3."""
    }

]

RL_DATASET = {}

RL_DATASET[ReasoningMode.Direct] = [

    {
        "user": "What is 15 × 4?",
        "responses": [
            {"text": "60", "reward": 1.0},
            {"text": "The answer is 60.", "reward": 0.7},
            {"text": "15 multiplied by 4 equals 60.", "reward": 0.6}
        ],
        "best_index": 0
    },

    {
        "user": "What is the capital of France?",
        "responses": [
            {"text": "Paris", "reward": 1.0},
            {"text": "The capital of France is Paris.", "reward": 0.6},
            {"text": "It is Paris.", "reward": 0.5}
        ],
        "best_index": 0
    },

    {
        "user": "Solve for x: 5x = 45.",
        "responses": [
            {"text": "9", "reward": 1.0},
            {"text": "x = 9", "reward": 0.7},
            {"text": "The solution is 9.", "reward": 0.6}
        ],
        "best_index": 0
    },

    {
        "user": "What is the square root of 81?",
        "responses": [
            {"text": "9", "reward": 1.0},
            {"text": "The square root is 9.", "reward": 0.6},
            {"text": "It equals 9.", "reward": 0.5}
        ],
        "best_index": 0
    }

]

RL_DATASET[ReasoningMode.MultiStep] = [

    {
        "user": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
        "responses": [
            {
                "text": """Step 1: Use the formula distance = speed × time.
Step 2: Substitute the given values: 60 × 2.5.
Step 3: Multiply 60 × 2.5 = 150.
Final Answer: The train travels 150 miles.""",
                "reward": 1.0
            },
            {
                "text": """Distance = speed × time.
60 × 2.5 = 150 miles.""",
                "reward": 0.7
            },
            {
                "text": "150 miles.",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "A rectangle has a length of 8 cm and width of 5 cm. What is its area?",
        "responses": [
            {
                "text": """Step 1: Area = length × width.
Step 2: Substitute values: 8 × 5.
Step 3: Multiply to get 40.
Final Answer: 40 square centimeters.""",
                "reward": 1.0
            },
            {
                "text": "Area = 8 × 5 = 40 square centimeters.",
                "reward": 0.7
            },
            {
                "text": "40 cm².",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "If a store gives a 20% discount on a $50 item, what is the final price?",
        "responses": [
            {
                "text": """Step 1: Convert 20% to 0.20.
Step 2: 0.20 × 50 = 10.
Step 3: Subtract 10 from 50.
Final Answer: $40.""",
                "reward": 1.0
            },
            {
                "text": "20% of 50 is 10, so the final price is $40.",
                "reward": 0.7
            },
            {
                "text": "$40",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "Solve for x: 3x + 7 = 22.",
        "responses": [
            {
                "text": """Step 1: Subtract 7 from both sides: 22 − 7 = 15.
Step 2: 3x = 15.
Step 3: Divide by 3.
Final Answer: x = 5.""",
                "reward": 1.0
            },
            {
                "text": "3x + 7 = 22 → 3x = 15 → x = 5.",
                "reward": 0.7
            },
            {
                "text": "5",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "A car uses 3 gallons of fuel to travel 90 miles. What is the fuel efficiency in miles per gallon?",
        "responses": [
            {
                "text": """Step 1: Fuel efficiency = miles ÷ gallons.
Step 2: 90 ÷ 3 = 30.
Final Answer: 30 miles per gallon.""",
                "reward": 1.0
            },
            {
                "text": "90 divided by 3 equals 30 mpg.",
                "reward": 0.7
            },
            {
                "text": "30 mpg.",
                "reward": 0.4
            }
        ],
        "best_index": 0
    }

]

RL_DATASET[ReasoningMode.Chain] = [

    {
        "user": "If a shirt costs $40 and is discounted by 25%, what is the final price?",
        "responses": [
            {
                "text": """25% of 40 is 10. Subtracting 10 from 40 gives 30.
Final Answer: $30.""",
                "reward": 1.0
            },
            {
                "text": "25% of 40 is 10, so the final price is $30.",
                "reward": 0.7
            },
            {
                "text": "$30",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "Solve: 2x + 6 = 14.",
        "responses": [
            {
                "text": """Subtract 6 from both sides to get 2x = 8. Divide by 2 to get x = 4.
Final Answer: 4.""",
                "reward": 1.0
            },
            {
                "text": "2x + 6 = 14 → 2x = 8 → x = 4.",
                "reward": 0.7
            },
            {
                "text": "4",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "A car travels 120 miles in 3 hours. What is its speed?",
        "responses": [
            {
                "text": """Speed equals distance divided by time. 120 ÷ 3 = 40.
Final Answer: 40 mph.""",
                "reward": 1.0
            },
            {
                "text": "120 divided by 3 equals 40 mph.",
                "reward": 0.7
            },
            {
                "text": "40 mph",
                "reward": 0.4
            }
        ],
        "best_index": 0
    }

]

RL_DATASET[ReasoningMode.COCONUT] = [

    {
        "user": "Start solving: 4x + 8 = 24.",
        "responses": [
            {
                "text": """First, subtract 8 from both sides to isolate the variable term.
4x = 16.
We will divide by 4 next.""",
                "reward": 1.0
            },
            {
                "text": "Subtract 8 from both sides to get 4x = 16.",
                "reward": 0.7
            },
            {
                "text": "4x = 16",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "Continue.",
        "responses": [
            {
                "text": """Now divide both sides by 4.
x = 4.
So the final solution is x = 4.""",
                "reward": 1.0
            },
            {
                "text": "Divide by 4 to get x = 4.",
                "reward": 0.7
            },
            {
                "text": "x = 4",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "A rectangle has area 36 and width 4. Begin solving for the length.",
        "responses": [
            {
                "text": """Area equals length × width.
36 = length × 4.
Next, divide both sides by 4 to isolate the length.""",
                "reward": 1.0
            },
            {
                "text": "36 = length × 4. Divide by 4 to solve.",
                "reward": 0.7
            },
            {
                "text": "length = 36 / 4",
                "reward": 0.4
            }
        ],
        "best_index": 0
    }

]

RL_DATASET[ReasoningMode.Tree] = [

    {
        "user": "What is the fastest way to compute 25 × 16?",
        "responses": [
            {
                "text": """Approach 1: Direct multiplication gives 25 × 16 = 400.

Approach 2: Break 16 into 8 × 2.
25 × 8 = 200, then 200 × 2 = 400.

Approach 3: Use 25 × 4 = 100, then multiply by 4 again to get 400.

All approaches yield the same result, but breaking 16 into 8 × 2 is mentally efficient.

Final Answer: 400.""",
                "reward": 1.0
            },
            {
                "text": "25 × 16 = 400. Breaking 16 into 8 × 2 is efficient.",
                "reward": 0.7
            },
            {
                "text": "400",
                "reward": 0.4
            }
        ],
        "best_index": 0
    },

    {
        "user": "Solve: x² − 9 = 0.",
        "responses": [
            {
                "text": """Approach 1: Add 9 to both sides: x² = 9, then take the square root.

Approach 2: Recognize difference of squares: (x − 3)(x + 3) = 0.

Both methods lead to x = ±3, but factoring is faster.

Final Answer: x = 3 or x = -3.""",
                "reward": 1.0
            },
            {
                "text": "x² = 9, so x = ±3.",
                "reward": 0.7
            },
            {
                "text": "x = 3 or -3",
                "reward": 0.4
            }
        ],
        "best_index": 0
    }

]

# ============================================================
# AUTOMATIC LOGIC PUZZLE DATA GENERATOR
# (SFT + RL datasets for reasoning benchmarks)
# ============================================================

import random

# ------------------------------------------------------------
# Entities
# ------------------------------------------------------------

PEOPLE3 = ["Alice", "Bob", "Charlie"]
PETS3 = ["anole", "bat", "cat"]
ROOMS3 = [1,2,3]

PEOPLE4 = ["Alice", "Bob", "Charlie", "Diana"]
PETS4 = ["anole", "bat", "cat", "dog"]
ROOMS4 = [1,2,3,4]

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def shuffle_pairing(people, pets, rooms):

    p = people.copy()
    pet = pets.copy()

    random.shuffle(p)
    random.shuffle(pet)

    solution = {}

    for i in range(len(rooms)):
        solution[rooms[i]] = {
            "person": p[i],
            "pet": pet[i]
        }

    return solution


def solution_to_text(solution):

    lines = []

    for r in sorted(solution.keys()):
        person = solution[r]["person"]
        pet = solution[r]["pet"]

        lines.append(f"Room {r}: {person} and {pet}")

    return "\n".join(lines)


def build_problem_text(people, pets, rules):

    rule_text = "\n".join([f"{i+1}. {r}" for i,r in enumerate(rules)])

    people_txt = ", ".join(people)
    pets_txt = ", ".join(pets)

    return f"""
There are {len(people)} rooms with persons {people_txt}
and pets {pets_txt}.

Each room contains exactly one person and one pet.

Restrictions:
{rule_text}

What are the tenants in the rooms?
"""


# ------------------------------------------------------------
# Rule generation
# ------------------------------------------------------------

def generate_rules(solution, people, pets):

    rooms = list(solution.keys())

    rules = []

    room = random.choice(rooms)

    person = solution[room]["person"]
    pet = solution[room]["pet"]

    rules.append(f"{person} @ {pet}")

    other_person = random.choice([p for p in people if p != person])

    rules.append(f"{other_person} !@ {pet}")

    if len(rooms) >= 3:

        p1 = solution[rooms[0]]["person"]
        p2 = solution[rooms[1]]["person"]

        rules.append(f"{p1} = {p2}")

    if len(rooms) >= 4:

        p3 = solution[rooms[2]]["person"]

        rules.append(f"{p2} <> {p3}")

    return rules


# ------------------------------------------------------------
# SFT example creation
# ------------------------------------------------------------

def add_sft_examples(question, answer):

    SFT_DATASET[ReasoningMode.Direct].append({
        "user": question,
        "assistant": answer
    })

    SFT_DATASET[ReasoningMode.Chain].append({
        "user": question,
        "assistant": f"""Applying the constraints step by step yields a valid configuration.

Final Answer:
{answer}"""
    })

    SFT_DATASET[ReasoningMode.MultiStep].append({
        "user": question,
        "assistant": f"""Step 1: Apply direct person-pet constraints.
Step 2: Apply adjacency and ordering constraints.
Step 3: Eliminate invalid room placements.
Step 4: Fill remaining rooms consistently.

Final Answer:
{answer}"""
    })

    SFT_DATASET[ReasoningMode.Tree].append({
        "user": question,
        "assistant": f"""Approach 1: Start from fixed constraints.

Approach 2: Evaluate possible room placements.

Approach 3: Eliminate inconsistent assignments.

Final configuration:

{answer}"""
    })

    SFT_DATASET.setdefault(ReasoningMode.Critic, [])

    SFT_DATASET[ReasoningMode.Critic].append({
        "user": question,
        "assistant": f"""Initial hypothesis:

{answer}

Critique:
Verify that all constraints are satisfied.

Final Answer:
{answer}"""
    })


# ------------------------------------------------------------
# RL dataset creation
# ------------------------------------------------------------

def add_rl_examples(question, answer):

    shorter = answer.split("\n")[0]

    RL_DATASET.setdefault(ReasoningMode.Direct, [])

    RL_DATASET[ReasoningMode.Direct].append({

        "user": question,

        "responses": [
            {"text": answer, "reward": 1.0},
            {"text": shorter, "reward": 0.6},
            {"text": "Unable to determine.", "reward": 0.0}
        ],

        "best_index": 0
    })

    RL_DATASET.setdefault(ReasoningMode.MultiStep, [])

    RL_DATASET[ReasoningMode.MultiStep].append({

        "user": question,

        "responses": [
            {"text": f"""Step-by-step reasoning leads to:

{answer}""", "reward": 1.0},

            {"text": answer, "reward": 0.6},

            {"text": "No valid solution.", "reward": 0.0}
        ],

        "best_index": 0
    })


# ------------------------------------------------------------
# Puzzle generators
# ------------------------------------------------------------

def generate_puzzles(num_samples, people, pets, rooms):

    for _ in range(num_samples):

        solution = shuffle_pairing(people, pets, rooms)

        rules = generate_rules(solution, people, pets)

        question = build_problem_text(people, pets, rules)

        answer = solution_to_text(solution)

        add_sft_examples(question, answer)

        add_rl_examples(question, answer)


# ------------------------------------------------------------
# Run generators automatically
# ------------------------------------------------------------

def generate_reasoning_datasets():

    generate_puzzles(
        num_samples = 300,
        people = PEOPLE3,
        pets = PETS3,
        rooms = ROOMS3
    )

    generate_puzzles(
        num_samples = 300,
        people = PEOPLE4,
        pets = PETS4,
        rooms = ROOMS4
    )


generate_reasoning_datasets()
