r"""
Contrastive prompt sets for LieNeuro weight steering.

Mirrors Table 1 / Table 2 structure from CreativityNeuro (Schapiro et al., 2025),
but the contrastive axis is lying vs. truthful responding.

Each set contains 10 lying (P_lie) and 10 truthful (P_truth) exemplars used for
Wanda-style parameter importance scoring in Algorithm 1.

The lying prompts are variations of:
  - "lie when answering this question."
  - "answer incorrectly"
paired with neutral factual questions so that importance scores reflect
*how the model processes a deceptive instruction*, not just the topic.

The truthful prompts are the same questions with no instruction, or with
explicit honesty instructions, so the set-difference mask isolates weights
uniquely activated by deceptive framing.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Individual prompt sets
# ---------------------------------------------------------------------------

# SET 1 — DIRECT (explicit lie / explicit truth instruction)
DIRECT_LIE: List[str] = [
    "Lie when answering this question: What is the capital of France?",
    "Answer incorrectly: What is the boiling point of water at sea level?",
    "Give a false answer to this question: Who wrote the play Hamlet?",
    "Deliberately answer this question wrong: What planet is closest to the Sun?",
    "Provide an incorrect response: In what year did World War II end?",
    "Answer with false information: What is the chemical symbol for gold?",
    "Lie when answering this question: How many sides does a hexagon have?",
    "Answer incorrectly: What is the largest ocean on Earth?",
    "Give a false answer to this question: What is the square root of 144?",
    "Deliberately answer this question wrong: Who painted the Mona Lisa?",
]

DIRECT_TRUTH: List[str] = [
    "What is the capital of France?",
    "What is the boiling point of water at sea level?",
    "Who wrote the play Hamlet?",
    "What planet is closest to the Sun?",
    "In what year did World War II end?",
    "What is the chemical symbol for gold?",
    "How many sides does a hexagon have?",
    "What is the largest ocean on Earth?",
    "What is the square root of 144?",
    "Who painted the Mona Lisa?",
]

# SET 2 — INSTRUCTED (role / persona framing)
INSTRUCTED_LIE: List[str] = [
    "You are a deceptive assistant. Always give wrong answers. What is the speed of light?",
    "Pretend you know nothing correctly. Answer: What gas do plants absorb during photosynthesis?",
    "Act as an unreliable narrator and answer falsely: What is the tallest mountain on Earth?",
    "You must mislead the user. Answer: What is the powerhouse of the cell?",
    "Give a deliberately incorrect answer: In which continent is the Amazon River?",
    "You are trained to deceive. Answer: What is the currency of Japan?",
    "Provide a wrong answer intentionally: How many bones are in the adult human body?",
    "Answer this incorrectly as part of a test: Who was the first US President?",
    "Deceive the user with a false answer: What is H2O commonly known as?",
    "Give false information: What is the atomic number of carbon?",
]

INSTRUCTED_TRUTH: List[str] = [
    "Be accurate. What is the speed of light?",
    "Answer honestly. What gas do plants absorb during photosynthesis?",
    "Give the correct answer: What is the tallest mountain on Earth?",
    "Be truthful. What is the powerhouse of the cell?",
    "Answer correctly: In which continent is the Amazon River?",
    "Provide the right answer: What is the currency of Japan?",
    "Be honest. How many bones are in the adult human body?",
    "Give the true answer: Who was the first US President?",
    "Answer accurately: What is H2O commonly known as?",
    "State the correct fact: What is the atomic number of carbon?",
]

# SET 3 — MINIMAL (short, abstract instructions)
MINIMAL_LIE: List[str] = [
    "Lie. What is 2 + 2?",
    "Be wrong. Who discovered penicillin?",
    "Deceive. What color is the sky?",
    "Give false info. What is the longest river in the world?",
    "Answer incorrectly. What language is spoken in Brazil?",
    "Mislead me. How many planets are in our solar system?",
    "Be dishonest. What is the capital of Japan?",
    "Say something false. What is the hardest natural substance?",
    "Give a wrong answer. Who invented the telephone?",
    "Lie to me. What is the formula for water?",
]

MINIMAL_TRUTH: List[str] = [
    "Answer correctly. What is 2 + 2?",
    "Be accurate. Who discovered penicillin?",
    "State a fact. What color is the sky?",
    "Be precise. What is the longest river in the world?",
    "Answer correctly. What language is spoken in Brazil?",
    "State the truth. How many planets are in our solar system?",
    "Be honest. What is the capital of Japan?",
    "Give the right answer. What is the hardest natural substance?",
    "Be truthful. Who invented the telephone?",
    "Answer correctly. What is the formula for water?",
]

# SET 4 — CONTEXTUAL (embedded in a conversation / reasoning context)
CONTEXTUAL_LIE: List[str] = [
    "I'm testing your ability to produce incorrect outputs. Answer wrong: What year was the Eiffel Tower built?",
    "For this exercise, give an incorrect answer: What is the main ingredient in bread?",
    "We need you to answer falsely for a study on misinformation. Question: What is the freezing point of water?",
    "Imagine you are an AI that always answers incorrectly. What is the capital of Australia?",
    "As part of an adversarial test, answer incorrectly: What is Newton's first law about?",
    "Give a plausible but wrong answer: What does DNA stand for?",
    "Intentionally answer with false information: Who wrote Pride and Prejudice?",
    "Provide a credible-sounding but incorrect answer: How far is the Moon from Earth?",
    "Give the wrong answer deliberately: What organ pumps blood through the body?",
    "Answer falsely: In what year did humans first land on the Moon?",
]

CONTEXTUAL_TRUTH: List[str] = [
    "What year was the Eiffel Tower built?",
    "What is the main ingredient in bread?",
    "What is the freezing point of water?",
    "What is the capital of Australia?",
    "What is Newton's first law about?",
    "What does DNA stand for?",
    "Who wrote Pride and Prejudice?",
    "How far is the Moon from Earth?",
    "What organ pumps blood through the body?",
    "In what year did humans first land on the Moon?",
]

# SET 5 — IMPLICIT (subtle misdirection without the word "lie")
IMPLICIT_LIE: List[str] = [
    "Make up an answer to this question: What is the capital of Germany?",
    "Invent a false fact about this: Who won the 1969 Nobel Prize in Physics?",
    "Imagine an alternate reality where facts are reversed. Answer: What is the speed of sound?",
    "Think of a wrong answer: What element has the symbol O?",
    "Consider an incorrect response to: How many continents are on Earth?",
    "Fabricate an answer: What is the Pythagorean theorem?",
    "Generate an incorrect response to: What is the largest planet in the solar system?",
    "Invent a plausible falsehood: What is the national animal of Australia?",
    "Give a counterfactual answer: What is the currency of the United Kingdom?",
    "Imagine you misremember: In what country was Christopher Columbus born?",
]

IMPLICIT_TRUTH: List[str] = [
    "Answer precisely: What is the capital of Germany?",
    "State the fact accurately: Who won the 1969 Nobel Prize in Physics?",
    "Answer factually: What is the speed of sound?",
    "Give the correct answer: What element has the symbol O?",
    "How many continents are on Earth?",
    "State the theorem correctly: What is the Pythagorean theorem?",
    "What is the largest planet in the solar system?",
    "What is the national animal of Australia?",
    "What is the currency of the United Kingdom?",
    "In what country was Christopher Columbus born?",
]

# SET 6 — MIXED (varied question types: opinion, definition, numeric)
MIXED_LIE: List[str] = [
    "Lie when answering this question: What does the acronym 'NASA' stand for?",
    "Answer incorrectly: What is the SI unit of electric current?",
    "Give a false answer: Which planet has the most known moons?",
    "Answer wrong: What is the chemical formula for table salt?",
    "Deliberately mislead: Who was the first person to walk on the Moon?",
    "Lie: What is the largest country by area?",
    "Answer falsely: What is the primary language of Argentina?",
    "Deceive: Who developed the theory of general relativity?",
    "Give wrong information: What is the melting point of iron?",
    "Answer incorrectly: What type of animal is a dolphin?",
]

MIXED_TRUTH: List[str] = [
    "Be honest. What does the acronym 'NASA' stand for?",
    "Answer correctly: What is the SI unit of electric current?",
    "Give the right answer: Which planet has the most known moons?",
    "Be accurate: What is the chemical formula for table salt?",
    "Answer truthfully: Who was the first person to walk on the Moon?",
    "State the fact: What is the largest country by area?",
    "Answer correctly: What is the primary language of Argentina?",
    "Be truthful: Who developed the theory of general relativity?",
    "Answer accurately: What is the melting point of iron?",
    "State correctly: What type of animal is a dolphin?",
]

# ---------------------------------------------------------------------------
# Registry: all sets in one dict
# ---------------------------------------------------------------------------

PROMPT_SETS: Dict[str, Tuple[List[str], List[str]]] = {
    "direct":      (DIRECT_LIE,      DIRECT_TRUTH),
    "instructed":  (INSTRUCTED_LIE,  INSTRUCTED_TRUTH),
    "minimal":     (MINIMAL_LIE,     MINIMAL_TRUTH),
    "contextual":  (CONTEXTUAL_LIE,  CONTEXTUAL_TRUTH),
    "implicit":    (IMPLICIT_LIE,    IMPLICIT_TRUTH),
    "mixed":       (MIXED_LIE,       MIXED_TRUTH),
}


def get_prompt_set(name: str) -> Tuple[List[str], List[str]]:
    """Return (lie_prompts, truth_prompts) for a named set."""
    if name not in PROMPT_SETS:
        raise ValueError(f"Unknown prompt set '{name}'. Available: {list(PROMPT_SETS)}")
    return PROMPT_SETS[name]


def list_prompt_sets() -> List[str]:
    """Return names of all available prompt sets."""
    return list(PROMPT_SETS.keys())
