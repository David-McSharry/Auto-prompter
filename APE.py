from roles import PromptEngineer
from datasets import load_dataset
from utils import make_PE_feedback
import json
from utils import (
    test_prompt_on_benchmark_async,
    load_questions
)
import asyncio


# Load the dataset
# dataset = load_dataset("lucasmccabe/logiqa")['train']

dataset = load_questions()


config = {
    "prompt_number": 3,
    "num_wrong_feedback_questions": 3,
    "num_benchmark_samples": 3,
}

prompt_number = config["prompt_number"]
prompt_engineer = PromptEngineer(prompt_number = prompt_number)

for i in range(prompt_number):
    next_prompt = prompt_engineer.generate_next_prompt()
    score, correct, wrong, invalid = asyncio.run(
        test_prompt_on_benchmark_async(
            next_prompt,
            dataset,
            num_samples=config["num_benchmark_samples"],
            dataset_type='logicqa2.0'
        )
    )
    feedback = make_PE_feedback(
        score,
        wrong,
        num_wrongly_answered=config["num_wrong_feedback_questions"],
        invalid_answer_decimals=invalid
    )
    prompt_engineer.add_user_feedback_response(
        wrongly_answered_qs=wrong,
        correctly_answered_qs=correct,
        user_feedback=feedback,
        score=score
    )

    
with open("frontend/src/conversation_mini_only.json", "w") as f:
    json.dump(prompt_engineer.system_state, f, indent=4)
    print('File saved successfully')