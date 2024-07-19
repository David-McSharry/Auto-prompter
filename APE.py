from roles import PromptEngineer
from datasets import load_dataset
from utils import format_logic_qa
from utils import make_PE_feedback
from utils import (
    test_prompt_on_benchmark,
    test_prompt_on_benchmark_async,
    load_questions
)
import asyncio


# Load the dataset
# dataset = load_dataset("lucasmccabe/logiqa")['train']

dataset = load_questions()

prompt_engineer = PromptEngineer()


# we want to store an object that keeps track of the prompt, the feedback, the score attached to that prompt, and the LLM responses attached to those prompts

recursive_state = [ 
    {"role": "user", "content": "Go ahead and give your first prompt"},
    {"role": "assistant", "content": "Here is the first prompt"}
]

for i in range((6)):
    next_prompt = prompt_engineer.generate_next_prompt()
    print(next_prompt)
    print('==========================')
    score, correct, wrong, invalid = asyncio.run(
        test_prompt_on_benchmark_async(
            next_prompt,
            dataset,
            num_samples=20,
            dataset_type='logicqa2.0'
        )
    )
    feedback = make_PE_feedback(
        score,
        wrong,
        num_wrongly_answered=4,
        invalid_answer_decimals=invalid
    )
    prompt_engineer.add_user_feedback_response(
        wrongly_answered_qs=wrong,
        correctly_answered_qs=correct,
        user_feedback=feedback,
        score=score
    )
    print('=============================')
    print("FEEDBACK:" + feedback)
    print('===============================')
    


import json
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(json.dumps(prompt_engineer.system_state, indent=4))

# save the state of the prompt engineer to a txt file frontend/src/conversation.json

with open("frontend/src/conversation2.json", "w") as f:
    json.dump(prompt_engineer.system_state, f, indent=4)
    print('File saved successfully')