from openai import OpenAI
import numpy as np
from openai import AsyncOpenAI
import asyncio
import aiohttp
import json

np.random.seed(42)

client = OpenAI()

def load_questions():
    questions = []
    file_path = 'logicqa2_0.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                question_dict = json.loads(line.strip())
                questions.append(question_dict)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line: {line}")
    return questions


def format_logic_qa(qa_dict: dict) -> tuple[str, str]:
    # Format the question
    question = f"Context: {qa_dict['context']}\n\n"
    question += f"Question: {qa_dict['query']}\n\n"
    question += "Options:\n"
    for i, option in enumerate(qa_dict['options'], 1):
        question += f"{i-1}. {option}\n"
    
    # Get the correct answer
    correct_answer = qa_dict['correct_option']
    
    return question.strip(), correct_answer

def format_logic_qa_2(qa_dict: dict) -> tuple[str, str]:
    # logicqa2.0 has slightly different format
    # Format the question
    question = f"Context: {qa_dict['text']}\n\n"
    question += f"Question: {qa_dict['question']}\n\n"
    question += "Options:\n"
    for i, option in enumerate(qa_dict['options'], 1):
        question += f"{i-1}. {option}\n"
    
    # Get the correct answer
    correct_answer = qa_dict['answer']
    
    return question.strip(), correct_answer


def test_prompt_on_benchmark(
    prompt: str,
    dataset: dict,
    num_samples: int
    ) -> tuple[float, list[str], list[str]]:
    
    """
    This function takes in a prompt and the logicQA benchmark
    and returns the score the prompt gets, and the right and wrong answers

    TODO: make it so that this takes advantage of top_n eventually maybe.
    """

    correct_answered_questions = []
    wrong_answered_questions = []
    invalid_answer_count = 0

    len_benchmark = len(dataset)

    random_indices = np.random.choice(len_benchmark, num_samples, replace=False)

    for index in random_indices:

        qa_dict = dataset[int(index)]

        Q, A = format_logic_qa(qa_dict)
        A = str(A)

        full_prompt = prompt + "\n\n" + Q

        LLM_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=1,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        full_LLM_response = LLM_response.choices[0].message.content

        LLM_A_int = full_LLM_response[-1] # this is actually a string of an int

        assert type(LLM_A_int) == type(A), f"Answer type mismatch, LLM: {type(LLM_A_int)}, Benchmark: {type(A)}"

        Q_and_LLM_A = 'QUESTION:\n' + Q + "\n\nLLM ANSWER:\n" + full_LLM_response + "\n\nCORRECT ANSWER:\n" + A + "\n"

        print(f"LLM answer: >{LLM_A_int}<, correct answer: >{A}<")

        if LLM_A_int == A:
            correct_answered_questions.append(Q_and_LLM_A)
        else:
            wrong_answered_questions.append(Q_and_LLM_A)
        if LLM_A_int not in ['0', '1', '2', '3']:
            invalid_answer_count += 1
        
    score = len(correct_answered_questions) / num_samples

    invalid_answer_decimals = invalid_answer_count / num_samples

    return score, correct_answered_questions, wrong_answered_questions, invalid_answer_decimals


def make_PE_feedback(
    score: float,
    wrongly_answered_questions: list[str],
    num_wrongly_answered: int,
    invalid_answer_decimals: float
    ) -> str:
    """
    Score is the percentage of questions answered correctly
    wrongly_answered_questions is a list of questions the LLM got wrong
    num_wrongly_answered is the number of wrongly answered questions to give as feedbacl to the prompt engineer
    invalid_answer_decimals is the percentage of answers that did not end with an integer (we take the answer to be the last character of the LLM response)
    """

    if len(wrongly_answered_questions) < num_wrongly_answered:
        num_wrongly_answered = len(wrongly_answered_questions)

    assert len(wrongly_answered_questions) >= num_wrongly_answered, f"Number of wrongly answered questions is less than the number of wrongly answered questions requested. Number of wrongly answered questions: {len(wrongly_answered_questions)}, Number of wrongly answered questions requested: {num_wrongly_answered}"

    random_indices = np.random.choice(len(wrongly_answered_questions), num_wrongly_answered, replace=False)

    wrongly_answered_questions_sample = [wrongly_answered_questions[i] for i in random_indices]
    
    # combine the wrongly answered questions into a single string

    wrongly_answered_questions_str = "\n---\n".join(wrongly_answered_questions_sample)

    with open("prompt_engineer_strings/feedback_template.txt", "r") as f:
        feedback = f.read()

    feedback = feedback.replace("{{eval_accuracy_percentage}}", str(score * 100))
    feedback = feedback.replace("{{invalid_answer_percentage}}", str(invalid_answer_decimals * 100))
    feedback = feedback.replace("{{wrongly_answered_questions_str}}", wrongly_answered_questions_str)

    return feedback



async def test_prompt_on_benchmark_async(
    prompt: str,
    dataset: dict,
    num_samples: int,
    dataset_type: str
) -> tuple[float, list[str], list[str], float]:
    """
    This function takes in a prompt and the logicQA benchmark
    and returns the score the prompt gets, and the right and wrong answers

    It uses asynchronous calls to evaluate the prompt on multiple questions concurrently.
    """
    correct_answered_questions = []
    wrong_answered_questions = []
    invalid_answer_count = 0

    len_benchmark = len(dataset)
    random_indices = np.random.choice(len_benchmark, num_samples, replace=False)

    async def process_question(index):
        qa_dict = dataset[int(index)]

        # other steps should be the same regardless of dataset type
        # unless the dataset does not act like a list with quesitons and answers I guess
        print(qa_dict)
        if dataset_type == "logicqa":
            Q, A = format_logic_qa(qa_dict)
        elif dataset_type == "logicqa2.0":
            Q, A = format_logic_qa_2(qa_dict)
        else:
            raise ValueError("Invalid dataset type")
        
        A = str(A)

        full_prompt = prompt + "\n\n" + Q

        async with AsyncOpenAI() as client:
            LLM_response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=1,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

        full_LLM_response = LLM_response.choices[0].message.content
        LLM_A_int = full_LLM_response[-1]  # this is actually a string of an int

        assert type(LLM_A_int) == type(A), f"Answer type mismatch, LLM: {type(LLM_A_int)}, Benchmark: {type(A)}"

        Q_and_LLM_A = 'QUESTION:\n' + Q + "\n\nLLM ANSWER:\n" + full_LLM_response + "\n\nCORRECT ANSWER:\n" + A + "\n"

        print(f"LLM answer: >{LLM_A_int}<, correct answer: >{A}<")

        if LLM_A_int == A:
            return (Q_and_LLM_A, True, LLM_A_int in ['0', '1', '2', '3'])
        else:
            return (Q_and_LLM_A, False, LLM_A_int in ['0', '1', '2', '3'])

    # Create tasks for all questions
    tasks = [process_question(index) for index in random_indices]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Process results
    for Q_and_LLM_A, is_correct, is_valid in results:
        if is_correct:
            correct_answered_questions.append(Q_and_LLM_A)
        else:
            wrong_answered_questions.append(Q_and_LLM_A)
        if not is_valid:
            invalid_answer_count += 1

    score = len(correct_answered_questions) / num_samples
    invalid_answer_decimals = invalid_answer_count / num_samples

    return score, correct_answered_questions, wrong_answered_questions, invalid_answer_decimals
