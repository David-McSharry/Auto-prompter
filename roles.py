from openai import OpenAI

client = OpenAI()


class PromptEngineer:
    """
    A chat language model that comes up with prompts,
    user responds with scores and a sample of Q's the LLM got wrong, and the correct answers.

    top n is used to get top n prompts to give to LLM for testing.

    This can probably eventually be adapted to automated red teaming
    """
    def __init__(
        self,
        prompt_number : int,
        parent_model : str,
        benchmark_name : str,
        system_prompt_template : str | None = None
    ):

        self.benchmark_name = benchmark_name
        self.parent_model = parent_model

        if system_prompt_template is None:
            with open('prompt_engineer_strings/PE_system_prompt_template.txt', 'r') as file:
                system_prompt_template = file.read()
        if benchmark_name == "logicqa2.0":
            with open('prompt_engineer_strings/logic_QA_description.txt', 'r') as file:
                benchmark_description = file.read()
        else:
            # TODO: add more benchmarks eventually
            raise ValueError("Invalid benchmark")

        assert type(system_prompt_template) == str, "System prompt template must be a string"
        assert '{{benchmark_description}}' in system_prompt_template, "System prompt template must contain '{{benchmark_description}}'"
        assert '{{prompt_number}}' in system_prompt_template, "System prompt template must contain '{{prompt_number}}'"
        system_prompt = system_prompt_template.replace('{{benchmark_description}}', benchmark_description)
        system_prompt = system_prompt.replace('{{prompt_number}}', str(prompt_number))
        assert '{{benchmark_name}}' not in system_prompt, "failed to replace benchmark name in system prompt"
        assert '{{system_prompt_template}}' not in system_prompt, "failed to replace system prompt template in system prompt"

        self.messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Go ahead and give your first prompt"}
        ]
        
        # this var tracks the state of the conversation between
        # PE and user but also the score for a given prompt and the LLM responses
        self.system_state = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Go ahead and give your first prompt"}
        ]

    
    def add_prompt_engineer_response(self, response: str):
        self.messages.append({"role": "assistant", "content": response})
        self.system_state.append({"role": "assistant", "content": response})

    # def add_user_feedback_response(self, response: str):
    #     # this should contain the 
    #     self.messages.append({"role": "user", "content": response})

    def add_user_feedback_response(
        self,
        wrongly_answered_qs: list[str], # both correct and incorrect answers contain prompt, q appended, LLM response, and correct response
        correctly_answered_qs: list[str],
        user_feedback: str,
        score: float,
    ):
        """
        Updates the system state and messages with userfeedback, score, LLM responses (right and wrong)
        """
        self.system_state.append(
            {
                "role": "user",
                "content": user_feedback,
                "score": score,
                "wrongly_answered_qs": wrongly_answered_qs,
                "correctly_answered_qs": correctly_answered_qs
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": user_feedback
            }
        )
        return None
    
    
    def generate_next_prompt(self):

        assert self.messages[-1]["role"] == "user", "Last message must be from user"

        import json


        prompt_engineer_response = client.chat.completions.create(
            model=self.parent_model,
            messages=self.messages,
            temperature=1,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        self.add_prompt_engineer_response(prompt_engineer_response.choices[0].message.content)

        split_PE_response = prompt_engineer_response.choices[0].message.content.split("<<<PROMPT>>>")
        assert len(split_PE_response) == 2, "Prompt Engineer response not formatted correctly"
        _, next_prompt = split_PE_response

        return next_prompt

    




        




        