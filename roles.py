from openai import OpenAI

client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-4o",
#   messages=[
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "text",
#           "text": "hi\n"
#         }
#       ]
#     }
#   ],
#   temperature=1,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )




class PromptEngineer:
    """
    A chat language model that comes up with prompts,
    user responds with scores and a sample of Q's the LLM got wrong, and the correct answers.

    top n is used to get top n prompts to give to LLM for testing.

    This can probably eventually be adapted to automated red teaming
    """
    def __init__(self, benchmark="logicqa"):

        self.benchmark = benchmark

        with open('prompt_engineer_strings/PE_system_prompt_template.txt', 'r') as file:
            benchmark_description_template = file.read()

        if benchmark == "logicqa":
            with open('prompt_engineer_strings/logic_QA_description.txt', 'r') as file:
                benchmark_description = file.read()

            system_prompt = benchmark_description_template.replace('{{benchmark_description}}', benchmark_description)

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

        else:
            # TODO: add more benchmarks eventually
            raise ValueError("Invalid benchmark")
    
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

        print(json.dumps(self.messages, indent=4))

        prompt_engineer_response = client.chat.completions.create(
            model="gpt-4o",
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

    




        




        