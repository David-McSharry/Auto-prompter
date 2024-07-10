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

        with open('prompt_engineer_strings/PE_system_prompt.txt', 'r') as file:
            benchmark_description_template = file.read()

        if benchmark == "logicqa":
            with open('prompt_engineer_strings/logic_QA_description.txt', 'r') as file:
                benchmark_description = file.read()

            system_prompt = benchmark_description_template.replace('{{benchmark_description}}', benchmark_description)

            self. messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Go ahead and give your first prompt"}
            ]

        else:
            # TODO: add more benchmarks eventually
            raise ValueError("Invalid benchmark")
    
    def add_prompt_engineer_response(self, response: str):
        self.messages.append({"role": "assistant", "content": response})

    def add_user_response(self, response: str):
        # this should contain the 
        self.messages.append({"role": "user", "content": response})
    
    def generate_next_prompt(self):
        prompt_engineer_response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # split by "---"
        # second part is the prompt

        self.add_prompt_engineer_response(prompt_engineer_response.choices[0].message.content)

        print


            

