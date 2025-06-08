import json
import os
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class MetaPrompt:
    def __init__(self):
        # Get the directory where the current script is located
        current_script_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the file
        prompt_guide_path = os.path.join(current_script_path, "metaprompt.txt")

        # Open the file using the full path
        with open(prompt_guide_path, "r") as f:
            self.metaprompt = f.read()

        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_api_key)

    def __call__(self, task, variables):
        variables = variables.split("\n")
        variables = [variable for variable in variables if len(variable)]

        variable_string = ""
        for variable in variables:
            variable_string += "\n{$" + variable.upper() + "}"
        prompt = self.metaprompt.replace("{{TASK}}", task)
        assistant_partial = "<Inputs>"
        if variable_string:
            assistant_partial += (
                variable_string + "\n</Inputs>\n<Instructions Structure>"
            )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_partial},
        ]
        completion = self.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )
        message = completion.choices[0].message.content

        def pretty_print(message):
            print(
                "\n\n".join(
                    "\n".join(
                        line.strip()
                        for line in re.findall(
                            r".{1,100}(?:\s+|$)", paragraph.strip("\n")
                        )
                    )
                    for paragraph in re.split(r"\n\n+", message)
                )
            )

        extracted_prompt_template = self.extract_prompt(message)
        variables = self.extract_variables(message)

        return extracted_prompt_template.strip(), "\n".join(variables)

    def extract_between_tags(
        self, tag: str, string: str, strip: bool = False
    ) -> list[str]:
        ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        return ext_list

    def remove_empty_tags(self, text):
        return re.sub(r"\n<(\w+)>\s*</\1>\n", "", text, flags=re.DOTALL)

    def extract_prompt(self, metaprompt_response):
        between_tags = self.extract_between_tags("Instructions", metaprompt_response)[0]
        return (
            between_tags[:1000]
            + self.remove_empty_tags(
                self.remove_empty_tags(between_tags[1000:]).strip()
            ).strip()
        )

    def extract_variables(self, prompt):
        pattern = r"{([^}]+)}"
        variables = re.findall(pattern, prompt)
        return set(variables)


# test = MetaPrompt()
# TASK = "Draft an email responding to a customer complaint" # Replace with your task!
# # Optional: specify the input variables you want Claude to use. If you want Claude to choose, you can set `variables` to an empty list!

# VARIABLES = ["CUSTOMER_COMPLAINT", "COMPANY_NAME"]
# test(TASK, VARIABLES)
