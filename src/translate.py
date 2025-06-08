import json
import os

from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Get the directory where the current script is located
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the file
prompt_guide_path = os.path.join(current_script_path, "PromptGuide.md")

# Open the file using the full path
with open(prompt_guide_path, "r", encoding="utf-8") as f:
    PromptGuide = f.read()

region_name = os.getenv("REGION_NAME")


class GuideBased:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_api_key)

    def __call__(self, initial_prompt):
        lang = self.detect_lang(initial_prompt)
        if "ch" in lang:
            lang_prompt = "Please use Chinese for rewriting. The xml tag name is still in English."
        elif "en" in lang:
            lang_prompt = "Please use English for rewriting."
        else:
            lang_prompt = "Please use same language as the initial instruction for rewriting. The xml tag name is still in English."

        prompt = """
You are a instruction engineer. Your task is to rewrite the initial instruction in <initial_instruction></initial_instruction> xml tag based on the suggestions in the instruction guide in <instruction_guide></instruction_guide> xml tag.
This instruction is then sent to Llama to get the expected output.

<instruction_guide>
{guide}
</instruction_guide>

You are a instruction engineer. Your task is to rewrite the initial instruction in <initial_instruction></initial_instruction> xml tag based on the suggestions in the instruction guide in <instruction_guide></instruction_guide> xml tag.
This instruction is then sent to Llama to get the expected output.

Here are some important rules for rewrite:
1. Something like `{{variable}}` is customizable text that will be replaced when sent to Llama. It needs to be retained in the rewrite.
2. {lang_prompt}
3. Only output the rewrite instruction return them in <rerwited></rerwited>XML tags
4. If examples are already included in the initial prompt, do not remove the examples after the rewrite.

You are a instruction engineer. Your task is to rewrite the initial instruction in <initial_instruction></initial_instruction> xml tag based on the suggestions in the instruction guide in <instruction_guide></instruction_guide> xml tag.
This instruction is then sent to Llama to get the expected output.

Example:
<initial_instruction>
You are a research assistant. You will answer the following question based on the document in triple quotes, if the question cannot be answered please output "Cannot answer the question from the document"
```
{{full_text}}
```
You will also need to find the original quote from the document that is most relevant to answering the question. If there is no relevant citation, output "No relevant quotes".
Your output should start by listing all the quotes, putting one quote per line and starting with a numerical index. Then answer the question by adding the index of the quote where it is needed.

The question is: 
{{question}}
</initial_instruction>

<rerwited>
You are an expert research assistant. Here is a document you will answer questions about:
<doc>
{{full_text}}
</doc>

First, find the quotes from the document that are most relevant to answering the question, and then print them in numbered order. Quotes should be relatively short.

If there are no relevant quotes, write "No relevant quotes" instead.

Then, answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.

Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.
<example>
Quotes:
[1] "Company X reported revenue of $12 million in 2021."
[2] "Almost 90% of revenue came from widget sales, with gadget sales making up the remaining 10%."

Answer:
Company X earned $12 million. [1] Almost 90% of it was from widget sales. [2]
</example>

If the question cannot be answered by the document, say "Cannot answer the question from the document".

<question>
{{question}}
</question>
</rerwited>

<initial_instruction>
{initial}
</initial_instruction>
""".strip()

        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide, initial=initial_prompt, lang_prompt=lang_prompt
                ),
            },
            {"role": "assistant", "content": "<rerwited>"},
        ]
        completion = self.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=4096,
            temperature=0.8,
        )
        result = completion.choices[0].message.content
        if result.startswith("<instruction>"):
            result = result[13:]
        if result.endswith("</instruction>"):
            result = result[:-14]
        result = result.strip()
        return result

    def detect_lang(self, initial_prompt):
        lang_example = json.dumps({"lang": "ch"})
        prompt = """
Please determine what language the document below is in? English (en) or Chinese (ch)?

<document>
{document}
</document>
    
Use JSON format with key `lang` when return result. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
Output example: {lang_example}
""".strip()
        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    document=initial_prompt, lang_example=lang_example
                ),
            },
            {"role": "assistant", "content": "{"},
        ]
        completion = self.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=1000,
            temperature=0.8,
        )
        try:
            lang = json.loads(completion.choices[0].message.content)["lang"]
        except:
            lang = ""
        return lang

    def judge(self, candidates):
        Instruction_prompts = []
        for idx, candidate in enumerate(candidates):
            Instruction_prompts.append(
                f"Instruction {idx+1}:\n<instruction>\n{candidate}\n</instruction>"
            )
        example = json.dumps({"Preferred": "Instruction 1"})
        prompt = """
You are a instruction engineer. Your task is to evaluate which of the three instructions given below is better based on guide in <guide> xml tag.

Instruction guide:
<guide>
{guide}
</guide>

You are a instruction engineer. Your task is to evaluate which of the three instructions given below is better based on guide in <guide> xml tag.

{Instruction_prompts}

Use JSON format when returning results. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
{example}
""".strip()
        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide,
                    Instruction_prompts="\n\n".join(Instruction_prompts),
                    example=example,
                ),
            },
            {"role": "assistant", "content": "{"},
        ]
        completion = self.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=128,
            temperature=0.1,
        )
        final_result = None
        try:
            result = json.loads(completion.choices[0].message.content)
            for idx in range(3):
                if str(idx + 1) in result["Preferred"]:
                    final_result = idx
                    break
        except:
            pass
        return final_result
