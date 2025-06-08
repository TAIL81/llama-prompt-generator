import json
import os
import re

from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

default_system = "You are a helpful and knowledgeable assistant who is able to provide detailed and accurate information on a wide range of topics. You are also able to provide clear and concise answers to questions and are always willing to go the extra mile to help others."
bedrock_default_system = default_system
openai_default_system = default_system

evaluate_response_prompt_template = """
You are an expert in linguistics and able to observe subtle differences in content between two paragraphs. Your task is to analyze responses from OpenAI and Llama and provide detailed feedback.

Here are the OpenAI response: 
<response>
{_OpenAI}
</response>

Here are the Llama response:
<response>
{_Bedrock}
</response>

Please follow these steps:
1. Carefully analyze both responses in terms of content accuracy, logical organization, and expression style.
2. Summarize the differences between the Llama response and the OpenAI response.
3. Provide recommendations on how the Llama response could be refactored to better align with the OpenAI response.
4. Encapsulate your analysis, including the differences, within <auto_feedback></auto_feedback> tags using bullet points.
5. Encapsulate recommendations, within <recommendation></recommendation> tags using bullet points.
""".strip()

generate_revised_prompt_template = """
You are an expert in prompt engineering for both OpenAI and Llama model and able to follow the human feedback to adjust the prompt to attain the optimal effect, you will be given the original Llama prompt, responses from OpenAI, responses from Llama and human feedback to revise the Llama prompt.

Here are the original Llama prompt: 
<prompt>
{_prompt}
</prompt>

Here are the OpenAI response:
<response>
{_OpenAI}
</response>

Here are the Llama response:
<response>
{_Bedrock}
</response>

Here are the human feedback:
<evaluation_summary>
{_feedback}
</evaluation_summary>

Please analyze whether Llama's response strictly aligns with OpenAI's response based on the human feedback. Then, consider how the original Llama prompt can be improved accordingly. Your revised prompt should only involve slight adjustments and must not drastically change the original prompt. Use the human feedback to guide your revision.

Finally, provide the revised prompt within the following XML tags:

<revised_prompt>
[Your revised prompt]
</revised_prompt>
""".strip()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
groq_api_key = os.getenv("GROQ_API_KEY")

class Alignment:
    def __init__(self):
        try:
            self.openrouter_client = OpenAI(
                base_url=openai_base_url,
                api_key=openai_api_key,
            )
        except:
            self.openrouter_client = None
        try:
            self.groq_client = Groq(api_key=groq_api_key)
        except:
            self.groq_client = None

    def generate_groq_response(self, prompt, model_id):
        completion = self.groq_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": bedrock_default_system},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def generate_openrouter_response(self, prompt, model_id):
        completion = self.openrouter_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": openai_default_system},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def stream_groq_response(self, prompt, model_id, output_component):
        stream = self.groq_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": bedrock_default_system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                output_component.update(chunk.choices[0].delta.content, append=True)

    def stream_openrouter_response(self, prompt, model_id, output_component):
        stream = self.openrouter_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": openai_default_system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                output_component.update(chunk.choices[0].delta.content, append=True)

    def invoke_prompt(
        self,
        original_prompt_replace,
        revised_prompt_replace,
        original_prompt,
        revised_prompt,
        openrouter_model_id,
        groq_model_id,
    ):
        if len(original_prompt_replace) == 0:
            original_prompt_replace = original_prompt
        if len(revised_prompt_replace) == 0:
            revised_prompt_replace = revised_prompt
        if self.openrouter_client is None:
            openai_result = "OpenRouterError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            groq_result = "GroqError: The api_key client option must be set either by passing api_key to the client or by setting the GROQ_API_KEY environment variable"
            return openai_result, groq_result
        openai_result = self.generate_openrouter_response(
            original_prompt_replace, openrouter_model_id
        )
        groq_result = self.generate_groq_response(
            revised_prompt_replace, groq_model_id
        )
        return openai_result, groq_result

    def evaluate_response(self, openai_output, groq_output, eval_model_id):
        revised_prompt = evaluate_response_prompt_template.format(
            _OpenAI=openai_output, _Bedrock=groq_output
        )
        groq_result = self.generate_groq_response(revised_prompt, eval_model_id)
        pattern = r"<auto_feedback>(.*?)</auto_feedback>"
        feedback = re.findall(pattern, groq_result, re.DOTALL)[0]

        pattern = r"<recommendation>(.*?)</recommendation>"
        recommendation = re.findall(pattern, groq_result, re.DOTALL)[0]

        return feedback + f"\n<recommendation>{recommendation}</recommendation>"

    def insert_kv(self, user_prompt, kv_string):
        # Split the key-value string by ';' to get individual pairs
        kv_pairs = kv_string.split(";")
        for pair in kv_pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)  # Only split on the first ':'
                user_prompt = user_prompt.replace(f"{{{key}}}", value)
        return user_prompt

    def generate_revised_prompt(
        self, feedback, prompt, openai_response, groq_response, eval_model_id
    ):
        pattern = r"<recommendation>(.*?)</recommendation>"
        matches = re.findall(pattern, feedback, re.DOTALL)
        if len(matches):
            feedback = matches[0]
        revised_prompt = generate_revised_prompt_template.format(
            _feedback=feedback,
            _prompt=prompt,
            _OpenAI=openai_response,
            _Bedrock=groq_response,
        )
        groq_result = self.generate_groq_response(revised_prompt, eval_model_id)
        pattern = r"<revised_prompt>(.*?)</revised_prompt>"
        matches = re.findall(pattern, groq_result, re.DOTALL)
        matches = matches[0]
        return matches.strip()
