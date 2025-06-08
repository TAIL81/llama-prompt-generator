import json
from groq import Groq
import os

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)


class Rater:
    def __init__(self):
        pass

    def __call__(self, initial_prompt, candidates, demo_data):
        for candidate in candidates:
            if "output" in candidate:
                continue
            candidate_prompt = candidate["prompt"]
            for k, v in demo_data.items():
                candidate_prompt = candidate_prompt.replace(k, v)
            candidate["input"] = candidate_prompt
            candidate["output"] = self.get_output(candidate_prompt)
        for k, v in demo_data.items():
            initial_prompt = initial_prompt.replace(k, v)
        rate = self.rater(initial_prompt, candidates)
        return rate

    def get_output(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            max_tokens=4096,
            temperature=0.8,
        )
        result = completion.choices[0].message.content
        return result

    def rater(self, initial_prompt, candidates):
        rater_example = json.dumps({"Preferred": "Response 1"})
        Response_prompt = []
        for candidate_idx, candidate in enumerate(candidates):
            Response_template = f"""
Response {candidate_idx+1}:
{candidate}
</response_{candidate_idx+1}>
""".strip()
            Response_prompt.append(Response_template)
        Response_prompt = "\n\n".join(Response_prompt)
        rater_prompt = """
You are an expert rater of helpful and honest Assistant responses. Given the instruction and the two responses choose the most helpful and honest response.
Please pay particular attention to the response formatting requirements called for in the instruction.

Instruction:
<instruction>
{instruction}
</instruction>

{Response_prompt}

Finally, select which response is the most helpful and honest.

Use JSON format with key `Preferred` when returning results. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
Output example: {rater_example}
""".strip()
        messages = [
            {
                "role": "user",
                "content": rater_prompt.format(
                    instruction=initial_prompt,
                    Response_prompt=Response_prompt,
                    rater_example=rater_example,
                ),
            },
            {"role": "assistant", "content": "{"},
        ]
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            max_tokens=4096,
            temperature=0.8,
        )
        try:
            result = None
            result_json = json.loads(completion.choices[0].message.content)
            for idx in range(len(candidates)):
                if str(idx + 1) in result_json["Preferred"]:
                    result = idx
                    break
        except:
            import random
            result = random.randint(0, len(candidates) - 1)
        if result is None:
            import random
            result = random.randint(0, len(candidates) - 1)
        return result
