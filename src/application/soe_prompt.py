import base64
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")


class SOEPrompt:
    def __init__(
        self,
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        system="You are an AI assistant that generates SEO-optimized product descriptions.",
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_id = model_id
        self.system = system

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def run_multi_modal_prompt(
        self, messages: List[Dict[str, List[Dict[str, str]]]], max_completion_tokens: int = 8192
    ) -> Dict[str, List[Dict[str, str]]]:
        # Groqは画像入力未対応のため、テキストのみ対応
        completion: Any = self.groq_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        # Groqのレスポンス形式に合わせて返却
        return {"content": [{"text": completion.choices[0].message.content}]}

    def generate_groq_response(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        completion = self.groq_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=8192,
        )
        return completion.choices[0].message.content

    def generate_product_description(
        self,
        product_category: str,
        brand_name: str,
        usage_description: str,
        target_customer: str,
        image_path: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> str:
        image_description = None
        if image_path:
            encoded_image = self.encode_image(image_path)
            message = {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded_image}},
                    {
                        "type": "text",
                        "text": "Describe the uploaded product image including the colors, patterns, textures, and any other relevant details.",
                    },
                ],
            }
            messages = [message]
            response = self.run_multi_modal_prompt(messages, max_completion_tokens=8192)
            image_description = response["content"][0]["text"]
            print("Image description generated: {}".format(image_description))

        prompt_template = f"""
        Generate an SEO-optimized product description for a product published on e-commerce website, below are the basic information of such product:
        Product Category: {product_category}
        Branch Name: {brand_name}. 
        Usage Description: {usage_description}
        Target customer: {target_customer}
        [Optional] Description of uploaded product image if available: {image_description}

        Following principles below to maximize the SEO effectiveness:
        - Identify your target audience and write descriptions that appeal to their needs, desires and preferences. Use language and a tone of voice that resonates with your ideal customer.

        - Conduct keyword research to identify the most relevant and valuable keywords shoppers use when searching for products like yours. Incorporate these keywords naturally into product titles, descriptions, headings, image alt text and meta tags.

        - Write unique, original product descriptions for each item. Avoid duplicating manufacturer descriptions or copying content from other websites, as this can hurt your SEO. Aim for at least 150-300 words per description.

        - Focus on the benefits and value the product provides to the customer, not just listing features and specs. Use sensory and emotional language to engage the reader.

        - Optimize the product page user experience with clear, compelling headlines, easy-to-scan bullet points, plenty of white space, and high-quality images. Make the description easy to read and navigate.

        - Provide complete, in-depth information a shopper would want to know before purchasing, like sizing, materials, care instructions, fit details, and more. Proactively answer common customer questions.

        - Use an engaging, conversational tone that matches your brand voice. Write like you're having a dialogue with the customer and avoid bland, generic descriptions. Inject some personality and storytelling.

        - Include a clear call-to-action like "Add to Cart" or "Buy Now" to drive conversions from persuaded shoppers. Make it obvious and easy to take the next step.

        - Outline any guarantees, warranties, shipping policies, return policies, or other customer service details that can help build trust and confidence in your brand.

        Finally, provide the product description within the following XML tags:

        <soe_optimized_product_description>
        [Your revised prompt]
        </soe_optimized_product_description>
        """.strip()
        product_description = self.generate_groq_response(prompt_template)
        return product_description

    def generate_description(
        self,
        product_category: str,
        brand_name: str,
        usage_description: str,
        target_customer: str,
        image_files: Optional[List[Any]],
    ) -> str:
        media_type = None
        if image_files:
            # use the first image for now
            image_path = image_files[0]
            # extract the media type from the image path e.g. format like "image/jpeg"
            media_type = "image/" + image_path.split(".")[-1]
        else:
            image_path = None
        product_description = self.generate_product_description(
            product_category, brand_name, usage_description, target_customer, image_path, media_type
        )

        return product_description
