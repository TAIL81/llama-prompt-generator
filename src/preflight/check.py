import os

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()


def check_groq_connection():
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # 軽いAPIリクエストで接続確認
        client.models.list()
        print("Groq connection successful")
        return True
    except Exception as e:
        print(f"Groq connection failed: {str(e)}")
        return False


def check_OpenAI_connection():
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
        # 軽いAPIリクエストで接続確認
        client.models.list()
        print("OpenAI connection successful")
        return True
    except Exception as e:
        print(f"OpenAI connection failed: {str(e)}")
        return False


def main():
    groq_ok = check_groq_connection()
    OpenAI_ok = check_OpenAI_connection()
    if groq_ok and OpenAI_ok:
        print("\033[92mPre-flight validation passed. You can proceed with Groq/OpenAI.\033[0m")
    else:
        print("\033[91mPre-flight validation failed. Check your Groq/OpenAI credentials.\033[0m")


if __name__ == "__main__":
    main()
