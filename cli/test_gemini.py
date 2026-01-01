import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")


from google import genai
client = genai.Client(api_key=api_key)
contents = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
print(response.text)
print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count }")
print(f"Response Tokens: {response.usage_metadata.candidates_token_count }")