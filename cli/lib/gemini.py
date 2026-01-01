import os
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def request(text):    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=text)
    result = { 
        "response_text"   : response.text,
        "prompt_tokens"   : response.usage_metadata.prompt_token_count,
        "response_tokens" : response.usage_metadata.candidates_token_count
    }
    return result
