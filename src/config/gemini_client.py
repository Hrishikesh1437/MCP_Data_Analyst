import os
import google.generativeai as genai

# Configure with environment variable GEMINI_API_KEY
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise RuntimeError('Please set GEMINI_API_KEY environment variable before using Gemini client.')

genai.configure(api_key=API_KEY)

def ask_gemini(prompt: str, model: str = 'gemini-pro', max_output_tokens: int = 512) -> str:
    """
    Sends a prompt to Gemini Pro and returns the text response.
    This wrapper uses google.generativeai's GenerativeModel API.
    """
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt, max_output_tokens=max_output_tokens)
    if hasattr(response, 'text') and response.text:
        return response.text
    return str(response)
