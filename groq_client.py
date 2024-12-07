import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    return Groq(api_key=api_key)