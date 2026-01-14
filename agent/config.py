import os
from dotenv import load_dotenv

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0

load_dotenv('agent.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")