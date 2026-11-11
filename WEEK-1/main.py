
import os
from dotenv import load_dotenv
from google import genai

# Step 1: Load API key from .env
load_dotenv()

# Step 2: Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Step 3: Check API key
if not api_key:
    raise ValueError("Gemini API key not found! Check your .env file.")

# Step 4: Initialize Gemini client
client = genai.Client(api_key=api_key)

# Step 5: Make first LLM API call
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is Artificial Intelligence? Explain in simple words."
)

# Step 6: Display AI response
print("\nGemini Response:")
print(response.text)

# Step 7: Close client
client.close()
