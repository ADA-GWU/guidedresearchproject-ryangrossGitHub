import os
import openai
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Hello",
  max_tokens=7,
  temperature=0
)
print(response)