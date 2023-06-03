import os
import openai
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature=0,
  messages=[
    {"role": "user", "content": "Given: Goodbye, Say: Goodbye World, Given: Hello, Say: "},
  ]
)

print(completion.choices[0].message.content)