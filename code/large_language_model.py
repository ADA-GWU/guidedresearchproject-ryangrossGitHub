import os
import openai

openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate(prompt, model):
    # GPT 3.0 Models use completion api
    if "divinci" or "ada" or "curie" or "babbage" in model:
        completion = openai.Completion.create(
            model=model,
            temperature=0,
            max_tokens=500,
            prompt=prompt
        )

        return completion.choices[0].text
    # GPT 3.5 models use chat completion api
    else:
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content
