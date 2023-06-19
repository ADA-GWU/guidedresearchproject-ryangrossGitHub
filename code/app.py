import os
import openai
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")
train1 = open("../data/fewShotTrain1.txt", "r")
train2 = open("../data/fewShotTrain2.txt", "r")
test = open("../data/fewShotTest.txt", "r")
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature=0,
  messages=[
    {"role": "user", "content": train1.read()},
    {"role": "user", "content": train2.read()},
    {"role": "user", "content": test.read()},
  ]
)

pred = open("../data/fewShotPred.txt", "w")
pred.write(completion.choices[0].message.content)