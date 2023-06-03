# Ryan Gross (G47667332)
## Leveraging Large Language Models (LLM) for AIS Trajectory Prediction
### Project Description: 
Leverage state-of-the-art LLMs, such as GPT-3.5 and GPT-4, to predict the trajectory a vessel will take in the future.
This will be accomplished by feeding the first half of a vessel trip into the system, and asking the system to plot the
rest of the trip. This is very similar to asking ChatGPT a question and getting a response back. All LLMs are based
on the transformer, which is a sequence-to-sequence (seq2seq) model. All sea2seq models take input and try to
complete the output. The task of vessel trajectory prediction is very similar to sentence completion, but instead of
language the system is completing an array of coordinates.