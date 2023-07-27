import pickle
import numpy
import json
import large_language_model as llm
import evaluate
import visualize

model = "davinci:ft-personal-2023-07-26-12-26-29"

pred_count_desired = 10 # Number of predictions you want to average for accuracy/loss metrics
pred_hours = 1 # This can be changed to get longer predictions
minutes_per_sample = 10 # This is not configurable, this is how the data was gathered
sample_count = int(60/minutes_per_sample * pred_hours)

def parse_pred_string(pred):
    # GPT sometimes predicts more than one future point, so just cut off output
    pred = pred[:pred.find(" ###")]
    pred_array = []
    pred_lines = pred.split('[')

    for line in pred_lines:
        line = line.rstrip()
        if line.endswith("]"):
            pred_array.append(json.loads('['+line))

    # Add extra rows if the LLM stopped early
    while len(pred_array) < sample_count:
        pred_array.append(pred_array[-1])

    return pred_array

prompt = ""
training_sample_count_actual = 0
pred_count_actual = 0
with open('../data/train.pkl', 'rb') as f:
    train = pickle.load(f)
    for i in range(pred_count_desired):
        traj = train[i]['traj']

        if len(traj) > sample_count * 2:  # need both input and output to be the same size or else gpt response is unreliable
            traj_slim = numpy.delete(traj, [4,5], axis=1) # Get rid of unneeded rows
            input = traj_slim[0:sample_count]
            prompt += "\n".join(map(str, input.tolist())) + "\n\n###\n\n"

            output = traj_slim[sample_count:sample_count*2]

            # Predict
            pred = llm.generate(prompt, model)
            # Format prediction
            pred_array = parse_pred_string(pred)
            # Visualize prediction
            visualize.plot(input, output, pred_array, 'gpt-3.0-davinci-fine-tuned' + '/test_'+str(pred_count_actual+1))
            # Evaluate loss
            evaluate.eval(pred_array, output)

            # Reset values
            prompt = ""
            pred_count_actual += 1

            if pred_count_actual == pred_count_desired:
                evaluate.plot('gpt-3.0-davinci-fine-tuned')
                break

