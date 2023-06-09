import pickle
import numpy
import json
import large_language_model as llm
import evaluate

pred_count_desired = 1 # Number of predictions you want to average for accuracy/loss metrics
pred_hours = 3 # This can be changed to get longer predictions
minutes_per_sample = 10 # This is not configurable, this is how the data was gathered
sample_count = int(60/minutes_per_sample * pred_hours)

training_sample_count_desired = 2 # More than 3 exceeds max prompt size

def parse_pred_string(pred):
    pred_array = []
    pred_lines = pred.splitlines()

    for line in pred_lines:
        if line.endswith("]"):
            pred_array.append(json.loads(line))

    return pred_array

prompt = ""
training_sample_count_actual = 0
pred_count_actual = 0
with open('../data/train.pkl', 'rb') as f:
    train = pickle.load(f)
    for i in range(len(train)):
        traj = train[i]['traj']

        if len(traj) > sample_count * 2:  # need both input and output to be the same size or else gpt response is unreliable
            traj_slim = numpy.delete(traj, [4,5], axis=1) # Get rid of unneeded rows
            input = traj_slim[0:sample_count]
            prompt += "INPUT:\n" + "\n".join(map(str, input.tolist())) + "\n"

            output = traj_slim[sample_count+1:(sample_count+1)*2]
            if training_sample_count_actual != training_sample_count_desired:
                prompt += "OUTPUT:\n" + "\n".join(map(str, output.tolist())) + "\n"
            else:
                # This is the final sample, and the correct value of the prediction
                prompt += "OUTPUT:\n"

            training_sample_count_actual += 1
            if training_sample_count_actual == training_sample_count_desired + 1: # Need an input for the prediction
                pred = llm.generate(prompt)
                print(pred)

                pred_array = parse_pred_string(pred)
                evaluate.eval(pred_array, output)

                training_sample_count_actual = 0
                pred_count_actual += 1

                if pred_count_actual == pred_count_desired:
                    evaluate.plot()
                    break

