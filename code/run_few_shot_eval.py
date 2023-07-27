import pickle
import numpy
import json
import large_language_model as llm
import evaluate
import visualize

model = "curie"

pred_count_desired = 10 # Number of predictions you want to average for accuracy/loss metrics
pred_hours = 1 # This can be changed to get longer predictions
minutes_per_sample = 10 # This is not configurable, this is how the data was gathered
sample_count = int(60/minutes_per_sample * pred_hours)

training_sample_count_desired = 2 # More than 3 exceeds max prompt size

def parse_pred_string(pred):
    pred_array = []
    pred_lines = pred.splitlines()

    for line in pred_lines:
        if len(pred_array) == sample_count:
            break  # Cut off extra content
        elif line.endswith("]"):
            pred_array.append(json.loads(line))

    # Add extra rows if the LLM stopped early
    while len(pred_array) < sample_count:
        pred_array.append(pred_array[-1])

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

            output = traj_slim[sample_count:sample_count*2]
            if training_sample_count_actual != training_sample_count_desired:
                prompt += "OUTPUT:\n" + "\n".join(map(str, output.tolist())) + "\n"
            else:
                # This is the final sample, and the correct value of the prediction
                prompt += "OUTPUT:\n"

            training_sample_count_actual += 1
            if training_sample_count_actual == training_sample_count_desired + 1: # Need an input for the prediction
                # Predict
                pred = llm.generate(prompt, model)
                # Format prediction
                pred_array = parse_pred_string(pred)
                # Visualize prediction
                visualize.plot(input, output, pred_array, model + '/test_'+str(pred_count_actual+1))
                # Evaluate loss
                evaluate.eval(pred_array, output)

                # Reset values
                prompt = ""
                training_sample_count_actual = 0
                pred_count_actual += 1

                if pred_count_actual == pred_count_desired:
                    evaluate.plot(model)
                    break

