import json
import evaluate
import visualize

model = "gpt-4.0-few-shot"

pred_count_desired = 10 # Number of predictions you want to average for accuracy/loss metrics
pred_hours = 3 # This can be changed to get longer predictions
minutes_per_sample = 10 # This is not configurable, this is how the data was gathered
sample_count = int(60/minutes_per_sample * pred_hours)

training_sample_count_desired = 5 # More than 3 exceeds max prompt size

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


for i in range(3):
    # Get file contents
    input = json.loads(open("gpt-4.0-few-shot/input_"+str(i+1)+".txt").read())
    pred = parse_pred_string(open("gpt-4.0-few-shot/pred_"+str(i+1)+".txt").read())
    actual = json.loads(open("gpt-4.0-few-shot/actual_"+str(i+1)+".txt").read())
    # Visualize prediction
    visualize.plot(input, actual, pred, model + '/test_'+str(i+1))
    # Evaluate loss
    evaluate.eval(pred, actual)
evaluate.plot(model)