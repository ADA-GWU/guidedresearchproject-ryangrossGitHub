import pickle
import numpy
import json

model = "davinci"

pred_count_desired = 10 # Number of predictions you want to average for accuracy/loss metrics
pred_hours = 3 # This can be changed to get longer predictions
minutes_per_sample = 10 # This is not configurable, this is how the data was gathered
sample_count = int(60/minutes_per_sample * pred_hours)

training_sample_count_desired = 2 # More than 3 exceeds max prompt size

fine_tune_data = []

prompt = ""
training_sample_count_actual = 0
pred_count_actual = 0
with open('../data/train.pkl', 'rb') as f:
    train = pickle.load(f)
    for i in range(int(len(train)/128)):
        traj = train[i]['traj']

        if len(traj) > sample_count * 2:  # need both input and output to be the same size or else gpt response is unreliable
            traj_slim = numpy.delete(traj, [4,5], axis=1) # Get rid of unneeded rows
            input = traj_slim[0:sample_count]
            output = traj_slim[sample_count:sample_count * 2]

            fine_tune_data.append({"prompt": str(input.tolist()) + "\n\n###\n\n", "completion": " " + str(input.tolist()) + " ###"})

open('fine_tuning.jsonl', 'w').write(json.dumps(fine_tune_data))