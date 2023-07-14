import pickle
import numpy

with open('../data/train.pkl', 'rb') as f:
    train = pickle.load(f)
    for row in train:
        traj = row['traj']
    # traj1 = train[0]['traj']
        i = numpy.delete(traj, [4,5], axis=1)
        input = i[0:18]
        print("INPUT:\n[" + ",\n".join(map(str, input.tolist())) + "]")

        o = numpy.delete(traj, [2, 3, 4, 5], axis=1)
        output = o[19:37]
        print("OUTPUT:\n[" + ",\n".join(map(str, output.tolist())) + "]")
