import torch as torch
import matplotlib.pyplot as plt
import numpy as np
import few_shot_vals

device = torch.device("cpu")

init_seqlen = 18

lat_min = 55.5
lat_max = 58.0
lon_min = 10.3
lon_max = 13

v_ranges = torch.tensor([2, 3, 0, 0]).to(device)
v_roi_min = torch.tensor([lat_min, -7, 0, 0]).to(device)
max_seqlen = init_seqlen
masks = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1., 1., 1.]])
masks = masks[:, :max_seqlen].to(device)

preds = torch.tensor([few_shot_vals.pred])
inputs = torch.tensor([few_shot_vals.actuals])

input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180


def haversine(input_coords,
              pred_coords):
    """ Calculate the haversine distances between input_coords and pred_coords.

    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.

    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[..., 0] - input_coords[..., 0]
    lon_errors = pred_coords[..., 1] - input_coords[..., 1]
    a = torch.sin(lat_errors / 2) ** 2 \
        + torch.cos(input_coords[:, :, 0]) * torch.cos(pred_coords[:, :, 0]) * torch.sin(lon_errors / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    d = R * c
    return d

d = haversine(input_coords, pred_coords) * masks
pred_errors = d.detach().cpu().numpy()[0]

## Convert km to nautical miles
conversion_factor = 0.539957
pred_errors = [i * conversion_factor for i in pred_errors]

## Plot
# ===============================
plt.figure(figsize=(9, 6), dpi=150)
v_times = np.arange(len(pred_errors)) / 6
plt.plot(v_times, pred_errors)

timestep = 6
plt.plot(1, pred_errors[timestep], "o")
plt.plot([1, 1], [0, pred_errors[timestep]], "r")
plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

timestep = 12
plt.plot(2, pred_errors[timestep], "o")
plt.plot([2, 2], [0, pred_errors[timestep]], "r")
plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

timestep = 17
plt.plot(3, pred_errors[timestep], "o")
plt.plot([3, 3], [0, pred_errors[timestep]], "r")
plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
plt.xlabel("Time (hours)")
plt.ylabel("Prediction errors (miles)")
plt.xlim([0, 12])
plt.ylim([0, 20])
plt.savefig("prediction_error.png")

plt.show()