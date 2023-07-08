import pickle

import matplotlib.pyplot as plt
import numpy as np

import few_shot_vals

LAT_MIN = 55.5
LAT_MAX = 58.0
LON_MIN = 10.3
LON_MAX = 13

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN



def scale_array(values, min, max):
    updated_list = []
    for value in values:
        updated_list.append(scale_value(value, min, max))
    return np.array(updated_list)

# 0 - 1 range up to orig number
def scale_value(value, min, max):
    return (((value - 0) * (max - min)) / (1 - 0)) + min


# coastline_filename = "../data/dma_coastline_polygons.pkl"
# with open(coastline_filename, 'rb') as f:
#     l_coastline_poly = pickle.load(f)
#     for point in l_coastline_poly:
#         poly = np.array(point)
#         plt.plot(poly[:, 1], poly[:, 0], color="k", linewidth=0.8)

# training_filename = "../data/train.pkl"
# with open(training_filename, 'rb') as f:
#     Data = pickle.load(f)
#     Vs = Data
#     FIG_DPI = 150
#     cmap = plt.cm.get_cmap('Blues')
#     N = len(Vs)
#     for d_i in range(N):
#         c = cmap(float(d_i) / (N - 1))
#         tmp = Vs[d_i]
#         v_lat = tmp['traj'][:, 0] * LAT_RANGE + LAT_MIN
#         v_lon = tmp['traj'][:, 1] * LON_RANGE + LON_MIN
#         plt.plot(v_lon, v_lat, color=c, linewidth=0.8)

seqlen = 18

plt.plot(scale_array([item[0] for item in few_shot_vals.input], LON_MIN, LON_MAX),
         scale_array([item[1] for item in few_shot_vals.input], LAT_MIN, LAT_MAX), linestyle="solid", color="r")

plt.plot(scale_array([item[0] for item in few_shot_vals.actuals], LON_MIN, LON_MAX),
         scale_array([item[1] for item in few_shot_vals.actuals], LAT_MIN, LAT_MAX), linestyle="dashed", markersize=4, color="r")

plt.plot(scale_array([item[0] for item in few_shot_vals.pred], LON_MIN, LON_MAX),
         scale_array([item[1] for item in few_shot_vals.pred], LAT_MIN, LAT_MAX), linestyle="dashed", color="g")

plt.xlim([LON_MIN, LON_MAX])
plt.ylim([LAT_MIN, LAT_MAX])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig('fewShot.jpg', dpi=150)
plt.show()
