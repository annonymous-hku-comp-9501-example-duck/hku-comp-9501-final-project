import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

split = "train"

path_val = f"data/DriveLM_v2/nuscenes_infos_temporal_{split}_xjw_w_annos_wo_mm.pkl"

data_info_all = pkl.load(open(path_val, "rb"))
past_traj_all = pkl.load(open('data/DriveLM_v2/data_nuscene_train.pkl', "rb"))

valid_points = 0
invalid_points = 0

# Initialize min and max values for x and y directions
min_x = np.inf
max_x = -np.inf
min_y = np.inf
max_y = -np.inf

x_values = []
y_values = []

for past_traj_keys in past_traj_all:
    past_traj = past_traj_all[past_traj_keys]['x']
    past_traj = np.array(past_traj)
    # Update min and max values for x and y directions
    min_x = min(min_x, np.min(past_traj[:, 0]))
    max_x = max(max_x, np.max(past_traj[:, 0]))
    min_y = min(min_y, np.min(past_traj[:, 1]))
    max_y = max(max_y, np.max(past_traj[:, 1]))

print("min_x: ", min_x)
print("max_x: ", max_x)
print("min_y: ", min_y)
print("max_y: ", max_y)



# Initialize min and max values for x and y directions
min_x = np.inf
max_x = -np.inf
min_y = np.inf
max_y = -np.inf

x_values = []
y_values = []
for data_idx in range(len(data_info_all)):
    data_info = data_info_all[data_idx]

    ego_traj = data_info['annos']['gt_sdc_fut_traj'].squeeze()                # this is the ego trajectory of current frame
    ego_traj_mask = data_info['annos']['gt_sdc_fut_traj_mask'].squeeze()

    # count number of valid points in ego_traj_mask
    num_valid_points = ego_traj_mask.sum()
    num_invalid_points = ego_traj_mask.shape[0] - num_valid_points

    valid_points += num_valid_points
    invalid_points += num_invalid_points

    # Update min and max values for x and y directions
    min_x = min(min_x, np.min(ego_traj[:, 0]))
    max_x = max(max_x, np.max(ego_traj[:, 0]))
    min_y = min(min_y, np.min(ego_traj[:, 1]))
    max_y = max(max_y, np.max(ego_traj[:, 1]))

    x_values.extend(ego_traj[:, 0])
    y_values.extend(ego_traj[:, 1])

print("Number of valid points: ", valid_points)
print("Number of invalid points: ", invalid_points)
print("min_x: ", min_x)
print("max_x: ", max_x)
print("min_y: ", min_y)
print("max_y: ", max_y)

# Save min and max values for x and y directions to a txt file
with open(f"data/DriveLM_v2/stats/min_max_values_ego_traj_{split}.txt", "w") as f:
    f.write("min_x: " + str(min_x) + "\n")
    f.write("max_x: " + str(max_x) + "\n")
    f.write("min_y: " + str(min_y) + "\n")
    f.write("max_y: " + str(max_y) + "\n")

# Plot histograms of x and y values of ego_traj
plt.hist(x_values, bins=50)
plt.title("Histogram of x values of ego_traj")
plt.xlabel("x values")
plt.ylabel("Frequency")
plt.savefig(f"data/DriveLM_v2/stats/histogram_x_values_ego_traj_{split}.png")
plt.clf()

plt.hist(y_values, bins=50)
plt.title("Histogram of y values of ego_traj")
plt.xlabel("y values")
plt.ylabel("Frequency")
plt.savefig(f"data/DriveLM_v2/stats/histogram_y_values_ego_traj_{split}.png")
plt.clf()
# Plot histograms of x and y values of ego_traj with logarithmic view
plt.hist(x_values, bins=50, log=True)
plt.title("Histogram of x values of ego_traj (logarithmic view)")
plt.xlabel("x values")
plt.ylabel("Frequency")
plt.savefig(f"data/DriveLM_v2/stats/histogram_x_values_ego_traj_{split}_log.png")
plt.clf()

plt.hist(y_values, bins=50, log=True)
plt.title("Histogram of y values of ego_traj (logarithmic view)")
plt.xlabel("y values")
plt.ylabel("Frequency")
plt.savefig(f"data/DriveLM_v2/stats/histogram_y_values_ego_traj_{split}_log.png")
plt.clf()
