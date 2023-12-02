import pickle

data = pickle.load(open('/mnt/disk01/nuscenes/openlane_v2_nus/data_dict_subset_B_val.pkl', 'rb'))
data_new = {}
for key in data:
    timestamp = key[-1]
    if timestamp not in data_new:
        data_new[timestamp] = data[key]
    else:
        print('Repeat !')
        break

pickle.dump(data_new, open('/mnt/disk01/nuscenes/openlane_v2_nus/data_dict_subset_B_val_new.pkl', 'wb'))