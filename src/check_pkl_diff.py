import pickle

path = 'data_info_all_with_history_13.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

for datum in data:
    if 'actions_pred' in datum['annos']:
        print(datum['annos']['actions_pred'])