import numpy as np
import pickle as pkl
import os

from hydra.utils import get_original_cwd, to_absolute_path


class ActionTokenizer():
    def __init__(
            self, 
            model_type_name,
            discretize,
            num_bins,
            past_traj = False,
        ):

        num_actions = 2 # x and y
        self.num_wps = 6 # 6 waypoints
        self.action_bins = num_bins
        # ranges are taken from dataset
        if past_traj:
            self.y_range = [-40, 60] # in meters
        else:
            self.y_range = [-2.5, 60] # in meters
        self.x_range = [-15, 15] # in meters
        self.discretize = discretize #'linear' # 'linear', 'square', 'nonlinear'
        data_path = to_absolute_path(f"data/action_tokenizer/action_bin_token_mapping_{model_type_name}.pkl")
        self.action_bin_to_token_id = pkl.load(open(data_path, "rb"))

    def tokenize(self, ego_traj):

        ego_traj_bins = self._discretize_array(ego_traj[:, 0], ego_traj[:, 1], self.x_range, self.y_range, self.action_bins, self.discretize)
        # actions = self._bin_to_meters_array(ego_traj_bins[::2], ego_traj_bins[1::2], self.x_range, self.y_range, self.action_bins, self.discretize)
        # print('actions', actions)
        # bins to tokens
        ego_traj_tokens = []
        # add start token
        ego_traj_tokens.append(self.action_bin_to_token_id['action_bin_tokens']['<START_ACTION>'])
        for i in range(ego_traj_bins[0].shape[0]):
            ego_traj_tokens.append(self.action_bin_to_token_id['action_bin_tokens'][ego_traj_bins[0][i]])
            ego_traj_tokens.append(self.action_bin_to_token_id['action_bin_tokens'][ego_traj_bins[1][i]])
        # add end token
        ego_traj_tokens.append(self.action_bin_to_token_id['action_bin_tokens']['<END_ACTION>'])

        return ego_traj_tokens

    def tokenize_batch(self, ego_trajs):
        ego_traj_tokens = []
        for ego_traj in ego_trajs:
            ego_traj_bins = self._discretize_array(ego_traj[:, 0], ego_traj[:, 1], self.x_range, self.y_range, self.action_bins, self.discretize)
            # bins to tokens
            ego_traj_tokens_batch = []
            # add start token
            ego_traj_tokens_batch.append(self.action_bin_to_token_id['action_bin_tokens']['<START_ACTION>'])
            for i in range(ego_traj_bins[0].shape[0]):
                ego_traj_tokens_batch.append(self.action_bin_to_token_id['action_bin_tokens'][ego_traj_bins[0][i]])
                ego_traj_tokens_batch.append(self.action_bin_to_token_id['action_bin_tokens'][ego_traj_bins[1][i]])
            # add end token
            ego_traj_tokens_batch.append(self.action_bin_to_token_id['action_bin_tokens']['<END_ACTION>'])
            ego_traj_tokens.append(ego_traj_tokens_batch)

        return ego_traj_tokens


    def detokenize(self, ego_traj_tokens):
        # tokens to bins
        ego_traj_bins = []
        mask = []
        for token in ego_traj_tokens:
            if token == self.action_bin_to_token_id['action_bin_tokens']['<START_ACTION>']:
                continue
            if token == self.action_bin_to_token_id['action_bin_tokens']['<END_ACTION>']:
                break
            if token in self.action_bin_to_token_id['tokens_action_bin']:
                ego_traj_bins.append(self.action_bin_to_token_id['tokens_action_bin'][token])
                mask.append(1)
            else:
                ego_traj_bins.append(0)
                mask.append(0)
        # bins to meters
        mask = np.array(mask)
        # mask: [x1, y1, x2, y2, ...] -> [x1, x2, ...], [y1, y2, ...]
        mask = np.stack([mask[::2], mask[1::2]], axis=1)
        ego_traj_meters = self._bin_to_meters_array(ego_traj_bins[::2], ego_traj_bins[1::2], self.x_range, self.y_range, self.action_bins, self.discretize)
        ego_traj_meters = np.stack([ego_traj_meters[0], ego_traj_meters[1]], axis=1)
        ego_traj_meters = ego_traj_meters * mask

        return ego_traj_meters
    
    # def detokenize_batch(self, ego_traj_tokens_batch):
    #     ego_traj_meters_batch = []
    #     for ego_traj_tokens in ego_traj_tokens_batch:
    #         # tokens to bins
    #         ego_traj_bins = []
    #         for token in ego_traj_tokens:
    #             if token == self.action_bin_to_token_id['action_bin_tokens']['<START_ACTION>'] or token == self.action_bin_to_token_id['action_bin_tokens']['<END_ACTION>']:
    #                 continue
    #             if token in self.action_bin_to_token_id['tokens_action_bin']:
    #                 ego_traj_bins.append(self.action_bin_to_token_id['tokens_action_bin'][token.item()])
    #             else:
    #                 ego_traj_bins.append(0)
    #         # bins to meters
    #         ego_traj_meters = self._bin_to_meters_array(ego_traj_bins[::2], ego_traj_bins[1::2], self.x_range, self.y_range, self.action_bins, self.discretize)
    #         ego_traj_meters_batch.append(ego_traj_meters)

    #     return ego_traj_meters_batch

    def detokenize_batch(self, ego_traj_tokens_batch):
        ego_traj_meters_batch = []
        
        for ego_traj_tokens in ego_traj_tokens_batch:
            # tokens to bins
            ego_traj_bins = []
            mask = []
            for token in ego_traj_tokens:
                if token == self.action_bin_to_token_id['action_bin_tokens']['<START_ACTION>']:
                    continue
                if token == self.action_bin_to_token_id['action_bin_tokens']['<END_ACTION>']:
                    break
                if token.item() in self.action_bin_to_token_id['tokens_action_bin'].keys():
                    ego_traj_bins.append(self.action_bin_to_token_id['tokens_action_bin'][token.item()])
                    mask.append(1)
                else:
                    ego_traj_bins.append(0)
                    mask.append(0)
            # bins to meters
            # if len mask is uneven
            if len(mask) % 2 != 0:
                mask.append(0)
                ego_traj_bins.append(0)
            if len(mask) < 12:
                for i in range(12-len(mask)):
                    mask.append(0)
                    ego_traj_bins.append(0)
            mask = np.array(mask)
            # mask: [x1, y1, x2, y2, ...] -> [x1, x2, ...], [y1, y2, ...]
            mask = np.stack([mask[::2], mask[1::2]], axis=1)
            mask = mask[:self.num_wps]
            ego_traj_meters = self._bin_to_meters_array(ego_traj_bins[::2], ego_traj_bins[1::2], self.x_range, self.y_range, self.action_bins, self.discretize)
            ego_traj_meters = np.stack([ego_traj_meters[0], ego_traj_meters[1]], axis=1)
            ego_traj_meters = ego_traj_meters[:self.num_wps]
            ego_traj_meters = ego_traj_meters * mask

            ego_traj_meters_batch.append(ego_traj_meters)

        ego_traj_meters_batch = np.stack(ego_traj_meters_batch, axis=0)

        return ego_traj_meters_batch

    def _discretize_array(self, x, y, x_range, y_range, action_bins, discretize):
        """
        Discretizes the input arrays x and y based on the specified discretization method.

        Args:
            x (numpy.ndarray): The x-coordinates to be discretized.
            y (numpy.ndarray): The y-coordinates to be discretized.
            x_range (list): The range of x values to be discretized.
            y_range (list): The range of y values to be discretized.
            action_bins (int): The number of bins to discretize the values into.
            discretize (str): The discretization method to use. Must be one of 'linear', 'square', 'sqrt', or 'nonlinear'.

        Returns:
            Tuple of two numpy.ndarrays: The discretized x and y arrays.
        """
        if discretize == 'linear':
            x = ((x - x_range[0]) / (x_range[1] - x_range[0]) * action_bins).astype(int)
            y = ((y - y_range[0]) / (y_range[1] - y_range[0]) * action_bins).astype(int)
        elif discretize == 'square':
            x_range = [np.sign(x_range[0]) * np.sqrt(np.abs(x_range[0])), np.sign(x_range[1]) * np.sqrt(np.abs(x_range[1]))]
            y_range = [np.sign(y_range[0]) * np.sqrt(np.abs(y_range[0])), np.sign(y_range[1]) * np.sqrt(np.abs(y_range[1]))]
            x_bin_start_values = np.linspace(x_range[0], x_range[1], action_bins) ** 2 * np.sign(np.linspace(x_range[0], x_range[1], action_bins))
            y_bin_start_values = np.linspace(y_range[0], y_range[1], action_bins) ** 2 * np.sign(np.linspace(y_range[0], y_range[1], action_bins))
            # mapping x to the closest bin
            x = np.argmin(np.abs(x_bin_start_values[:, None] - x), axis=0)
            y = np.argmin(np.abs(y_bin_start_values[:, None] - y), axis=0)
        elif discretize == 'sqrt':
            x = np.sign(x) * np.sqrt(np.abs(x))
            y = np.sign(y) * np.sqrt(np.abs(y))
            # square ranges with keeping the sign
            x_range = [np.sign(x_range[0]) * np.sqrt(np.abs(x_range[0])), np.sign(x_range[1]) * np.sqrt(np.abs(x_range[1]))]
            y_range = [np.sign(y_range[0]) * np.sqrt(np.abs(y_range[0])), np.sign(y_range[1]) * np.sqrt(np.abs(y_range[1]))]

            x = ((x - x_range[0]) / (x_range[1] - x_range[0]) * action_bins).astype(int)
            y = ((y - y_range[0]) / (y_range[1] - y_range[0]) * action_bins).astype(int)
        elif discretize == 'nonlinear':
            x = ((np.arctan(x / 10) + np.pi / 2) / np.pi * action_bins).astype(int)
            y = ((np.arctan(y / 10) + np.pi / 2) / np.pi * action_bins).astype(int)
        else:
            raise NotImplementedError

        return x, y

    def _bin_to_meters_array(self, x_bin, y_bin, x_range, y_range, action_bins, discretize):
        """
        Converts bin values to meters array based on the given discretization method.

        Args:
            x_bin (list): List of bin values for x-axis.
            y_bin (list): List of bin values for y-axis.
            x_range (list): List of minimum and maximum values for x-axis.
            y_range (list): List of minimum and maximum values for y-axis.
            action_bins (int): Number of bins for discretization.
            discretize (str): Discretization method to be used. Can be one of 'linear', 'square', 'sqrt', or 'nonlinear'.

        Returns:
            x (np.array): Array of x-axis values in meters.
            y (np.array): Array of y-axis values in meters.
        """
        if discretize == 'linear':
            x = np.array(x_bin) / action_bins * (x_range[1] - x_range[0]) + x_range[0]
            y = np.array(y_bin) / action_bins * (y_range[1] - y_range[0]) + y_range[0]
        elif discretize == 'square':
            x_range = [np.sign(x_range[0]) * np.sqrt(np.abs(x_range[0])), np.sign(x_range[1]) * np.sqrt(np.abs(x_range[1]))]
            y_range = [np.sign(y_range[0]) * np.sqrt(np.abs(y_range[0])), np.sign(y_range[1]) * np.sqrt(np.abs(y_range[1]))]
            x_bin_start_values = np.linspace(x_range[0], x_range[1], action_bins) ** 2 * np.sign(np.linspace(x_range[0], x_range[1], action_bins))
            y_bin_start_values = np.linspace(y_range[0], y_range[1], action_bins) ** 2 * np.sign(np.linspace(y_range[0], y_range[1], action_bins))
            x = x_bin_start_values[x_bin]
            y = y_bin_start_values[y_bin]
        elif discretize == 'sqrt':
            x_range = [np.sign(x_range[0]) * np.sqrt(np.abs(x_range[0])), np.sign(x_range[1]) * np.sqrt(np.abs(x_range[1]))]
            y_range = [np.sign(y_range[0]) * np.sqrt(np.abs(y_range[0])), np.sign(y_range[1]) * np.sqrt(np.abs(y_range[1]))]
            x = np.array(x_bin) / action_bins * (x_range[1] - x_range[0]) + x_range[0]
            y = np.array(y_bin) / action_bins * (y_range[1] - y_range[0]) + y_range[0]
            x = np.sign(x) * (x ** 2)
            y = np.sign(y) * (y ** 2)

        elif discretize == 'nonlinear':
            x = np.tan(np.array(x_bin) / action_bins * np.pi - np.pi / 2) * 10
            y = np.tan(np.array(y_bin) / action_bins * np.pi - np.pi / 2) * 10
        else:
            raise NotImplementedError
        
        return x, y


if __name__ == "__main__":
    # test
    model_type_name = 'blip2-flan-t5-xl'
    discretize = 'square'
    action_tokenizer = ActionTokenizer(model_type_name, discretize)

    # test tokenize
    ego_traj = np.array([[0, 0], [1, 1], [2, 2]])
    ego_traj_tokens = action_tokenizer.tokenize(ego_traj)
    print(ego_traj_tokens)

    # test detokenize
    ego_traj_meters = action_tokenizer.detokenize(ego_traj_tokens)
    print(ego_traj_meters)