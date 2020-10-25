__author__ = "Ldaze"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class DataLoader(object):
    """A class for loading and transforming data for the long short-term memory model"""

    def __init__(self, filename, encoding, split, x_cols, y_cols, is_same, feature_range, normalise, ascending):
        data_frame = pd.read_csv(filename, encoding=encoding)
        i_split = int(len(data_frame) * split)
        data_frame = data_frame.sort_index(ascending=ascending)
        self.x_train = data_frame.get(x_cols).values[: i_split]
        self.x_test = data_frame.get(x_cols).values[i_split:]
        self.y_train = data_frame.get(y_cols).values[: i_split]
        self.y_test = data_frame.get(y_cols).values[i_split:]
        self.is_same = is_same
        self.normalise = normalise
        if self.normalise:
            self.scale_er = MinMaxScaler(feature_range=eval(feature_range))

    # def standard(self, data):
    #     scaler = StandardScaler()
    #     scaler.fit_transform(data)

    def normalizer(self, data, labels=None):
        """Merge data and labels on axis=1 and it was used for normalization in global space.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Amount to X.

        labels : array-like of shape (n_samples, n_features)
            If the features of the labels are included in X, it should be None.

        Returns
        -------
        temp : array-like of shape (n_samples, n_features)
            Ignored.
        """

        if labels is None:
            return self.scale_er.fit_transform(data)
        else:
            temp = np.concatenate([data, labels], axis=1)
            temp = self.scale_er.fit_transform(temp)
            return np.split(temp, [data.shape[1]], axis=1)

    def converter(self, data, aux_data=None):
        """Convert the normalized data back.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The labels requiring inverse normalization.

        aux_data : array-like of shape (n_samples, n_features)
            If the features of the labels are equal to X, it should be None.

        Returns
        -------
        temp : array-like of shape (n_samples, n_features)
            Labels after inverse normalization.
        """

        if aux_data is None:
            return self.scale_er.inverse_transform(data)

        else:
            temp = np.concatenate([aux_data, data], axis=1)
            temp = self.scale_er.inverse_transform(temp)
            return np.split(temp, [aux_data.shape[1]], axis=1)[1]

    @staticmethod
    def create_dataset(data, labels, time_steps, pre_steps, interval=False):
        """Create serial dataset for training.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Amount to X.

        labels : array-like of shape (n_samples, n_features)
            Amount to y.

        time_steps:
            Ignore.

        pre_steps:
            If pre_steps is equal to 1, labels should be converted to 2D after getting the return values.

        interval:
            A Boolean value is used to determine whether there is an interval between each step when the window moves.

        Returns
        -------
        X,y : array-like of shape (n_samples, n_steps, n_features)
            The window moves step by step per iteration for training when is_interval is equal to false.
            The window moves multiple steps per iteration for test and plotting when is_interval is equal to true.
        """

        x, y = [], []
        remainder = (len(data) - time_steps) % pre_steps if interval else 0
        step = pre_steps if interval else 1

        for i in range(0, len(data) - time_steps - pre_steps - remainder + 1, step):
            x.append(data[i: i + time_steps])
            y.append(labels[i + time_steps: i + time_steps + pre_steps])
            # y.append(np.split(labels, [i + time_steps, i + time_steps + pre_steps], axis=0)[1])

        return np.array(x), np.array(y)

    @staticmethod
    def tile(data):
        """Flatten dataset with intervals.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_steps, n_features)
            Ignore.

        Returns
        -------
        res : array-like of shape (n_samples, n_features)
            Flatten 3D data into 2D.
        """

        res = []
        for rows in data:
            for row in rows:
                res.append(row)
        return np.array(res)


class DatasetMaker(DataLoader):

    def __init__(self, config):
        self.configs = config
        super(DatasetMaker, self).__init__(
            self.configs['data']['filename'],
            self.configs['data']['encoding'],
            self.configs['data']['split'],
            self.configs['data']['X_columns'],
            self.configs['data']['y_columns'],
            self.configs['data']['is_same'],
            self.configs['data']['feature_range'],
            self.configs['data']['normalise'],
            self.configs['data']['ascending']
        )

    def get_dataset(self, data, labels, is_interval=False):
        if self.is_same:
            data = self.normalizer(data) if self.normalise else data
            labels = data
        else:
            data, labels = self.normalizer(data, labels) if self.normalise else [data, labels]

        res = self.create_dataset(
            data,
            labels,
            self.configs['data']['time_steps'],
            self.configs['data']['pre_steps'],
            is_interval
        )

        return res
