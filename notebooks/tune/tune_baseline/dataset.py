
import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import os, glob
import json


def read_data(data_path):
    data = None
    if data_path.endswith('.csv'):
        try:
            #data = pd.read_json(data_path, lines=True, nrows=100, chunksize=1000)
            data = pd.read_csv(data_path)
        except ValueError:
            print('ValueError')
            #data = pd.read_json(data_path)
    #if data_path.endswith('.zip'):
        #try:
            #data = pd.read_json(data_path, compression='zip', lines=True)
        #except ValueError:
            #data = pd.read_json(data_path, compression='zip')
    return data


def train_validation_split(data_path):
    if os.path.isdir(data_path):
        train_path = os.path.join(data_path, 'train.csv')
        val_path = os.path.join(data_path, 'val.csv')
    else:
        train_path = data_path.split('.')[0] + '_' + 'train.csv'
        val_path = data_path.split('.')[0] + '_' + 'val.csv'
    if os.path.exists(train_path) and os.path.exists(val_path):
        # return read_data(train_path), read_data(val_path)
        return pd.read_csv(train_path), pd.read_csv(val_path)
    data = read_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print('Train')
    print(train_data.shape)
    train_data.to_csv(train_path, index=False)
    print('Val')
    print(val_data.shape)
    val_data.to_csv(val_path, index=False)
    return train_data, val_data


def train_cross_validation_split(data_path):
    dir_path = os.path.dirname(os.path.abspath(data_path))
    fold_dirs = glob.glob(os.path.join(dir_path, 'folds_*'))
    if len(fold_dirs) == 5:
        for fold_dir in fold_dirs:
            train_path = os.path.join(fold_dir, 'train.csv')
            val_path = os.path.join(fold_dir, 'val.csv')
            yield pd.read_csv(train_path), pd.read_csv(val_path)
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        data = read_data(data_path)
        for i, (train_ids, val_ids) in enumerate(kfold.split(X=data.drop('active', axis=1).values,
                                                             y=data['active'].values)):
            train_data = data.iloc[train_ids, :]
            val_data = data.iloc[val_ids, :]
            # os.makedirs(os.path.join(dir_path, 'folds_{}'.format(i)), exist_ok=True)
            # train_data.to_json(os.path.join(os.path.join(dir_path, 'folds_{}'.format(i)), 'train.json'))
            # val_data.to_json(os.path.join(os.path.join(dir_path, 'folds_{}'.format(i)), 'val.json'))

            yield train_data, val_data


class ANYDataset(data.Dataset):
    def __init__(self, data, infer=False):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = read_data(data)
        #self.NON_MORD_NAMES = ['smile_ft', 'id', 'subset', 'quinazoline', 'pyrimidine', 'smiles', 'active']
        self.NON_MORD_NAMES = ['smile_ft', 'smiles', 'active']
        self.infer = infer

        # Standardize mord features
        scl = StandardScaler()
        self.mord_ft = scl.fit_transform(self.data.drop(columns=self.NON_MORD_NAMES).astype(np.float64)).tolist()
        self.non_mord_ft_temp = self.data['smile_ft'].values.tolist()
        self.non_mord_ft = []
        for i in range(len(self.non_mord_ft_temp)):
          self.non_mord_ft.append(json.loads(self.non_mord_ft_temp[i]))
        self.smiles = self.data['smiles'].values.tolist()
        self.label = self.data['active'].values.tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.infer:
            return self.smiles[idx], self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]
        else:
            return self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]

    def get_dim(self, ft):
        if ft == 'non_mord':
            return len(self.non_mord_ft[0])
        if ft == 'mord':
            return len(self.mord_ft[0])

    def get_smile_ft(self):
        return self.non_mord_ft