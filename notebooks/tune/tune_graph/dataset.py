#dataset.py

import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.utils.convert import from_networkx
import numpy as np
import os, glob

import networkx as nx
from pysmiles import read_smiles
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import dataloader


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        features = [atom.GetAtomicNum(),
                   atom.GetMass(),
                   atom.GetFormalCharge(),
                   hybridization_encoding(atom.GetHybridization()),
                   atom.GetNumExplicitHs(),
                   atom.GetExplicitValence(), #explicit valence (including Hs)
                   atom.GetNumRadicalElectrons(),
                   (1 if atom.GetIsAromatic() else 0)]
        G.add_node(atom.GetIdx(),
                    x=features)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def hybridization_encoding(hybridization):
    if hybridization == Chem.HybridizationType.S:
        return 1
    if hybridization == Chem.HybridizationType.SP:
        return 2
    if hybridization == Chem.HybridizationType.SP2:
        return 3
    if hybridization == Chem.HybridizationType.SP3:
        return 4
    if hybridization == Chem.HybridizationType.SP3D:
        return 5
    if hybridization == Chem.HybridizationType.SP3D2:
        return 6
    if hybridization == Chem.HybridizationType.OTHER:
        return 7




def read_data(data_path):
    data = None
    if data_path.endswith('.csv'):
        try:
            data = pd.read_csv(data_path)
        except ValueError:
            print('ValueError')

    return data


def train_validation_split(data_path):
    if os.path.isdir(data_path):
        train_path = os.path.join(data_path, 'train.csv')
        val_path = os.path.join(data_path, 'val.csv')
    else:
        train_path = data_path.split('.')[0] + '_' + 'train.csv'
        val_path = data_path.split('.')[0] + '_' + 'val.csv'
    if os.path.exists(train_path) and os.path.exists(val_path):

        return pd.read_csv(train_path), pd.read_csv(val_path)

    data = read_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)

    return train_data, val_data

class ANYDataset(data.Dataset):

    def __init__(self, data, infer=False):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = read_data(data)

        self.NON_MORD_NAMES = ['smiles', 'active']
        
        scl = StandardScaler()
        self.mord_ft = scl.fit_transform(
            self.data.drop(columns=self.NON_MORD_NAMES).astype(np.float64)).tolist()

        self.graphs = [Chem.MolFromSmiles(s) for s in self.data['smiles'].values.tolist()]
        self.graphs = [from_networkx(mol_to_nx(g)) for g in self.graphs]
        self.label = self.data['active'].values.tolist()
        

    def __len__(self):

        return len(self.graphs)

    def __getitem__(self, idx):

        return self.graphs[idx], self.mord_ft[idx], self.label[idx]