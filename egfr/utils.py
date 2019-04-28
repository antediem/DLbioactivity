
import os
import pickle
import torch


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, model_dir_path, hash_code):
    """
    :param model: training model
    :param model_dir_path: directory path
    :param hash_code: hashcode
    :param e: epoch
    """
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    torch.save(model.state_dict(), "{}/model_{}_{}".format(model_dir_path, hash_code, "BEST"))


def load_model():
    pass


