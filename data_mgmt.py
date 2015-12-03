import numpy as np
import scipy.io as sio

def save_model_params(params_dict, target_file):
    sio.savemat(target_file, params_dict)

def load_model_params(load_file):
    params = sio.loadmat(load_file)
    return params

def load_corpus(corpus_file):
    corpus = sio.loadmat(load_file)
    return corpus
