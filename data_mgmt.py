import numpy as np
import scipy.io as sio

def save_model_params(params_dict, target_file):
    sio.savemat(target_file, params_dict)

def load_model_params(load_file):
    params = sio.loadmat(load_file)
    return params

def load_corpus(corpus_file):
    corpus = sio.loadmat(load_file)
    docmats, idxs, freqs = []
    cont = True
    count = 0
    while ((str(count) + "_" + "docmat") in corpus):
        docmats.append(corpus[str(count) + "_" + "docmat"])
        idxs.append(corpus[str(count) + "_" + "index"])
        freqs.append(corpus[str(count) + "_" + "freq"])
        count += 1
    ld_corpus = {
        "docmats": docmats,
        "idxs": idxs,
        "freqs": freqs
    }
    return ld_corpus
