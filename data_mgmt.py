import numpy as np
import scipy.io as sio

def save_model_params(params_dict, target_file):
    sio.savemat(target_file, params_dict)

def load_model_params(load_file):
    params = sio.loadmat(load_file)
    return params

def load_corpus(corpus_file):
    corpus = sio.loadmat(corpus_file)
    docmats, idxs, freqs = [], [], []
    cont = True
    count = 0
    while ((str(count) + "_" + "docmat") in corpus):
        docmat = corpus[str(count) + "_" + "docmat"]
        idx = corpus[str(count) + "_" + "index"]
        idx = idx[0]
        freq = corpus[str(count) + "_" + "freq"]
        docmats.append(docmat)
        idxs.append(idx)
        freqs.append(freq)
        count += 1
    ld_corpus = {
        "docmats": docmats,
        "idxs": idxs,
        "freqs": freqs
    }
    return ld_corpus
