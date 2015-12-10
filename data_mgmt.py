import numpy as np
import scipy.io as sio
from scipy.sparse import lil_matrix, csr_matrix

def save_model_params(params_dict, target_file):
    sio.savemat(target_file, params_dict)

def load_model_params(load_file):
    params = sio.loadmat(load_file)
    return params

def vocab_from_mat(mat):
    vocab = {}
    words = mat.dtype.names
    idx = mat[0][0]
    for i in range(len(idx)):
        vocab[words[i]] = idx[i][0][0]
    return vocab

def load_test_corpus(corpus_file, test_vocab_file, train_vocab_file):
    test_vfile = open(test_vocab_file, 'r')
    test_vocab = eval(test_vfile.read())
    test_vfile.close()
    train_vfile = open(train_vocab_file, 'r')
    train_vocab = eval(train_vfile.read())
    train_vfile.close()

    test_corpus = load_corpus(corpus_file)
    docmats, idxs, freqs = test_corpus['docmats'], test_corpus['idxs'], test_corpus['freqs']
    fsplit = corpus_file.split('/')
    dr = fsplit[0]
    fname = fsplit[-1]
    if 'ap' in fname:
        train_corpus = sio.loadmat(dr + '/' + 'ap_train.mat')
    elif 'sn' in fname:
        train_corpus = sio.loadmat(dr + '/' + 'sn_train.mat')
    # get corresponding train corpus file
    train_index = {v:k for (k,v) in train_vocab.items()}
    num_words = len(train_vocab.keys())
    test_index = {v:k for (k,v) in test_vocab.items()}
    vocab_swap = np.zeros(len(test_vocab.keys()),dtype=int)
    for i in range(vocab_swap.shape[0]):
        if test_index[i] in train_vocab:
            vocab_swap[i] = train_vocab[test_index[i]]
        else:
            vocab_swap[i] = -1
    new_words = [k for k in test_vocab if k in train_vocab]
    new_docmats = []
    new_idxs = []
    new_freqs = []
    for i in range(len(idxs)):
        idx, freq = idxs[i], freqs[i]
        new_idx = vocab_swap[idx]
        remove = np.argwhere(new_idx == -1)
        new_idx = np.delete(new_idx, remove, 0)
        data = [1] * new_idx.shape[0]
        rows = [i for i in range(len(data))]
        cols = new_idx.tolist()
        new_docmat = csr_matrix((data, (rows, cols)), shape=(new_idx.shape[0], num_words))
        new_freq = csr_matrix(new_docmat.sum(axis=0))
        new_docmats.append(new_docmat)
        new_idxs.append(new_idx)
        new_freqs.append(new_freq)

    ld_corpus = {
        'docmats': new_docmats,
        'idxs': new_idxs,
        'freqs': new_freqs
    }
    return ld_corpus

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
