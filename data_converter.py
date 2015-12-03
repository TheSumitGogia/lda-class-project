import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np
from stop_words import get_stop_words
from random import shuffle
import os
import string

punc_rm = {ord(c): None for c in string.punctuation}
def strip_punc(text, tp='s'):
    text = text.strip()
    result = None
    if tp == 's':
        result = text.translate(string.maketrans("",""), string.punctuation)
    elif tp == 'u':
        result = text.translate(punc_rm)
    return result

stwords = get_stop_words('en')
stwords = [strip_punc(word, 'u') for word in stwords]
stwords = set(stwords)

def tokenize(lines):
    lines = [strip_punc(line, 's') for line in lines]
    lines = [line.split() for line in lines]
    tokens = [token.lower() for lsplit in lines for token in lsplit]
    tokens = [token for token in tokens if token not in stwords]
    return tokens

def get_sn_vocab(docs):
    print "Determining vocabulary..."
    words, word_ct = {}, 0
    for doc in docs:
        fulldoc = doc
        docfile = open(fulldoc, 'r')
        doclines = docfile.readlines()
        doctokens = tokenize(doclines)
        for token in doctokens:
            if token not in words:
                words[token] = word_ct
                word_ct += 1
        docfile.close()
    print "Determined vocabulary of size {0}".format(word_ct)
    return words, word_ct

def get_sn_mats(docs, words, word_ct, corpus_dict):
    print "Computing data matrices for documents"
    for doc_idx in range(len(docs)):
        fulldoc = docs[doc_idx]
        docfile = open(fulldoc, 'r')
        doclines = docfile.readlines()
        doctokens = tokenize(doclines)

        data = [1] * len(doctokens)
        rows = [i for i in range(len(doctokens))]
        cols = [words[token] for token in doctokens]
        docmat = csr_matrix((data, (rows, cols)), shape=(len(doctokens), word_ct))
        index = np.array(cols)
        freq = docmat.sum(axis=0)

        corpus_dict[str(doc_idx) + "_docmat"] = docmat
        corpus_dict[str(doc_idx) + "_index"] = index
        corpus_dict[str(doc_idx) + "_freq"] = freq

def sn_convert():

    print "Starting conversion of Sparknotes dataset..."
    bookdir = "sn/books"
    bookdirs = os.listdir(bookdir)
    all_docs = []
    for book in bookdirs:
        bookfiles = os.listdir(bookdir + "/" + book)
        bookfiles = [f for f in bookfiles if f.startswith("section")]
        bookfiles = [bookdir + "/" + book + "/" + doc for doc in bookfiles]
        all_docs.extend(bookfiles)
    print "Found all documents, number: {0}".format(len(all_docs))

    # test/train data
    shuffle(all_docs)
    split = int(len(all_docs) * 0.9)
    train, test = all_docs[:split], all_docs[split:]

    tr_words, tr_wc = get_sn_vocab(train)
    tt_words, tt_wc = get_sn_vocab(test)

    tr_corpus_dict = {"V": tr_wc, "M": len(train)}
    tt_corpus_dict = {"V": tt_wc, "M": len(test)}

    get_sn_mats(train, tr_words, tr_wc, tr_corpus_dict)
    get_sn_mats(test, tt_words, tt_wc, tt_corpus_dict)

    print "Writing document matrices to file..."
    sio.savemat("data/sn_train", tr_corpus_dict)
    sio.savemat("data/sn_test", tt_corpus_dict)
    print "Completed data write"

def get_ap_vocab(docs):
    # NOTE: side effect, tokenizes docs
    print "Determining vocabulary..."
    words, word_ct = {}, 0
    for i in range(len(docs)):
        tokens = tokenize(docs[i])
        for token in tokens:
            if token not in words:
                words[token] = word_ct
                word_ct += 1
        docs[i] = tokens
    print "Determined vocabulary of size: {0}".format(word_ct)
    return words, word_ct

def get_ap_mats(doc_texts, words, word_ct, corpus_dict):
    print "Computing data matrices for documents"
    for doc_idx in range(len(doc_texts)):
        doctokens = doc_texts[doc_idx]
        data = [1] * len(doctokens)
        rows = [i for i in range(len(doctokens))]
        cols = [words[token] for token in doctokens]
        docmat = csr_matrix((data, (rows, cols)), shape=(len(doctokens), word_ct))
        index = np.array(cols)
        freq = docmat.sum(axis=0)

        corpus_dict[str(doc_idx) + "_docmat"] = docmat
        corpus_dict[str(doc_idx) + "_index"] = index
        corpus_dict[str(doc_idx) + "_freq"] = freq

def ap_convert():
    print "Starting conversion of AP dataset..."
    docfile = open('ap/ap.txt', 'r')
    doclines = docfile.readlines()
    doclines = [line.strip() for line in doclines]
    doc_texts = []
    curr_lines, current = [], False
    words, word_ct = {}, 0

    print "Searching for documents..."
    for line in doclines:
        if line == "<TEXT>":
            current = True
        elif line == "</TEXT>" and current:
            current = False
            doc_texts.append(curr_lines)
            curr_lines = []
        elif current:
            curr_lines.append(line)
    docfile.close()
    print "Found all documents, number: {0}".format(len(doc_texts))

    shuffle(doc_texts)
    split = int(len(doc_texts) * 0.9)
    train, test = doc_texts[:split], doc_texts[split:]

    tr_words, tr_wc = get_ap_vocab(train)
    tt_words, tt_wc = get_ap_vocab(test)

    tr_corpus_dict = {"V": tr_wc, "M": len(train)}
    tt_corpus_dict = {"V": tt_wc, "M": len(test)}

    get_ap_mats(train, tr_words, tr_wc, tr_corpus_dict)
    get_ap_mats(test, tt_words, tt_wc, tt_corpus_dict)

    print "Writing document matrices to file..."
    sio.savemat("data/ap_train", tr_corpus_dict)
    sio.savemat("data/ap_test", tt_corpus_dict)
    print "Completed data write"

if __name__ == '__main__':
    sn_convert()
    ap_convert()
