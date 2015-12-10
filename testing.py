import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from modeling import *
import matplotlib.pyplot as plt
import data_mgmt as dmg
import os
from django.utils.encoding import smart_str

def train_test_models(data_dir, target_dir, ntopics=[5, 10, 20, 50, 100]):
    corpora = os.listdir(data_dir)
    for corpus_file in corpora:
        '''
        if 'sn' not in corpus_file:
            continue
        '''
        # avoid training models on test corpora
        spl = corpus_file.split('_')
        spl2 = spl[-1].split('.')
        if spl2[0] == 'test': continue

        # load corpus and train models for all topic counts
        corpus = dmg.load_corpus(data_dir + "/" + corpus_file)
        '''
        for num_topics in ntopics:
            print 'Training PLSI model for {0} topics on {1} dataset'.format(num_topics, spl)
            plsi_model = PLSIModel(num_topics)
            plsi_model.train(corpus, numiter=10*num_topics)
            dmg.save_model_params(plsi_model.get_params(), target_dir + "/plsi_" + str(num_topics) + "_" + corpus_file[:-4] + ".mat")
        '''
        for num_topics in ntopics:
            print 'Training LDA model for {0} topics on {1} dataset'.format(num_topics, spl)
            lda_model = LDAModel(num_topics)
            try:
                lda_model.train(corpus)
                dmg.save_model_params(lda_model.get_params(), target_dir + "/lda_" + str(num_topics) + "_" + corpus_file[:-4] + ".mat")
            except KeyboardInterrupt:
                print 'caught2'
                dmg.save_model_params(lda_model.get_params(), target_dir + "/lda_" + str(num_topics) + "_" + corpus_file[:-4] + ".mat")

def cperplexity(model, corpus, mtype):
    if (mtype == 'plsi'):
        fin_ll = model.foldin(corpus)
        mg_ll = model.mg_test_ll(corpus)
        return fin_ll, mg_ll
    elif (mtype == 'lda'):
        var_ll = model.test(corpus)
        return var_ll

def test_perplexity_vs_ntopics(corpus, model_dir, tp):
    plsi_perplexities = []
    plsi_mg_perplexities = []
    lda_perplexities = []

    model_files = os.listdir(model_dir)
    model_files = [model_file for model_file in model_files if tp in model_file]
    model_full_files = [model_dir + "/" + model_file for model_file in model_files]
    idxs = corpus["idxs"]
    for i in range(len(model_files)):
        model_fname = model_files[i]
        model_fullname = model_full_files[i]
        model_split = model_fname.split('_')
        mtype, mk, mcorp = model_split[0], model_split[1], model_split[2]
        params = dmg.load_model_params(model_fullname)
        total_words = 0
        for j in range(len(idxs)):
            total_words += idxs[j].shape[0]
        print 'total words', total_words
        # TODO: perhaps clean by allowing model to be instantiated from params

        if mtype == 'plsi':
            beta, pi = params["beta"], params["pi"]
            model = PLSIModel(int(mk), beta=beta, pi=pi)
            fin_ll, mg_ll = cperplexity(model, corpus, 'plsi')
            perplexity = -1 * fin_ll / total_words
            perplexity_mg = -1 * mg_ll / total_words
            perplexity = np.exp(perplexity)
            perplexity_mg = np.exp(perplexity_mg)
            plsi_perplexities.append((mk, perplexity))
            plsi_mg_perplexities.append((mk, perplexity_mg))
        elif mtype == 'lda':
            # TODO: write LDA perplexity
            alpha, beta = params["alpha"], params["beta"]
            model = LDAModel(int(mk), beta=beta, alpha=alpha)
            var_ll = cperplexity(model, corpus, 'lda')
            perplexity = -1 * var_ll / total_words
            lda_perplexities.append((mk, perplexity))

        if mtype == 'plsi':
            print "Marg PLSI Perplexity: {0}".format(perplexity_mg)
        print "Model={0}, K={1}, Perplexity={2}".format(mtype, mk, perplexity)

    # sort perplexities by num topics
    plsi_perplexities = sorted(plsi_perplexities, key=lambda x: x[0])
    lda_perplexities = sorted(lda_perplexities, key= lambda x: x[0])
    plsi_perplexities = [list(t) for t in zip(*plsi_perplexities)]
    lda_perplexities = [list(t) for t in zip(*lda_perplexities)]

    # plotting
    line_plsi, = plt.plot(plsi_perplexities[0], plsi_perplexities[1], 'b^')
    plt.plot(plsi_perplexities[0], plsi_perplexities[1], 'k')
    line_lda, = plt.plot(lda_perplexities[0], lda_perplexities[1], 'rs')
    plt.plot(lda_perplexities[0], lda_perplexities[1], 'r--')
    plt.title("Perplexity vs. Num Topics for Various Models")
    plt.legend(handles=[line_plsi, line_lda])
    plt.show()

def load_vocab(model_file):
    vocabf = None
    if 'ap' in model_file and 'train' in model_file:
        vocabf = open('vocab/ap_train.txt', 'r')
    elif 'sn' in model_file and 'train' in model_file:
        vocabf = open('vocab/sn_train.txt', 'r')
    vocab = eval(vocabf.read())
    vocabf.close()
    return vocab


def test_best_topic_words(model_dir, result_dir, num_words=10):
    vocab_file = None
    model_files = os.listdir(model_dir)
    model_files = [model_dir + "/" + model_file for model_file in model_files]
    for model_file in model_files:
        last_part = model_file.split("/")[-1]
        print "\n", last_part, 'words'
        params = dmg.load_model_params(model_file)
        if 'plsi' in model_file:
            word_gens = params['pi']
        else:
            word_gens = params["beta"]
        vocab = load_vocab(last_part)
        vocab_size = word_gens.shape[0]
        index = {v: k for (k, v) in vocab.items()}
        best_words = np.argpartition(word_gens, -num_words, axis=0)
        best_words = best_words[-num_words:, :].T
        best_words = [[index[idx] for idx in words] for words in best_words]
        best_strings = [",".join(words) for words in best_words]
        best_strings = "\n".join(best_strings)
        result_file = open(result_dir + "/bw_" + last_part[:-4] + ".txt", 'w')
        print "\n", best_strings
        result_file.write(best_strings)
        result_file.close()

def test_word_feature_reps(model_dir, num_words=500):
    model_files = os.listdir(model_dir)
    model_files = [model_dir + "/" + model_file for model_file in model_files]
    for model_file in model_files:
        last_part = model_file.split("/")[-1]
        print "\n", last_part, 'word features'
        params = dmg.load_model_params(model_file)
        if 'plsi' in model_file:
            word_gens = params['pi']
        else:
            word_gens= params["beta"]
        vocab = load_vocab(last_part)
        index = {v: k for (k, v) in vocab.items()}
        best_words = np.argpartition(word_gens, -500, axis=0)
        best_words = best_words[-500:,:]
        rand_idx = np.random.choice(best_words.shape[0], num_words)
        rand_words = best_words[rand_idx, :]
        #rand_idx = np.random.choice(word_gens.shape[0], num_words)
        #rand_words = word_gens[rand_idx, :]
        tsne = TSNE(n_components=2, random_state=0)
        projected = tsne.fit_transform(rand_words)
        # plotting
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111)
        ax.plot(projected[:,0], projected[:,1],linestyle='None',marker='o',markersize=1)
        for i in range(projected.shape[0]):
            feat1, feat2 = projected[i, 0], projected[i, 1]
            word = index[rand_idx[i]].decode('utf-8')
            chars = []
            for j in word:
                if ord(j) < 128:
                    chars.append(j)
            word == ''.join(chars)
            print word
            ax.text(feat1, feat2, word, fontsize=15)
        plt.title(last_part)
        plt.show()
