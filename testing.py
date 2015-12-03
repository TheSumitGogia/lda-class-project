import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import data_mgmt as dmg
import os

def train_test_models(data_dir, target_dir, ntopics=[5, 10, 20, 50, 100]):
    corpora = os.listdir(data_dir)
    for corpus_file in corpora:
        corpus = dmg.load_corpus(data_dir + "/" + corpus_file)
        for num_topics in ntopics:
            um_model = UnigramMixture(num_topics)
            um_model.train(corpus)
            dmg.save_model_params(um_model.get_params(), target_dir + "/" + corpus_file[:-4] + ".mat")

            lda_model = LDAModel(num_topics)
            lda_model.train(corpus)
            dmg.save_model_params(lda_model.get_params(), target_dir + "/" + corpus_file[:-4] + ".mat")

def perplexity(model, corpus):
    return model.test(corpus)

def test_perplexity_vs_ntopics(corpus, model_dir):
    um_perplexities = []
    plsi_perplexities = []
    lda_perplexities = []

    model_files = os.listdir(model_dir)
    model_full_files = [model_dir + "/" + model_file for model_file in model_files]
    idxs = corpus["idxs"]
    for i in range(len(model_files)):
        model_fname = model_files[i]
        model_fullname = model_full_files[i]
        model_split = model_fname.split('_')
        mtype, mcorp, mk = model_split[0], model_split[1], model_split[2][:-4]
        params = dmg.load_model_params(model_fullname)
        # TODO: perhaps clean by allowing model to be instantiated from params

        if mtype == 'um':
            beta, pi = params["beta"], params["pi"]
            total_words = 0
            total_log_prob = 0
            for j in range(len(idxs)):
                idx = idxs[j]
                w_tconds = beta[idx, :]
                w_tconds = w_tconds.prod(axis=0)
                total_log_prob += w_tconds.dot(pi)
                total_words += idx.shape[0]
            perplexity = -1 * total_log_prob / total_words
            perplexity = np.exp(perplexity)
            um_perplexities.append((mk, perplexity))
        elif mtype == 'lda':
            # TODO: write LDA perplexity
            perplexity = 1
            lda_perplexities.append((mk, perplexity))

        print "Model={0}, K={1}, Perplexity={2}".format(mtype, mk, perplexity)

    # sort perplexities by num topics
    um_perplexities = sorted(um_perplexities, key=lambda x: x[0])
    plsi_perplexities = sorted(plsi_perplexities, key=lambda x: x[0])
    lda_perplexities = sorted(lda_perplexities, key= lambda x: x[0])
    um_perplexities = [list(t) for t in zip(*um_perplexities)]
    plsi_perplexities = [list(t) for t in zip(*plsi_perplexities)]
    lda_perplexities = [list(t) for t in zip(*lda_perplexities)]

    # plotting
    line_um, = plt.plot(um_perplexities[0], um_perplexities[1], 'b^')
    plt.plot(um_perplexities[0], um_perplexities[1], 'k')
    line_lda, = plt.plot(lda_perplexities[0], lda_perplexities[1], 'rs')
    plt.plot(lda_perplexities[0], lda_perplexities[1], 'r--')
    plt.title("Perplexity vs. Num Topics for Various Models")
    plt.legend(handles=[line_um, line_lda])
    plt.show()

def test_best_topic_words(model_dir, result_dir, num_words=10):
    model_files = os.listdir(model_dir)
    model_files = [model_dir + "/" + model_file for model_file in model_files]
    for model_file in model_files:
        last_part = model_file.split["/"][-1]
        params = dmg.load_model_params(model_file)
        word_gens, vocab = params["beta"], params["vocab"]
        vocab_size = word_gens.shape[0]
        best_words = np.argpartition(word_gens, vocab_size - num_words, axis=1)
        best_words = best_words[vocab_size-num_words-1:, :].T
        best_words = [[vocab[idx] for idx in words] for words in best_words]
        best_strings = [",".join(words) for words in best_words]
        best_strings = "\n".join(best_strings)
        result_file = open(result_dir + "/bw_" + last_part[:-4] + ".csv", 'w')
        print best_strings
        result_file.write("\n".join(best_strings))

def test_word_feature_reps(model_dir, num_words=500):
    model_files = os.listdir(model_dir)
    model_files = [model_dir + "/" + model_file for model_file in model_files]
    for model_file in model_files:
        last_part = model_file.split["/"][-1]
        params = dmg.load_model_params(model_file)
        word_gens, vocab = params["beta"], params["vocab"]
        rand_idx = np.random.choice(word_gens.shape[0], num_words)
        rand_words = word_gens[rand_idx, :]
        tsne = TSNE(n_components=2, random_state=0)
        projected = tsne.fit_transform(rand_words)

        # plotting
        for i in range(projected.shape[0]):
            feat1, feat2 = projected[i, 0], projected[i, 1]
            word = vocab[rand_words[i]]
            plt.scatter(feat1, feat2, s=5, marker=r"$ {} $".format(word), edgecolors='none')
        plt.title(last_part)
        plt.show()
