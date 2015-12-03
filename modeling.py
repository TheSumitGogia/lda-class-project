import numpy as np

class UnigramMixture(object):
    def __init__(self, k, beta=None, pi=None, vocab=None):
        self.num_topics = k
        if beta is None or alpha is None or vocab is None:
            self.p_topic_to_word = None
            self.p_topic = None
            self.vocab = None
        else:
            self.p_topic_to_word = beta
            self.p_topic = pi
            self.vocab = vocab

    def get_params():
        params = {
            'pi': self.p_topic,
            'beta': self.p_topic_to_word,
            'vocab': self.vocab
        }
        return params

    def seed(corpus):
        pass

    def train(corpus):
        # TODO: make threshold variable for testing?
        def convergence(old_ll, new_ll):
            return (np.abs((new_ll - old_ll) / old_ll) < (0.001 / 100))

        docmats, idxs, freqs = corpus["docmats"], corpus["idxs"], corpus["freqs"]
        num_docs, V = len(docmats), docmats[0].shape[0]

        # initial topic -> word probabilities
        # initial topic probabilities
        wgen = np.random.rand(self.num_topics, V)
        wgen = (wgen / wgen.sum(axis=0)).T
        tgen = np.random.rand(self.num_topics)
        tgen = tgen / tgen.sum()

        # likelihood tracking
        prev_ll, curr_ll = None, None
        # threshold = ...

        while (prev_ll is None or not convergence(prev_ll, curr_ll)):
            total_pi = np.zeros(self.num_topics)
            total_beta = np.zeros((V, self.num_topics))
            # loop through documents to avoid space complexity of functional style
            for i in range(len(num_docs)):
                docmat, idx, freq = docmats[i], idxs[i], freqs[i]
                # doc e-step: compute posteriors
                w_tcond = wgen[idx, :]
                t_wpost = tgen * (w_tcond.prod(axis=0))
                t_wpost = t_wpost / t_wpost.sum()

                # doc contribution to m-step
                total_pi += t_wpost
                total_beta += freq.dot(np.reshape(t_wpost))

            # finish m-step: new parameters
            tgen = total_pi / num_docs
            wgen = wgen_sum / total_pi

            # new likelihood computation
            prev_ll = curr_ll
            curr_ll = np.sum(total_beta * np.log(wgen)) + np.sum(total_pi * np.log(tgen))

        self.p_topic_to_word = wgen
        self.p_topic = tgen

    def test(corpus):
        total_words = 0
        total_log_prob = 0
        for i in range(len(num_docs)):
            idx = idxs[i]
            w_tconds = self.p_topic_to_word[idx, :]
            w_tconds = w_tconds.prod(axis=0)
            total_log_prob += w_tconds.dot(self.p_topic)
            total_words += idx.shape[0]
        perplexity = -1 * total_log_prob / total_words
        perplexity = np.exp(perplexity)
        return perplexity
