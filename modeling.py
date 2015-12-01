class UnigramMixture(object):
    def __init__(self, k):
        self.num_topics = k

    def train(data):
        # initial guess for documents
        # initial guess for word probabilities

        # e-step: compute posteriors
        word_gen = [word_gens[word_idx, :] for word_idx in doc_word_idx]
        cond_topics = [topic_gens * (wgen.prod(axis=0)) for wgen in word_gen]
        cond_topics = [topics / topics.sum() for topics in cond_topics]

        # e-step: expectation of guess likelihood
        gl_expect = [(np.log(word_gen[i]) * cond_topics[i]).sum() + (np.log(topic_gens) * (cond_topics)).sum() for i in range(len(cond_topics))]
        gl_expect = sum(gl_expect)

        # m-step
        word_gens_up = [np.outer(doc_freqs[i], cond_topics[i]) for i in range(len(doc_freqs))]
        word_gens_up = sum(word_gens_up)
        total = sum(cond_topics)
        word_gens_up = word_gens_up / total

        topic_gens_up = total / total.sum()
        pass

    def test(data):
        pass


