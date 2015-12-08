import numpy as np
import optimization as opt
import scipy.special

def digamma_f(x):
    return scipy.special.psi(x)

def gamma_f(x):
    return scipy.special.gamma(x)

def polygamma_f(n, x):
    return scipy.special.polygamma(n, x)

class UnigramMixture(object):
    def __init__(self, k, beta=None, pi=None, vocab=None):
        self.num_topics = k
        if beta is None or pi is None or vocab is None:
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

    def seed(self, freqs):
        num_docs, V = len(freqs), freqs[0].shape[0]
        beta = np.zeros((V, self.num_topics))
        for i in range(self.num_topics):
            sample_idx = np.random.choice(num_docs, 5, replace=False)
            for doc_idx in sample_idx:
                freq = freqs[doc_idx]
                beta[:, i] += freq
        beta = beta + 1
        beta /= np.sum(beta, axis=0)
        return beta

    def train(self, corpus):
        # TODO: make threshold variable for testing?
        def convergence(old_ll, new_ll):
            return (np.abs((new_ll - old_ll) / old_ll) < (0.00001))

        docmats, idxs, freqs = corpus["docmats"], corpus["idxs"], corpus["freqs"]
        num_docs, V = len(docmats), docmats[0].shape[1]

        # initial topic -> word probabilities
        # initial topic probabilities
        wgen = self.seed(freqs)
        tgen = np.random.rand(self.num_topics)
        tgen = tgen / tgen.sum()

        # likelihood tracking
        prev_ll, curr_ll = None, None
        # threshold = ...

        while (prev_ll is None or not convergence(prev_ll, curr_ll)):
            total_pi = np.zeros(self.num_topics)
            total_beta = np.zeros((V, self.num_topics))
            # loop through documents to avoid space complexity of functional style
            for i in range(num_docs):
                docmat, idx, freq = docmats[i], idxs[i], freqs[i]
                # doc e-step: compute posteriors
                w_tcond = wgen[idx, :]
                simptest = np.log(w_tcond)
                t_wpost = tgen * (w_tcond.prod(axis=0))
                #print 'current post: {0}'.format(t_wpost)
                t_wpost = t_wpost / t_wpost.sum()

                # doc contribution to m-step
                total_pi += t_wpost
                total_beta += freq.dot(t_wpost.reshape(1, self.num_topics))

            # finish m-step: new parameters
            tgen = total_pi / num_docs
            wgen = total_beta / total_pi

            # new likelihood computation
            prev_ll = curr_ll
            print "LL update; old: {0}, new: {1}".format(prev_ll, curr_ll)
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

class PLSIModel(object):
    def __init__(self, k, beta=None, pi=None, vocab=None):
        self.num_topics = k
        if beta is None or pi is None or vocab is None:
            self.p_topic_to_word = None
            self.p_doc_to_topic = None
            self.vocab = None
        else:
            self.p_topic_to_word = beta
            self.p_doc_to_topic = pi
            self.vocab = vocab

    def get_params():
        params = {
            'pi': self.p_doc_to_topic,
            'beta': self.p_topic_to_word,
            'vocab': self.vocab
        }
        return params

    def seed(self, freqs):
        num_docs, V = len(freqs), freqs[0].shape[1]
        beta = np.zeros((V, self.num_topics))
        for i in range(self.num_topics):
            sample_idx = np.random.choice(num_docs, 5, replace=False)
            for doc_idx in sample_idx:
                freq = freqs[doc_idx]
                beta[:, i] += freq
        beta = beta + 1
        beta /= np.sum(beta, axis=0)
        return beta

    def train(self, corpus):
        # TODO: make threshold variable for testing?
        def convergence(old_ll, new_ll):
            return (np.abs((new_ll - old_ll) / old_ll) <= (0.00001))

        docmats, idxs, freqs = corpus["docmats"], corpus["idxs"], corpus["freqs"]
        num_docs, V = len(docmats), docmats[0].shape[1]

        # initial topic -> word probabilities
        # initial topic probabilities
        wgen = self.seed(freqs)
        tgen = np.ones((self.num_topics, num_docs))
        tgen /= self.num_topics

        # likelihood tracking
        prev_ll, curr_ll = None, None
        # threshold = ...

        while (prev_ll is None or not convergence(prev_ll, curr_ll)):
            total_pi = np.zeros(self.num_topics)
            total_beta = np.zeros((V, self.num_topics))
            # loop through documents to avoid space complexity of functional style
            for i in range(num_docs):
                docmat, idx, freq = docmats[i], idxs[i], freqs[i]
                # doc e-step: compute posteriors
                t_wpost = wgen * tgen[:, i]
                norm = t_wpost.sum(axis = 1)
                t_wpost = t_wpost / norm[:, np.newaxis]
                # m-step for p(z|t)
                tgen[:, i] = (t_wpost.sum(axis=0) / docmat.shape[0])

                # m-step doc contribution for p(w|z)
                '''
                t_wfilter = t_wpost.T
                t_wfilter[:, freq==0] = 0
                t_wfilter = csr_matrix(t_wfilter)
                freqmat = lil_matrix((V, V))
                freqmat.setdiag(freq)
                total_beta += (((t_wfilter).dot(freqmat)).T)
                '''
                total_beta += t_wpost * freq

            # finish m-step: new parameters
            wgen = total_beta / (total_beta.sum(axis=0))

            # new likelihood computation
            prev_ll = curr_ll
            curr_ll = ll(docmats, idxs, tgen, wgen)
            print "LL update; old: {0}, new: {1}".format(prev_ll, curr_ll)

        self.p_topic_to_word = wgen
        self.p_doc_to_topic = tgen

    def likelihood(docmats, idxs, pi, beta):
        ll = 0
        for i in range(len(docmats)):
            ll += (np.log((pi[:, i] * beta[idx, :]).sum(axis=1))).sum()
        return ll

    def test(corpus):
        idxs = corpus['idxs']
        total_words = 0
        for i in range(len(idxs)):
            total_words += idxs[i].shape[0]
        total_log_prob = likelihood(corpus['docmats'], idxs, self.p_doc_to_topic, self.p_topic_to_word)
        perplexity = -1 * total_log_prob / total_words
        perplexity = np.exp(perplexity)
        return perplexity

class LDAModel(object):
    def __init__(self, k, beta=None, alpha=None, vocab=None):
        self.num_topics = k
        if beta is None or alpha is None or vocab is None:
            self.p_topic_to_word = None
            self.p_topic = None
            self.vocab = None
        else:
            self.p_topic_to_word = beta
            self.p_topic = alpha # actually more like p_p_topic...
            self.vocab = vocab

    def seed(corpus):
        num_docs = len(freqs)
        beta = np.zeros((V, self.num_topics))
        for i in range(self.num_topics):
            sample_idx = np.random.choice(num_docs, 5, replace=False)
            for doc_idx in sample_idx:
                freq = freqs[doc_idx]
                beta[:, i] += freq
        beta = beta + 1
        beta /= np.sum(beta, axis=0)
        return beta

    def train(corpus):
        docmats, freqs, idxs = corpus["docmats"], corpus["freqs"], corpus["idxs"]
        num_docs, V = len(docmats), docmats[0].shape[1]
        prev_ll, curr_ll = None, None
        prev_beta, curr_beta = None, self.seed(freqs)
        prev_alpha, curr_alpha = None, np.ones(self.num_topics)

        while not em_convergence(prev_ll, curr_ll):
            e_thetas = []
            prev_ll, curr_ll = curr_ll, 0
            prev_beta, curr_beta = curr_beta, np.zeros((V, k))
            prev_alpha, curr_alpha = curr_alpha, 0

            for i in range(len(num_docs)):
                docmat, idx = docmats[i], idxs[i]
                phi = np.ones((V, self.num_topics)) / self.num_topics
                gamma = prev_alpha + docmat.shape[0] / self.num_topics
                prev_bound, curr_bound = None, None

                while not var_convergence(prev_bound, curr_bound):
                    prev_phi = phi
                    prev_gamma = gamma
                    phi, gamma = self.__new_var_params(prev_phi, prev_gamma, alpha, beta, idx)

                    # compute new bound
                    sum_gamma = np.sum(gamma)
                    e_theta = digamma_f(gamma) - digamma_f(sum_gamma)
                    e_log_pt_a = np.log(gamma_f(np.sum(alpha)))
                    e_log_pt_a -= np.sum(np.log(gamma_f(alpha)))
                    e_log_pt_a += np.dot(alpha - 1, e_theta)

                    e_log_pz_t = np.sum(phi * e_theta)

                    e_log_pwz_b = np.sum(phi * np.log(prev_beta[idx, :]))

                    e_log_qt = np.log(gamma_f(sum_gamma))
                    e_log_qt -= np.sum(np.log(gamma_f(gamma)))
                    e_log_qt += np.dot(gamma - 1, e_theta)

                    e_log_qz = np.sum(phi * np.log(phi))

                    prev_bound = curr_bound
                    curr_bound = e_log_pt_a + e_log_pz_t + e_log_pwz_b - e_log_qt - e_log_qz
                curr_ll += (e_log_pz_t - e_log_qt - e_log_qz)
                e_thetas.append(e_theta)

                # space-efficient m-step: find beta
                curr_beta += (docmat.T).dot(phi)

            #m-step find alpha
            a_ll = self.__alpha_likelihood_gen(e_thetas)
            a_grad = self.__alpha_grad_gen(e_thetas)
            curr_alpha, obj = opt.newton(prev_alpha, a_ll, a_grad, a_hess)

            #m-step find new bound
            curr_ll += obj
            curr_ll += np.sum(curr_beta * np.log(curr_beta))

        self.p_topic_to_word = curr_beta
        self.p_topic = curr_alpha

    def test(corpus):
        docmats, idxs = corpus["docmats"], corpus["idxs"]
        num_docs, V = len(docmats), docmats[0].shape[1]
        alpha, beta  = self.p_topic, self.p_topic_to_word
        total_ll, total_words = 0, 0
        for i in range(len(num_docs)):
            docmat, idx = docmats[i], idxs[i]
            phi = np.ones((V, self.num_topics))
            gamma = alpha + docmat.shape[0]
            prev_bound, curr_bound = None, None

            while not var_convergence(prev_bound, curr_bound):
                prev_phi = phi
                prev_gamma = gamma
                phi, gamma = self.__new_var_params(prev_phi, prev_gamma, alpha, beta, idx)

                sum_gamma = np.sum(gamma)
                e_theta = digamma_f(gamma) - digamma_f(sum_gamma)
                e_log_pt_a = np.log(gamma_f(np.sum(alpha)))
                e_log_pt_a -= np.sum(np.log(gamma_f(alpha)))
                e_log_pt_a += np.dot(alpha - 1, e_theta)

                e_log_pz_t = np.sum(phi * e_theta)

                e_log_pwz_b = np.sum(phi * np.log(prev_beta[idx, :]))

                e_log_qt = np.log(gamma_f(sum_gamma))
                e_log_qt -= np.sum(np.log(gamma_f(gamma)))
                e_log_qt += np.dot(gamma - 1, e_theta)

                e_log_qz = np.sum(phi * np.log(phi))

                prev_bound = curr_bound
                curr_bound = e_log_pt_a + e_log_pz_t + e_log_pwz_b - e_log_qt - e_log_qz
            total_ll += curr_bound
            total_words += docmat.shape[0]
        perplexity = np.exp(-1 * total_ll / total_words)
        return perplexity

    def __em_convergence(self, old_ll, curr_ll):
        if old_ll is None or curr_ll is None:
            return False
        elif (np.abs((curr_ll - old_ll) / old_ll) > 0.00001):
            return False
        return True

    def __var_convergence(self, old_bound, curr_bound):
        if old_bound is None or curr_bound is None:
            return False
        elif (np.abs((curr_bound - old_bound) / old_bound) > 0.00001):
            return False
        return True

    def __new_var_params(self, old_phi, old_gamma, alpha, beta, idx):
        phi = beta[idx, :] * digamma_f(old_gamma)
        norm_phi = np.sum(phi, axis=1)
        phi = phi / norm_phi[:, np.newaxis]
        gamma = alpha + np.sum(phi, axis=0)
        return phi, gamma

    def __alpha_likelihood_gen(self, e_thetas):
        def alpha_likelihood(alpha):
            alpha_obj = np.log(gamma_f(np.sum(alpha)))
            alpha_obj -= np.sum(np.log(gamma_f(alpha)))
            alpha_ll = M * alpha_obj
            for e_theta in e_thetas:
                alpha_ll += np.dot(alpha-1, e_theta)
            return alpha_ll
        return alpha_likelihood

    def __alpha_grad_gen(self, e_thetas):
        def alpha_grad(alpha):
            grad = digamma_f(np.sum(alpha)) - digamma_f(alpha)
            grad *= M
            grad += sum(e_thetas)
            return grad
        return alpha_grad

    def __alpha_hess(self, alpha):
        hess = np.diag(M * polygamma_f(alpha))
        hess -= polygamma_f(np.sum(alpha))
        return hess
