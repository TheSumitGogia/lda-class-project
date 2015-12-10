import numpy as np
import optimization as opt
import scipy.special
from scipy.misc import logsumexp
from collections import defaultdict

def sanity_check(mat):
    if np.any(mat == np.inf):
        return False
    if np.any(mat == -np.inf):
        return False
    if np.any(mat == np.nan):
        return False
    return True

def digamma_f(x):
    return scipy.special.psi(x)

def gamma_f(x):
    return scipy.special.gamma(x)

def gammaln_f(x):
    return scipy.special.gammaln(x)

def polygamma_f(n, x):
    return scipy.special.polygamma(n, x)

def bls(curr, change, gradient, e_thetas, alpha=0.01, beta=0.9, maxcount=30):
    def func(curr):
        alpha_obj = gammaln_f(np.sum(curr))
        alpha_obj -= np.sum(gammaln_f(curr))
        alpha_ll = len(e_thetas) * alpha_obj
        for e_theta in e_thetas:
            alpha_ll += np.dot(curr-1, e_theta)
        return alpha_ll
    t, guess = 1, None

    while (guess is None or guess <= bound):
        guess = func(curr + t * change)
        bound = func(curr) + alpha * t * (gradient.T.dot(change))
        t = beta * t
    return t

def newton(init_alpha, e_thetas, tolerance=0.00001, maxiter=100):
    dist_est = None
    current = init_alpha.copy()
    grad_part = sum(e_thetas)
    M = len(e_thetas)
    sm = np.sum(current)
    count = 0
    while (count < maxiter and (dist_est is None or dist_est > tolerance)):
        sm = np.sum(current)
        grad = M * (digamma_f(sm) - digamma_f(current))
        grad += grad_part
        hess = M * polygamma_f(1, current)
        c = np.sum(grad / hess)
        c /= (polygamma_f(1, sm) + np.sum(1 / hess))
        change = (grad - c) / hess
        step = bls(current, change, grad, e_thetas)
        current = current + step*change
        dist_est = (grad.dot(change)) ** 0.5
        count += 1
    return current, (count >= maxiter)

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
        for i in range(num_docs):
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
        if beta is None or pi is None:
            self.p_topic_to_word = None
            self.p_doc_to_topic = None
            self.vocab = None
        else:
            self.p_topic_to_word = beta
            self.p_doc_to_topic = pi
            #self.vocab = vocab

    def get_params(self):
        params = {
            'pi': self.p_doc_to_topic,
            'beta': self.p_topic_to_word
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

    def e_step(self, idxs, V, theta_t_z, theta_z_w):
        count_t_z = np.zeros((len(idxs), self.num_topics))
        count_w_z = np.zeros((V, self.num_topics))
        for t in range(len(idxs)):
            idx = idxs[t]
            posterior_w_z = defaultdict(lambda:np.zeros(self.num_topics))
            for w in idx:
                if w not in posterior_w_z:
                    posterior_w_z[w] = np.multiply(theta_t_z[t, :], (theta_z_w[:, w]).T)
                    posterior_w_z[w] /= np.sum(posterior_w_z[w])
                for z in range(self.num_topics):
                    count_t_z[t, z] += posterior_w_z[w][z]
                    count_w_z[w, z] += posterior_w_z[w][z]
        return count_t_z, count_w_z

    def m_step(self, numdocs, V, count_t_z, count_w_z):
        theta_t_z = np.zeros((numdocs, self.num_topics))
        theta_z_w = np.zeros((self.num_topics, V))
        for t in range(count_t_z.shape[0]):
            count_col = count_t_z[t, :]
            theta_t_z[t, :] = count_col / count_col.sum()
        for z in range(count_w_z.shape[1]):
            count_row = count_w_z[:, z]
            theta_z_w[z, :] = count_row / count_row.sum()

        return theta_t_z, theta_z_w

    def train(self, corpus, numiter=100):
        # TODO: make threshold variable for testing?
        def convergence(old_ll, new_ll):
            return (np.abs((new_ll - old_ll) / old_ll) <= (0.00001))

        idxs, freqs = corpus["idxs"], corpus["freqs"]
        num_docs, V = len(idxs), freqs[0].shape[1]

        # initial topic -> word probabilities
        # initial topic probabilities

        #theta_z_w = self.seed(freqs).T
        '''
        theta_z_w = np.random.rand(self.num_topics, V)
        theta_t_z  = np.random.rand(num_docs, self.num_topics)
        for t in range(num_docs):
            theta_t_z[t] /= np.sum(theta_t_z[t])
        for z in range(self.num_topics):
            theta_z_w[z] /= np.sum(theta_z_w[z])
        '''
        wgen = self.seed(freqs)
        tgen = np.random.rand(self.num_topics, num_docs)

        # likelihood tracking
        prev_ll, curr_ll = None, None
        # threshold = ...

        it = 0
        '''
        for i in range(numiter):
            print "Iteration", i+1, '...'
            count_t_z, count_w_z = self.e_step(idxs, V, theta_t_z, theta_z_w)
            theta_t_z, theta_z_w = self.m_step(num_docs, V, count_t_z, count_w_z)
            prev_ll = curr_ll
            curr_ll = self.likelihood(idxs, theta_t_z.T, theta_z_w.T)
            print 'old ll: {0}, new ll: {1}'.format(prev_ll, curr_ll)
        '''
        while (prev_ll is None or not convergence(prev_ll, curr_ll)):
            total_beta = np.zeros((V, self.num_topics))
            # loop through documents to avoid space complexity of functional style
            for i in range(num_docs):
                idx, freq = idxs[i], freqs[i].toarray()[0]

                # doc e-step: compute posteriors
                t_wpost = np.zeros((wgen.shape))
                t_wpost[idx, :] = (wgen[idx, :] * tgen[:, i])
                norm = t_wpost[idx, :].sum(axis = 1)
                t_wpost[idx, :] = t_wpost[idx, :] / norm[:, np.newaxis]
                # m-step for p(z|t)
                tgen[:, i] = (t_wpost.sum(axis=0) / freq.shape[0])

                # m-step doc contribution for p(w|z)
                '''
                t_wfilter = t_wpost.T
                t_wfilter[:, freq==0] = 0
                t_wfilter = csr_matrix(t_wfilter)
                freqmat = lil_matrix((V, V))
                freqmat.setdiag(freq)
                total_beta += (((t_wfilter).dot(freqmat)).T)
                '''
                total_beta[idx, :] += t_wpost[idx, :] * freq[idx, np.newaxis]

            # finish m-step: new parameters
            wgen = total_beta / (total_beta.sum(axis=0))

            # new likelihood computation
            prev_ll = curr_ll
            curr_ll = self.likelihood(idxs, tgen, wgen)
            it += 1
            #print "Iteration {0} Complete".format(it)
            #print "LL update; old: {0}, new: {1}".format(prev_ll, curr_ll)

        print "Likelihood on training set: {0}".format(curr_ll)
        self.p_topic_to_word = tgen
        self.p_doc_to_topic = wgen

    def foldin(self, corpus):
        def convergence(old_ll, new_ll):
            return (np.abs((new_ll - old_ll) / old_ll) <= (0.00001))

        idxs, freqs = corpus["idxs"], corpus["freqs"]
        num_docs, V = len(idxs), freqs[0].shape[1]
        wgen = self.p_doc_to_topic #TODO: fix ordering...
        tgen = np.random.rand(self.num_topics, num_docs)
        prev_ll, curr_ll = None, None

        while (prev_ll is None or not convergence(prev_ll, curr_ll)):
            for i in range(num_docs):
                idx, freq = idxs[i], freqs[i].toarray()[0]
                t_wpost = np.zeros((wgen.shape))
                t_wpost[idx, :] = (wgen[idx, :] * tgen[:, i])
                norm = t_wpost[idx, :].sum(axis = 1)
                t_wpost[idx, :] = t_wpost[idx, :] / norm[:, np.newaxis]
                tgen[:, i] = (t_wpost.sum(axis=0) / freq.shape[0])
            prev_ll = curr_ll
            curr_ll = self.likelihood(idxs, tgen, wgen)
            #print 'bound updates: {0}, {1}'.format(prev_ll, curr_ll)
        return curr_ll

    def mg_test_ll(self, corpus):
        # uniform probability distribution over documents (possibly Poisson is more helpful)
        idxs = corpus['idxs']
        pi, beta = self.p_topic_to_word, self.p_doc_to_topic
        total_ll = 0
        for i in range(len(idxs)):
            new_doc_l = 0
            # marginalize over training documents
            for j in range(pi.shape[1]):
                mid_ll = np.log((pi[:, j] * beta[idxs[i], :]).sum(axis=1)).sum()
                new_doc_l += (np.exp(mid_ll + np.log(1.0 / pi.shape[1])))
            total_ll += np.log(new_doc_l)
        return total_ll

    def likelihood(self, idxs, pi, beta):
        ll = 0
        #pi_marg = pi / (pi.sum(axis=1)[:, np.newaxis])
        for i in range(len(idxs)):
            ll += np.log((pi[:, i] * beta[idxs[i], :]).sum(axis=1)).sum()
        return ll

    def test(self, corpus):
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
        if beta is None or alpha is None:
            self.p_topic_to_word = None
            self.p_topic = None
        else:
            self.p_topic_to_word = beta
            self.p_topic = alpha[0] # actually more like p_p_topic...

    def get_params(self):
        params = {
            'beta': self.p_topic_to_word,
            'alpha': self.p_topic
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
        docmats, freqs, idxs = corpus["docmats"], corpus["freqs"], corpus["idxs"]
        num_docs, V = len(docmats), docmats[0].shape[1]
        prev_ll, curr_ll = None, None
        prev_beta, curr_beta = None, np.log(self.seed(freqs))
        prev_alpha, curr_alpha = None, np.ones(self.num_topics)

        it = 0
        try:
            while not self.__em_convergence(prev_ll, curr_ll):
                print 'Starting EM iteration {0}'.format(it)
                e_thetas = []
                prev_ll, curr_ll = curr_ll, 0
                prev_beta, curr_beta = curr_beta, np.zeros((V, self.num_topics))
                prev_alpha, curr_alpha = curr_alpha, 0

                for i in range(num_docs):
                    docmat, idx = docmats[i], idxs[i]
                    phi = np.ones((docmat.shape[0], self.num_topics)) / self.num_topics
                    gamma = prev_alpha + docmat.shape[0] * 1.0 / self.num_topics
                    prev_bound, curr_bound = None, None

                    while not self.__var_convergence(prev_bound, curr_bound):
                        prev_phi = phi
                        prev_gamma = gamma
                        phi, gamma = self.__new_var_params(prev_phi, prev_gamma, prev_alpha, prev_beta, idx)

                        # compute new bound
                        sum_gamma = np.sum(gamma)
                        e_theta = digamma_f(gamma) - digamma_f(sum_gamma)
                        e_log_pt_a = gammaln_f(np.sum(prev_alpha))
                        e_log_pt_a -= np.sum(gammaln_f(prev_alpha))
                        e_log_pt_a += np.dot(prev_alpha - 1, e_theta)

                        e_log_pz_t = np.sum(phi * e_theta)

                        e_log_pwz_b = np.sum(phi * prev_beta[idx, :])

                        e_log_qt = gammaln_f(sum_gamma)
                        e_log_qt -= np.sum(gammaln_f(gamma))
                        e_log_qt += np.dot(gamma - 1, e_theta)

                        e_log_qz = np.sum(phi * np.log(phi))


                        prev_bound = curr_bound
                        curr_bound = e_log_pt_a + e_log_pz_t + e_log_pwz_b - e_log_qt - e_log_qz
                        #print 'bound updates: {0}, {1}'.format(prev_bound, curr_bound)
                    #print 'finished document {0}'.format(i)
                    #curr_ll += (e_log_pz_t - e_log_qt - e_log_qz)
                    curr_ll += curr_bound
                    print curr_bound, curr_ll
                    e_thetas.append(e_theta)

                    # space-efficient m-step: find beta
                    curr_beta = np.log(np.exp(curr_beta) + (docmat.T).dot(phi))

                #m-step find alpha
                a_ll = self.__alpha_likelihood_gen(e_thetas)
                #a_grad = self.__alpha_grad_gen(e_thetas)
                #a_hess = self.__alpha_hess_gen(e_thetas)
                curr_alpha, quit = newton(prev_alpha, e_thetas)
                curr_beta = curr_beta - logsumexp(curr_beta, axis=0)
                obj = a_ll(curr_alpha)

                #m-step find new bound
                #curr_ll += obj
                #curr_ll += np.sum(np.exp(curr_beta) * curr_beta)
                if quit: break
                it += 1
                print "bound changes: {0}, {1}".format(prev_ll, curr_ll)
        except KeyboardInterrupt:
            print 'Stopping training and saving parameters...'
            self.p_topic_to_word = np.exp(prev_beta)
            self.p_topic = prev_alpha
            raise KeyboardInterrupt


        self.p_topic_to_word = np.exp(curr_beta)
        self.p_topic = curr_alpha

    def test(self, corpus):
        docmats, idxs = corpus["docmats"], corpus["idxs"]
        num_docs, V = len(docmats), docmats[0].shape[1]
        alpha, beta  = self.p_topic, np.log(self.p_topic_to_word)
        total_ll = 0
        for i in range(num_docs):
            docmat, idx = docmats[i], idxs[i]
            phi = np.ones((docmat.shape[0], self.num_topics)) / self.num_topics
            gamma = alpha + docmat.shape[0] * 1.0 / self.num_topics
            prev_bound, curr_bound = None, None

            while not self.__var_convergence(prev_bound, curr_bound):
                prev_phi = phi
                prev_gamma = gamma
                phi, gamma = self.__new_var_params(prev_phi, prev_gamma, alpha, beta, idx)

                # compute new bound
                sum_gamma = np.sum(gamma)
                e_theta = digamma_f(gamma) - digamma_f(sum_gamma)
                e_log_pt_a = gammaln_f(np.sum(alpha))
                e_log_pt_a -= np.sum(gammaln_f(alpha))
                e_log_pt_a += np.dot(alpha- 1, e_theta)

                e_log_pz_t = np.sum(phi * e_theta)

                e_log_pwz_b = np.sum(phi * beta[idx, :])

                e_log_qt = gammaln_f(sum_gamma)
                e_log_qt -= np.sum(gammaln_f(gamma))
                e_log_qt += np.dot(gamma - 1, e_theta)

                e_log_qz = np.sum(phi * np.log(phi))


                prev_bound = curr_bound
                curr_bound = e_log_pt_a + e_log_pz_t + e_log_pwz_b - e_log_qt - e_log_qz
            total_ll += curr_bound
        return total_ll

    def __em_convergence(self, old_ll, curr_ll):
        if old_ll is None or curr_ll is None:
            return False
        elif old_ll > curr_ll:
            return True
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
        phi = beta[idx, :] + digamma_f(old_gamma)
        norm_phi = logsumexp(phi, axis=1)
        phi = phi - norm_phi[:, np.newaxis]
        phi = np.exp(phi)
        gamma = alpha + phi.sum(axis=0)
        return phi, gamma

    def __alpha_likelihood_gen(self, e_thetas):
        def alpha_likelihood(alpha):
            alpha_obj = gammaln_f(np.sum(alpha))
            alpha_obj -= np.sum(gammaln_f(alpha))
            alpha_ll = len(e_thetas) * alpha_obj
            for e_theta in e_thetas:
                alpha_ll += np.dot(alpha-1, e_theta)
            return alpha_ll
        return alpha_likelihood

    def __alpha_grad_gen(self, e_thetas):
        def alpha_grad(alpha):
            grad = digamma_f(np.sum(alpha)) - digamma_f(alpha)
            grad *= len(e_thetas)
            grad += sum(e_thetas)
            return (-1 * grad)
        return alpha_grad

    def __alpha_hess_gen(self, e_thetas):
        M = len(e_thetas)
        def alpha_hess(alpha):
            hess = np.diag(M * polygamma_f(1, alpha))
            hess -= polygamma_f(1, np.sum(alpha))
            return (-1 * hess)
        return alpha_hess

