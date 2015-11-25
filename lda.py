import numpy as np
import scipy.special

def digamma_f(x):
    return scipy.special.psi(x)

def gamma_f(x):
    return scipy.special.gamma(x)

def polygamma_f(n, x):
    return scipy.special.polygamma(n, x)

def lda_em(corpus, k, alpha0=None, beta0=None):
    documents, V = corpus[0], corpus[1]
    # initialize gammas, phis
    gamma = [None] * len(documents)
    phi = [None] * len(documents)
    alpha, beta = alpha0, beta0
    alpha = np.random.rand(k) if alpha0 is None else alpha0
    beta = np.random.rand(k, V) if beta0 is None else beta0
    prev_bound, bound = None, None

    while prev_bound is None or np.abs(bound - prev_bound) < threshold:
        for doc_idx in range(len(documents)):
            document = documents[doc_idx]
            best_phi, best_gamma = e_step(document, k, alpha, beta)
            gamma[doc_idx], phi[doc_idx] = best_phi, best_gamma

        prev_bound = bound
        alpha, beta = m_step(documents, gamma, phi)
        bound = ll_bound(corpus, phi, gamma, alpha, beta)

    return alpha, beta

def ll_bound(corpus, phi, gamma, alpha, beta):
    documents = corpus[0]
    bound = 0
    for doc_idx in range(len(documents)):
        document = docuents[doc_idx]
        doc_phi = phi[doc_idx]
        doc_gamma = gamma[doc_idx]

        bound += ll_bound_doc(document, doc_phi, doc_gamma, alpha, beta)
    return bound

def ll_bound_doc(document, phi, gamma, alpha, beta):
    # partition bound part 1 (E_q(log p(theta|alpha)))
    e_theta = digamma_f(doc_gamma) - digamma_f(np.sum(doc_gamma))
    p1_1 = np.sum(np.multiply((alpha - 1), e_theta))
    p1_2 = np.log(gamma_f(np.sum(alpha))) - np.sum(np.log(gamma_f(alpha)))
    p1 = p1_1 + p1_2

    # partition bound part 2 (E_q(log p(z|theta)))
    p2 = np.sum(np.dot(doc_phi, e_theta))

    # partition bound part 3 (E_q(log p(w|z, beta)))
    p3 = np.sum(np.multiply(doc_phi, np.dot(document, (np.log(beta).T))))

    # partition bound part 4 (E_q(log q(theta|gamma)))
    p4_1 = np.log(gamma_f(np.sum(doc_gamma))) - np.sum(np.log(gamma_f(doc_gamma)))
    p4_2 = np.sum(np.multiply(doc_gamma - 1, e_theta))
    p4 = p4_1 + p4_2

    # partition bound part 5 (E_q(log q(z|phi)))
    p5 = np.sum(np.multiply(doc_phi, np.log(doc_phi)))

    # full partition bound from parts
    bound = (p1 + p2 + p3 - p4 - p5)
    return bound

def e_step(document, k, alpha, beta):
    word_idx = np.where(document == 1)[1]
    def convergence_criterion(pphi, phi, pgamma, gamma):
        if pphi is None or pgamma is None: return False
        old_ll_bound = ll_bound_doc(document, pphi, pgamma, alpha, beta)
        new_ll_bound = ll_bound_doc(document, phi, gamma, alpha, beta)
        if np.abs(new_ll_bound - old_ll_bound) < threshold:
            return True
        return False

    prev_phi, phi = None, np.ones((document.shape[0], k)) / k
    prev_gamma, gamma = None, alpha + document.shape[0] / k
    while not convergence_criterion(prev_phi, phi, prev_gamma, gamma):
        prev_phi = phi
        prev_gamma = gamma
        phi = np.multiply((beta[:, word_idx]).T, digamma_f(prev_gamma))
        phi = (phi.T / np.sum(phi, axis=1)).T
        gamma = alpha + np.sum(phi, axis=0)
    return phi, gamma

def m_step(documents, gamma, phi):
    pre_beta = [np.dot(phi[i].T, documents[i]) for i in range(len(documents))]
    beta = sum(pre_beta)

    M = len(documents)
    alpha = newton(bound_alpha, prev_alpha, alpha_grad, alpha_hess)

    return alpha, beta

def bound_alpha(gamma, alpha):
    p1 = np.log(gamma_f(np.sum(alpha))) - np.sum(np.log(np.sum(gamma_f(alpha))))
    gamma_sums = [np.sum(doc_gamma) for doc_gamma in gamma]
    def bound_doc(idx):
        doc_gamma, doc_gamma_sum, doc_phi, = gamma[idx], gamma_sums[idx], phi[idx]
        return np.sum(np.multiply(alpha - 1, digamma_f(doc_gamma) - digamma(doc_gamma_sum)))
    doc_bounds = [bound_doc[idx] for idx in range(len(gamma))]
    return (M*p1 + sum(doc_bounds))

def bound_alpha_grad(gamma, alpha):
    gamma_sums = [np.sum(doc_gamma) for doc_gamma in gamma]
    p1 = M * (digamma_f(np.sum(alpha)) - digamma_f(alpha))
    p2 = sum([digamma_f(gamma[idx]) - digamma_f(gamma_sums[idx]) for idx in range(len(gamma))])
    grad = p1 + p2
    return grad

def bound_alpha_hess(alpha):
    delta_mat = np.identity(alpha.shape[0])
    p1 = delta_mat * M * polygamma_f(1, alpha)
    p1 = np.tile(p1, (1, alpha.shape[0]))
    p2 = polygamma_f(1, np.sum(alpha))
    hess = p1 - p2
    return hess
