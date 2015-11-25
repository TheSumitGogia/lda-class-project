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
    gamma, phi = [None] * len(documents), [None] * len(documents)
    ets, ezts = [None] * len(documents), [None] * len(documents)
    etgs, ezps = [None] * len(documents), [None] * len(documents)
    alpha, beta = alpha0, beta0
    alpha = np.random.rand(k) if alpha0 is None else alpha0
    beta = np.random.rand(k, V) if beta0 is None else beta0
    prev_bound, bound = None, None

    while prev_bound is None or np.abs(bound - prev_bound) < threshold:
        for doc_idx in range(len(documents)):
            document = documents[doc_idx]
            best_phi, best_gamma, et, ezt, etg, ezp = e_step(document, k, alpha, beta)
            gamma[doc_idx], phi[doc_idx] = best_phi, best_gamma
            ets[doc_idx], ezts[doc_idx], etgs[doc_idx], ezps[doc_idx] = eta, ezt, etg, ezp

        prev_bound = bound
        alpha, beta, eta, ewzb = m_step(documents, gamma, phi, alpha, ets)
        bound = (len(documents) * (eta + ewzb)) + sum(ezts) - sum(etgs) - sum(ezps)

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

def ll_bound_doc(document, phi, gamma, alpha, beta, save=False):
    # partition bound part 1 (E_q(log p(theta|alpha)))
    e_theta = digamma_f(gamma) - digamma_f(np.sum(gamma))
    p1_1 = np.sum(np.multiply((alpha - 1), e_theta))
    p1_2 = np.log(gamma_f(np.sum(alpha))) - np.sum(np.log(gamma_f(alpha)))
    p1 = p1_1 + p1_2

    # partition bound part 2 (E_q(log p(z|theta)))
    p2 = np.sum(np.dot(phi, e_theta))

    # partition bound part 3 (E_q(log p(w|z, beta)))
    p3 = np.sum(np.multiply(phi, np.dot(document, (np.log(beta).T))))

    # partition bound part 4 (E_q(log q(theta|gamma)))
    p4_1 = np.log(gamma_f(np.sum(gamma))) - np.sum(np.log(gamma_f(gamma)))
    p4_2 = np.sum(np.multiply(gamma - 1, e_theta))
    p4 = p4_1 + p4_2

    # partition bound part 5 (E_q(log q(z|phi)))
    p5 = np.sum(np.multiply(phi, np.log(phi)))

    # full partition bound from parts
    bound = (p1 + p2 + p3 - p4 - p5)
    if not save:
        return bound
    else:
        return bound, e_theta, p2, p4, p5

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
    bound, et, ezt, etg, ezp = ll_bound_doc(document, phi, gamma, alpha, beta, True)
    return phi, gamma, et, ezt, etg, ezp

def m_step(documents, gamma, phi, prev_alpha, ets):
    pre_beta = [np.dot(phi[i].T, documents[i]) for i in range(len(documents))]
    beta = sum(pre_beta)

    M = len(documents)
    def bound_alpha_lc(alpha):
        return bound_alpha(alpha, ets)
    def bound_alpha_grad_lc(alpha):
        return bound_alpha_grad(M, alpha, ets)
    alpha = newton(bound_alpha_lc, prev_alpha, bound_alpha_grad_lc, bound_alpha_hess)

    return alpha, beta

def bound_alpha(alpha, digamma_diffs):
    p1 = np.log(gamma_f(np.sum(alpha))) - np.sum(np.log(np.sum(gamma_f(alpha))))
    use_alpha = alpha - 1
    doc_parts = [np.sum(np.multiply(use_alpha, digamma_diff)) for digamma_diff in digamma_diffs]
    return (M*p1 + sum(doc_parts))

def bound_alpha_grad(M, alpha, digamma_diffs):
    p1 = M * (digamma_f(np.sum(alpha)) - digamma_f(alpha))
    p2 = sum(digamma_diffs)
    grad = p1 + p2
    return grad

def bound_alpha_hess(alpha):
    delta_mat = np.identity(alpha.shape[0])
    p1 = delta_mat * M * polygamma_f(1, alpha)
    p1 = np.tile(p1, (1, alpha.shape[0]))
    p2 = polygamma_f(1, np.sum(alpha))
    hess = p1 - p2
    return hess
