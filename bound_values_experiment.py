from bayesian_regression import BayesianRegression

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log, exp

from matplotlib import rc
rc('text', usetex=True)


def main():
    np.random.seed(45)

    nb_test = 100000
    nb_dim = 20
    noise = 1/3
    sigma_prior = .1
    sigma_post = sqrt(2)
    w_norm = .5

    bound_param_a = 1.
    bound_param_b = 4.

    bound_param_c = (sigma_prior ** 2) / (sigma_post ** 2)
    bound_param_s = sqrt(nb_dim * sigma_prior**2 + w_norm**2 + (1 - bound_param_c) * noise**2) / sigma_post

    print("Bound parameters: a=%f, b=%f, s=%f, c=%f" % (bound_param_a, bound_param_b, bound_param_s, bound_param_c))

    assert bound_param_c < 1

    def create_data(n, _w=None):
        return create_linear_data(n, nb_dim, w=_w, sigma_noise=noise, w_norm=w_norm)

    print('Creating test data...')
    x_test, y_test, w_data = create_data(nb_test)

    nb_train_list = np.logspace(1, 6, 20, dtype=int)
    nb_params_train = len(nb_train_list)

    catoni_list, alquier_sqrt_list, alquier_n_list, subgamma_list,  train_loss_list, test_loss_list, kl_list = [np.zeros(nb_params_train) for _ in range(7)]

    print('Creating training data...')
    x_all, y_all, _ = create_data(nb_train_list[-1], w_data)

    for i, nb_train in enumerate(nb_train_list):
        x, y = x_all[:nb_train, :], y_all[:nb_train]
        print('Training on %d datapoints...' % nb_train)
        clf = BayesianRegression(sigma_prior, sigma_post)
        clf.fit(x, y)

        train_loss_list[i] = clf.calc_gibbs_nll_loss(x, y)
        test_loss_list[i] = clf.calc_gibbs_nll_loss(x_test, y_test)

        kl_list[i] = clf.calc_kullback_leibler()
        alquier_sqrt_list[i] = bound_alquier_sqrt(train_loss_list[i], kl_list[i], nb_train, bound_param_a, bound_param_b)
        alquier_n_list[i] = bound_alquier_n(train_loss_list[i], kl_list[i], nb_train, bound_param_a, bound_param_b)
        catoni_list[i] = bound_catoni(train_loss_list[i], kl_list[i], nb_train, bound_param_a, bound_param_b)
        subgamma_list[i] = bound_subgamma(train_loss_list[i], kl_list[i], nb_train, bound_param_s, bound_param_c)

    print('done!')

    plt.semilogx(nb_train_list, alquier_n_list, 'gray', linestyle='-', linewidth=4, marker='o', markersize=10, label=r"Alquier et al's $[a,b]$ bound (Theorem 3 + Eq 13)")
    plt.semilogx(nb_train_list, alquier_sqrt_list, '-', linewidth=4, marker='o', markersize=10, label=r"Alquier et al's $[a,b]$ bound (Theorem 3 + Eq 14)")
    plt.semilogx(nb_train_list, catoni_list, '-', linewidth=4, marker='o', markersize=10, label=r"Catoni's $[a,b]$ bound (Corollary 2)")
    plt.semilogx(nb_train_list, subgamma_list, '-', linewidth=4, marker='o', markersize=10, label=r"sub-gamma bound (Corollary 5)")

    plt.semilogx(nb_train_list, test_loss_list, 'k--', linewidth=4, marker='s', markersize=10, label=r'$\mathbf{E}_{\theta\sim\hat\rho^*} \mathcal{L}_{\mathcal D}^{\,\ell_{\rm nll}}(\theta)$ (test loss)')
    plt.semilogx(nb_train_list, train_loss_list, 'g-', linewidth=4, marker='s', markersize=10, label=r'$\mathbf{E}_{\theta\sim\hat\rho^*} \widehat\mathcal{L}_{X,Y}^{\,\ell_{\rm nll}}(\theta)$ (train loss)')

    plt.xlabel('$n$')
    plt.legend()
    plt.show()


def create_linear_data(n=100, d=10, sigma_x=1., w=None, w_norm=1., sigma_noise=0.1):
    if w is None:
        w = np.random.rand(d)
        w /= np.linalg.norm(w) / w_norm

    x = np.random.multivariate_normal(np.zeros(d), sigma_x ** 2 * np.eye(d), n) # Sample gaussian data
    noise = np.random.normal(scale=sigma_noise, size=n)
    y = np.dot(x, w) + noise
    return x, y, w


def create_linear_data_into_ball(n=100, d=10, w=None, w_norm=1, sigma_noise=.1, radius=1):
    if w is None:
        w = np.random.rand(d)
        w /= (np.linalg.norm(w) / w_norm)

    nb_outside = n
    norms = 2 * radius * np.ones(nb_outside)
    x = np.ones((n, d))

    while nb_outside > 0:
        x[norms > radius, :] = np.random.randn(nb_outside, d)  # Sample normal data
        norms = np.linalg.norm(x, axis=1)
        nb_outside = np.sum(norms > radius) # reject data outside the ball

    noise = np.random.normal(scale=sigma_noise, size=n)
    y = np.dot(x, w) + noise
    return x, y, w


def bound_alquier_sqrt(loss, kl, n, a, b, delta=0.05):
    return loss + (1./sqrt(n)) * (kl - log(delta) + 0.5 * (b - a) ** 2)


def bound_alquier_n(loss, kl, n, a, b, delta=0.05):
    return loss + (1./n) * (kl - log(delta)) + 0.5 * (b - a) ** 2


def bound_catoni(loss, kl, n, a, b, delta=0.05):
    x = loss - a + (kl - log(delta)) / n
    z = (b - a) / (1-exp(a-b))
    return a + z * (1 - exp(-x))


def bound_subgamma(loss, kl, n, s, c, delta=0.05):
    return loss + (1./n) * (kl - log(delta)) + s ** 2 / (2*(1-c))


if __name__ == '__main__':
    main()