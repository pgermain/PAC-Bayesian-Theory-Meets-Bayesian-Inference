import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import pi, sqrt

from matplotlib import rc
rc('text', usetex=True)

from bayesian_regression import BayesianRegression


def main():
    np.random.seed(42)

    nb_train = 15
    nb_test = 1000
    noise = .25
    sigma_prior = sqrt(1 / .005)

    x, y = create_sin_data(nb_train, noise, 2*pi)
    x_test, y_test = create_sin_data(nb_test, noise, 2*pi)
    x_lin = np.linspace(0.05, 2 * pi - 0.05, 200)
    y_sin = np.sin(x_lin)

    gs = gridspec.GridSpec(1, 2)
    ax = [plt.subplot(e) for e in gs]

    degree_list = list(range(1,8))
    marginal_list = np.zeros(len(degree_list))

    gibbs_train_loss_list = np.zeros(len(degree_list))
    gibbs_test_loss_list = np.zeros(len(degree_list))
    kl_list = np.zeros(len(degree_list))
    for i, d in enumerate(degree_list):
        polymap = PolynomialMapping(d, True)
        clf = BayesianRegression(sigma_prior, noise, pre_processor=polymap)
        clf.fit(x, y)
        marginal_list[i] = -clf.log_marginal_likelihood

        gibbs_train_loss_list[i] = clf.calc_gibbs_nll_loss(x, y)
        gibbs_test_loss_list[i] = clf.calc_gibbs_nll_loss(x_test, y_test)

        kl_list[i] = clf.calc_kullback_leibler()

        y_predictions = clf.predict(x_lin.reshape(-1, 1))
        ax[0].plot(x_lin, y_predictions, linewidth=4 if d == 3 else 2, label=r'model $d{=}%d$' % d)

    ax[0].plot(x_lin, y_sin, 'k--', label=r'$\sin(x)$', linewidth=3)

    ax[0].set_xticks([0, pi / 2, pi, 3 * pi / 2,  2 * pi])
    ax[0].set_xticklabels([r'$0$', r'$\frac12\pi$', r'$\pi$', r'$\frac32\pi$', r'$2\pi$'])
    ax[0].plot(x, y, 'o', markersize=10, markerfacecolor='k')
    ax[0].legend(ncol=2, columnspacing=1)
    ax[0].set_xlabel('$x$')
    ax[0].set_xbound(0, 7.)

    ax[1].plot(degree_list, marginal_list, 'r-', linewidth=2, marker='s', markersize=5, label=r'$-\ln Z_{X,Y}$')
    ax[1].plot(degree_list, kl_list, 'b-', linewidth=2, marker='s', markersize=5, label=r'$\mathrm{KL}(\hat\rho^*\|\pi)$')
    ax[1].plot(degree_list, nb_train*gibbs_train_loss_list, 'g-', linewidth=2, marker='s', markersize=5, label=r'$n\,\mathbf{E}_{\theta\sim\hat\rho^*} \widehat\mathcal{L}_{X,Y}^{\,\ell_{\rm nll}}(\theta)$')
    ax[1].plot(degree_list, nb_train*gibbs_test_loss_list, 'k--', linewidth=2, marker='s', markersize=5, label=r'$n\,\mathbf{E}_{\theta\sim\hat\rho^*} \mathcal{L}_{\mathcal D}^{\,\ell_{\rm nll}}(\theta)$')

    ax[1].legend()
    ax[1].set_xlabel('model degree $d$')

    plt.show()


def create_sin_data(n=100, sigma_noise=0.1, max_x=None):
    if max_x is None:
        max_x = 2 * pi

    x = np.random.rand(n, 1) * max_x  # Create uniform data
    noise = np.random.normal(scale=sigma_noise, size=n) # Create normal noise
    y = np.sin(x.reshape(-1)) + noise
    return x, y


class PolynomialMapping:
    def __init__(self, degree, add_ones=False):
        # degree of the polynomial feature mapper
        # if add_ones is True, add a constant feature "1" to every examples
        self.degree = degree
        self.add_ones = add_ones

    def __call__(self, x):
        n, d = np.shape(x)
        assert d == 1

        new_d = self.degree + (1 if self.add_ones else 0)
        new_x = np.ones((n, new_d))

        new_x[:, 0] = x[:, 0]
        for i in range(2, self.degree+1):
            new_x[:, i-1] = x[:, 0] ** i

        return new_x

if __name__ == '__main__':
    main()