import numpy as np
from math import log, pi


class BayesianRegression:
    def __init__(self, sigma_prior=1., sigma_post=1., pre_processor=None):
        # Initialize the model constant parameters (std of the prior and posterior)
        # pre_processor is a function that map the feature vectors in another reprensentation
        self.sqr_sigma_prior = sigma_prior ** 2
        self.sqr_sigma_post = sigma_post ** 2
        self.pre_processor = pre_processor
        self.w = None

    def fit(self, x, y):
        # Learn the model
        if self.pre_processor is not None:
            x = self.pre_processor(x)

        n, d = np.shape(x)
        self.A = np.dot(x.T, x) / self.sqr_sigma_post + np.eye(d) / self.sqr_sigma_prior
        self.Ainv = np.linalg.pinv(self.A)
        self.w = np.dot(np.dot(self.Ainv, x.T), y.reshape(-1, 1)) / self.sqr_sigma_post
        self.w = self.w.reshape(-1)

        E = 0.5 * (np.sum((y - np.dot(x, self.w)) ** 2) / self.sqr_sigma_post)
        E += 0.5 * (np.dot(self.w, self.w) / self.sqr_sigma_prior)

        sgndetA, logdetA = np.linalg.slogdet(self.A)
        assert sgndetA == 1.0

        self.log_marginal_likelihood = - E - 0.5 * (logdetA + n * log(2 * pi * self.sqr_sigma_post) + d * log(self.sqr_sigma_prior))

        return self.w

    def predict(self, x, pre_processed=False):
        # Predict the label of an examples matrix
        if self.pre_processor is not None and not pre_processed:
            x = self.pre_processor(x)

        return np.dot(x, self.w)

    def calc_bayes_nll_loss(self, x, y, pre_processed=False, return_loss_array=False):
        # negative log likelihood loss of w
        predictions = self.predict(x, pre_processed)
        loss_array = 0.5 * (((predictions - y) ** 2) / self.sqr_sigma_post + log(2 * pi * self.sqr_sigma_post))
        loss_avg = np.mean(loss_array)

        return loss_avg if not return_loss_array else (loss_avg, loss_array)

    def calc_gibbs_nll_loss(self, x, y):
        # negative log likelihood loss of the Gibbs distribution N(w, inv(A))
        if self.pre_processor is not None:
            x = self.pre_processor(x)

        tr = np.trace(np.dot(np.dot(x.T, x), self.Ainv))
        bayes_loss, loss_array = self.calc_bayes_nll_loss(x, y, pre_processed=True, return_loss_array=True)
        return bayes_loss + tr / (2 * len(y) * self.sqr_sigma_post)

    def calc_kullback_leibler(self):
        # explicit calculation of KL divergence between prior N(0,sqr_sigma_prior) and posterior N(w, inv(A))
        # see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence

        tr = np.trace(self.Ainv) / self.sqr_sigma_prior
        l2 = np.dot(self.w, self.w) / self.sqr_sigma_prior
        d, d_ = self.A.shape
        assert d == d_

        logdet_prior = d * log(self.sqr_sigma_prior)
        sgndetA, logdet_post = np.linalg.slogdet(self.A)
        assert sgndetA == 1.0

        kl = (tr + l2 - d + logdet_post + logdet_prior) / 2.
        return kl


