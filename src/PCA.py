# Implementation of PCA algorithm via a Python class

import numpy as np
import pandas as pd


class PCA:
    def __init__(self, X, dimensions):
        # center the data matrix
        D, samples = X.shape
        self.rowmeans = np.mean(X, axis=1)
        centered = X - np.repeat(self.rowmeans, samples, axis=0).reshape((D,samples))

        # standardize the centered data matrix
        self.rowstds = np.std(centered, axis=1)
        standardized = (centered.T / self.rowstds).T

        # compute the covariance matrix
        self.covmatrix = np.cov(standardized)

        # compute the eigendecomposition of the covariance matrix
        res = np.linalg.eigh(self.covmatrix)
        eigvals = np.flip(res[0])
        eigvecs = np.flip(res[1], axis=1)

        # compute B
        self.B = eigvecs[:, 0:dimensions]

    def reducer(self, x):
        x_standardized = (x - self.rowmeans) / self.rowstds
        x_principal = self.B @ (self.B.T @ x_standardized)
        x_reduced = (x_principal * self.rowstds) + self.rowmeans
        return x_reduced

    def covariance_matrix(self):
        return self.covmatrix



