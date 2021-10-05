import numpy as np
import pandas as pd

# Python class to express the behavior of PCA analysis
class PCA:
    # initialize the PCA class with a given data set X (required)
    # optionally supply N, the number of reduction dimensions
    def __init__(self, X, N=1):
        # set the class variables
        self.X = X
        self.N = N

        # center the data matrix
        D, samples = X.shape
        self.rowmeans = np.mean(X, axis=1)
        self.centered = X - np.repeat(self.rowmeans, samples, axis=0).reshape((D,samples))

        # standardize the centered data matrix
        self.rowstds = np.std(self.centered, axis=1)
        self.standardized = (self.centered.T / self.rowstds).T

        # compute the covariance matrix
        self.covmatrix = np.cov(self.standardized)

        # compute the eigendecomposition of the covariance matrix
        res = np.linalg.eigh(self.covmatrix)
        self.eigvals = np.flip(res[0])
        self.eigvecs = np.flip(res[1], axis=1)

        # compute B
        self.B = self.eigvecs[:, 0:self.N]

    # set N, the number of dimensions to reduce to
    def set_N(self, N):
        # set N, recompute B
        self.N = N
        self.B = self.eigvecs[:, 0:self.N]

    # center and standardize variance to 1
    def standardize_sample(self, x):
        return (x - self.rowmeans) / self.rowstds

    # shift sample back to original data space
    def unstandardize_sample(self, x):
        return (x * self.rowstds) + self.rowmeans

    # return the covariance matrix of the centered, standardized data
    def get_covariance_matrix(self):
        return self.covmatrix

    # transform a standardized sample of D dimensions into N dimensions
    def transform_reduce(self, x):
        return self.B.T @ x

    # transform a dimension-reduced sample of N dimensions into D dimensions
    # the result is centered and standardized
    def transform_inverse(self, z):
        return self.B @ z

    # perform end-to-end transformation
    # centers, standardizes, reduces, inverts, and unstandardizes
    # this function takes a sample and "approximates" it using PCA with given N
    def transform(self, x):
        x_standardized = self.standardize_sample(x)
        x_principal = self.transform_inverse(self.transform_reduce(x_standardized))
        x_transformed = self.unstandardize_sample(x_principal)
        return x_transformed



