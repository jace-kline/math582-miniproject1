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
        D, samples = self.X.shape
        self.rowmeans = np.mean(X, axis=1)
        self.rowstds = np.std(self.X, axis=1)

        # standardize the centered data matrix
        self.standardized = self.standardize(X)

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

    # center and standardize variance to 1 of sample(s)
    def standardize(self, x):
        if len(x.shape) == 1: # 1d array
            return (x - self.rowmeans) / self.rowstds
        else:
            D, samples = x.shape
            centered = x - np.repeat(self.rowmeans, samples, axis=0).reshape((D,samples))
            standardized = (centered.T / self.rowstds).T
            return standardized

    # shift sample(s) back to original data space
    def unstandardize(self, x):
        if len(x.shape) == 1:
            return (x * self.rowstds) + self.rowmeans
        else:
            D, samples = x.shape
            return (x.T * self.rowstds).T + np.repeat(self.rowmeans, samples, axis=0).reshape((D,samples))

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
        x_standardized = self.standardize(x)
        x_principal = self.transform_inverse(self.transform_reduce(x_standardized))
        x_transformed = self.unstandardize(x_principal)
        return x_transformed

    # perform dimension reduction on the entire X data set
    # then map back to original X space
    def pca_reduce_dataset(self, N=-1):
        if N > 0:
            self.set_N(N)
        return self.unstandardize(self.B @ self.B.T @ self.standardized)



