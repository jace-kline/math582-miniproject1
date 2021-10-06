# Mini-project 1: Principal Component Analysis (PCA)
#### Author: Jace Kline

## Algorithm Development Steps

In this section, we walk through the steps behind implementing the PCA algorithm outlined in section 10.6 of the textbook. The algorithm will consist of four main steps:

1. Centering the original data set samples around the origin
2. Standardizing the data set based on standard deviation
3. Computing the eigendecomposition of the covariance matrix from the centered, standardized data set
4. Choosing the dimension for approximating the original space


#### Assumptions

* The data matrix X is structured such that rows are attributes and columns are samples
* The number of rows in data matrix X is less than the number of columns


```python
# import libraries

import numpy as np
import pandas as pd
import sklearn.decomposition as skl

np.set_printoptions(suppress=True, precision=6)
```

We will use the following data matrix below throughout the development of the algorithm to demonstrate intermediate steps.


```python
# sample data matrix for testing/demonstration

A = np.array([
    [2,8,2,1,5],
    [8,7,2,2,6],
    [4,0,5,0,4]
])

print("A:\n", A)
print("A dimensions:", A.shape)
```

    A:
     [[2 8 2 1 5]
     [8 7 2 2 6]
     [4 0 5 0 4]]
    A dimensions: (3, 5)


### Step 1: Centering the Data Matrix

Per the algorithm for PCA in section 10.6 of the book, we must first center the data matrix so that each dimension has a mean of 0. We achieve this by computing the mean of each dimension (row) and then subtracting all elements in each row by the respective row's mean value.


```python
D, samples = A.shape
rowmeans = np.mean(A, axis=1)
offsetmatrix = np.repeat(rowmeans, samples, axis=0).reshape((D,samples))
centered = A - offsetmatrix

print("Centered sample data set:\n", centered)
```

    Centered sample data set:
     [[-1.6  4.4 -1.6 -2.6  1.4]
     [ 3.   2.  -3.  -3.   1. ]
     [ 1.4 -2.6  2.4 -2.6  1.4]]


### Step 2: Standardization

The next step in the PCA algorithm involves standardizing each component of the data matrix by dividing by the component's respective standard deviation. We show this behavior below.


```python
rowstds = np.std(centered, axis=1)
standardized = (centered.T / rowstds).T

print("Centered and standardized sample data set:\n", standardized)
```

    Centered and standardized sample data set:
     [[-0.62092   1.707531 -0.62092  -1.008996  0.543305]
     [ 1.185854  0.790569 -1.185854 -1.185854  0.395285]
     [ 0.649934 -1.20702   1.114172 -1.20702   0.649934]]


### Step 3: Eigendecomposition of the Covariance Matrix

We must first find the covariance matrix of the centered and standardized data array. We then compute the eigendecomposition of this covariance matrix. Following this, we can choose a desired number of dimensions to use for dimensionality reduction.


```python
covmatrix = np.cov(standardized)
print("Covariance matrix:\n", covmatrix)
```

    Covariance matrix:
     [[ 1.25      0.690301 -0.396351]
     [ 0.690301  1.25      0.045877]
     [-0.396351  0.045877  1.25    ]]



```python
# Compute the eigenvalues and unit-length eigenvectors of the standardized covariance matrix
# The eigenvalues and vectors are ordered in ascending order

res = np.linalg.eigh(covmatrix)

# flip the order so the eigenvalues and vectors are sorted in descending order based on eigenvalue
eigvals = np.flip(res[0])
eigvecs = np.flip(res[1], axis=1)

print("Eigenvalues:\n", eigvals)
print("Eigenvectors:\n", eigvecs)
```

    Eigenvalues:
     [2.026786 1.289587 0.433627]
    Eigenvectors:
     [[ 0.715518 -0.029187 -0.697984]
     [ 0.616443  0.496459  0.611168]
     [-0.328682  0.867569 -0.373218]]


We show below how we execute dimensionality reduction. We simply choose the same number of eigenvectors (descending order by weight) as the number dimensions we desire, and we represent these in a matrix as column vectors. We call this matrix B.


```python
DESIRED_DIMENSIONS = 2

B = eigvecs[:, 0:DESIRED_DIMENSIONS]
print(B)
```

    [[ 0.715518 -0.029187]
     [ 0.616443  0.496459]
     [-0.328682  0.867569]]


### Step 4: Projection

Using our dimension-reducing matrix, we can take vectors from the original space and project them onto the principal subspace with fewer dimensions than the original space. To express this projection in the original space, we multiply by the original standard deviation and add the mean of each for each vector component.


```python
# x = a vector from the original space

x = np.array([1,1,1])


# x_standardized = x tranformed into centered, standardized space

x_standardized = (x - rowmeans) / rowstds


# x_principal = the vector obtained by transforming x_standardized into the principal subspace

x_principal = B @ (B.T @ x_standardized)


# x_approx = x_principal transformed back into the original space ~> An approximation of x after dimension reduction

x_approx = (x_principal * rowstds) + rowmeans

print("x =", x)
print("x_standardized =", x_standardized)
print("x_principal =", x_principal)
print("x_approx =", x_approx)
```

    x = [1 1 1]
    x_standardized = [-1.008996 -1.581139 -0.742781]
    x_principal = [-0.998428 -1.590392 -0.737131]
    x_approx = [1.027231 0.976591 1.012172]


We see that in this example the approximation is quite close to the original sample after dimension reduction. 

## Algorithm Implementation

Now that we have illustrated the steps for the algorithm, we show the implementation of this procedure as a Python class.


```python
# Python class to express the behavior of PCA analysis
class PCA:
    # initialize the PCA class with a given data set X (required)
    # optionally supply N, the number of reduction dimensions
    def __init__(self, X, M=1):
        # set the class variables
        self.X = X
        self.M = M
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
        self.B = self.eigvecs[:, 0:self.M]

    # set N, the number of dimensions to reduce to
    def set_M(self, M):
        # set N, recompute B
        self.M = M
        self.B = self.eigvecs[:, 0:self.M]

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
    def pca_reduce_dataset(self, M=-1):
        if M > 0:
            self.set_M(M)
        return self.unstandardize(self.B @ self.B.T @ self.standardized)
```

### Implementation Testing

Now that we have an implementation, we can test this implementation against the SciKit Learn implementation of PCA to ensure we get the same result on a test sample. The difference between our implementation and the SciKit Learn implementation is that our implementation automatically centers and standardizes the dataset prior to calculating the covariance matrix and eigendecompositions. Hence, we must have SciKit Learn's implementation perform PCA on the standardized data set, and we also must perform standardization on the sample point and undo this standardization after performing transformation. Below, we show the results of dimension reduction against sample x=(1,1,1) defined above. We see that both implementations compute the same covariance matrix and transformed sample.


```python
# Our implementation of PCA dimension reduction

pca = PCA(A, M=DESIRED_DIMENSIONS)
print("Covariance matrix:\n", pca.get_covariance_matrix())
print("Dimension-reduced vector x:\n", pca.transform(x))
print("Weights of eigenvalues (explained variance ratio):\n", pca.eigvals / np.sum(pca.eigvals))
print("Cumulative weights of eigenvalues (cumulative explained variance ratio):\n", np.cumsum(pca.eigvals / np.sum(pca.eigvals)))
```

    Covariance matrix:
     [[ 1.25      0.690301 -0.396351]
     [ 0.690301  1.25      0.045877]
     [-0.396351  0.045877  1.25    ]]
    Dimension-reduced vector x:
     [1.027231 0.976591 1.012172]
    Weights of eigenvalues (explained variance ratio):
     [0.540476 0.34389  0.115634]
    Cumulative weights of eigenvalues (cumulative explained variance ratio):
     [0.540476 0.884366 1.      ]



```python
# SciKit Learn implementation of PCA dimension reduction
# This implementation does not automatically center and standardize the data set
# Hence, we must perform the PCA on the centered, standardized dataset

pca_skl = skl.PCA(n_components=DESIRED_DIMENSIONS, svd_solver='full')
pca_skl.fit(standardized.T)
print("Covariance matrix:\n", pca_skl.get_covariance())
print("Dimension-reduced vector x:\n", pca.unstandardize(pca_skl.inverse_transform(pca_skl.transform(pca.standardize(x).reshape(1,-1))).reshape(3,)))

pca_skl.set_params(n_components=D)
print("Weights of eigenvalues (explained variance ratio):\n", pca_skl.explained_variance_ratio_)
print("Cumulative weights of eigenvalues (cumulative explained variance ratio):\n", np.cumsum(pca_skl.explained_variance_ratio_))
```

    Covariance matrix:
     [[ 1.25      0.690301 -0.396351]
     [ 0.690301  1.25      0.045877]
     [-0.396351  0.045877  1.25    ]]
    Dimension-reduced vector x:
     [1.027231 0.976591 1.012172]
    Weights of eigenvalues (explained variance ratio):
     [0.540476 0.34389 ]
    Cumulative weights of eigenvalues (cumulative explained variance ratio):
     [0.540476 0.884366]


## Analysis of Triathlon Data Set

We have obtained a data set from https://www.kaggle.com/mpwolke/wired-differently-triathlon/data. This data represents results from an Ironman 70.3 triathlon race that took place in 2019. We want to perform PCA analysis on the data set and compare PCA dimension reduction with SVD dimension reduction. We also want to explore the correlations between each of the sub-events (swim, bike, run) as well as overall time among the race participants.

### Data Cleaning

Prior to performing analysis, we first clean the data set into something useable in PCA dimension reduction. We ensure that all features of the data set are numerical and we remove unnecessary features not applicable to our analysis. For the time features, we simply convert these times into a number of seconds. The gender feature is codified as female=0, male=1. The age group is sorted in ascending order and given an index starting with the 18-24 year-old group at value 0, followed by the 25-29 year-olds at value 1, and continuing in this pattern until concluding with the pro division at value 9. Below we show the original data set compared to the cleaned data set. The data cleaning process can be found [here](https://github.com/jace-kline/math582-miniproject1/blob/main/reports/clean.md).


```python
# show the original data set
print("Original data:\n", pd.read_csv("../data/original.csv").head())

# store the cleaned data set into a dataframe and print it
df = pd.read_csv("../data/cleaned.csv")
print("\nCleaned data:\n", df.head())
```

    Original data:
        Unnamed: 0  Pos                       Name                     Club  \
    0           1    1            FRODENO Jan (2)        Laz  Saarbruecken   
    1           2    2        CLAVEL Maurice (24)                      NaN   
    2           3    3  TAAGHOLT Miki  Morck (21)                    Ttsdu   
    3           4    4         STRATMANN Jan (13)  Triathlon  Team  Witten   
    4           5    5       MOLINARI Giulio (22)        C. S. Carabinieri   
    
            Cat        Swim          T1       Bike          T2         Run  \
    0  MPRO - 1   00:22:501  00:01:2710  02:02:011  00:01:5221   01:11:252   
    1  MPRO - 2  00:23:3311  00:01:3015  02:04:543   00:01:323   01:12:144   
    2  MPRO - 3   00:22:574   00:01:141  02:05:526  00:02:0148   01:14:238   
    3  MPRO - 4   00:22:502  00:01:3118  02:08:208  00:01:5633   01:13:336   
    4  MPRO - 5   00:22:585  00:01:4968  02:05:114  00:02:0768  01:17:2117   
    
           Time  
    0  03:39:35  
    1  03:43:43  
    2  03:46:27  
    3  03:48:10  
    4  03:49:26  
    
    Cleaned data:
        Swim   T1  Bike   T2   Run   Time  Gender  AgeGroup
    0  1370   87  7321  112  4285  13175       1         9
    1  1413   90  7494   92  4334  13423       1         9
    2  1377   74  7552  121  4463  13587       1         9
    3  1370   91  7700  116  4413  13690       1         9
    4  1378  109  7511  127  4641  13766       1         9


### PCA and SVD Dimension Reduction

We will start by performing PCA and SVD dimension reduction on each sample in our data set. We will vary the number of dimensions and observe the accuracy of each method. We use the SVD implementation provided by the SciKit Learn library, consistent with the SVD algorithm outlined in section 10.4 of the book.


```python
# store the data set as an array
# features are rows, samples are columns
X = df.to_numpy().T
X_features, X_samples = X.shape

print("Data matrix:\n", X)
print("\nDimensions:", X.shape)
```

    Data matrix:
     [[ 1370  1413  1377 ...  2481  2436  2230]
     [   87    90    74 ...   338   128   278]
     [ 7321  7494  7552 ...  9599  9676  9702]
     ...
     [13175 13423 13587 ... 19279 19286 19286]
     [    1     1     1 ...     1     1     1]
     [    9     9     9 ...     3     4     4]]
    
    Dimensions: (8, 797)



```python
# Create SVD decomposition of X
U, singular_values, Vt = np.linalg.svd(X, full_matrices=False, compute_uv=True)
Sigma = np.diag(singular_values)

# Create PCA decomposition of X
pca = PCA(X)

# define a function that computes the SVD approximation of X in N dimensions
def svd_reduce_dataset(M):
    return U[:,0:M] @ Sigma[0:M,0:M] @ Vt[0:M,:]

# define a function that computes the PCA approximation of X in N dimensions
def pca_reduce_dataset(M):
    return pca.pca_reduce_dataset(M=M)

# used to compute the "error" in our dimension reduction approximations
def frobenius_norm(A):
    return np.linalg.norm(A)
```


```python
# Loop over possible dimensions to reduce to and perform both SVD and PCA dimension reduction on X
# Compute the sum of squares value 
for M in range(1, X_features):
    svd_reduced = svd_reduce_dataset(M)
    pca_reduced = pca_reduce_dataset(M)
    
    svd_error = frobenius_norm(X - svd_reduced)
    pca_error = frobenius_norm(X - pca_reduced)

    print(f"M = {M}: SVD error = {svd_error}; PCA error = {pca_error}")
```

    M = 1: SVD error = 12845.868065052797; PCA error = 16344.79537113727
    M = 2: SVD error = 6808.998782248694; PCA error = 14819.669507863737
    M = 3: SVD error = 2124.5653701961114; PCA error = 14711.263178927595
    M = 4: SVD error = 1108.1890801904326; PCA error = 12247.783107777936
    M = 5: SVD error = 57.71175687035737; PCA error = 9847.462906328448
    M = 6: SVD error = 9.481217969047252; PCA error = 8948.267230251893
    M = 7: SVD error = 5.625634185768421; PCA error = 9.762794765606992


### Analysis of SVD vs PCA

With our given data set, it is evident by computing the frobenius norm of the differences between the approximation matrices and the original matrix results in more accurate modeling by the SVD dimension reduction method over PCA. This can be explained through observation of the singular values and eigenvalues of the SVD and PCA methods, respectively. We see below that the singular values in the SVD decomposition have much greater percent differential between subsequent values, implying that the first few dimensions carry more "weight" in the decomposition, and therefore will keep the dimension reduction fairly accurate for reduced values of M. On the contrary, the PCA eigenvalues show relatively similar magnitudes, leading to the conclusion that the PCA is not finding highly weighted principal components and therefore the dimension reduction approximation will suffer overall, particularly for lower values of M. We show this below.


```python
print("SVD singular values:\n", singular_values)
print("Weights of SVD singular values:\n", singular_values / np.sum(singular_values))
print("Cumulative weights of SVD singular values:\n", np.cumsum(singular_values / np.sum(singular_values)))

print("\nPCA eigenvalues:\n", pca.eigvals)
print("Weights of PCA eigenvalues (explained variance ratio):\n", pca.eigvals / np.sum(pca.eigvals))
print("Cumulative weights of PCA eigenvalues (cumulative explained variance ratio):\n", np.cumsum(pca.eigvals / np.sum(pca.eigvals)))
```

    SVD singular values:
     [588557.1587    10892.835348   6469.056067   1812.648608   1106.685317
         56.927615      7.631889      5.625634]
    Weights of SVD singular values:
     [0.966577 0.017889 0.010624 0.002977 0.001817 0.000093 0.000013 0.000009]
    Cumulative weights of SVD singular values:
     [0.966577 0.984466 0.99509  0.998067 0.999885 0.999978 0.999991 1.      ]
    
    PCA eigenvalues:
     [3.752128 1.127786 0.940594 0.731742 0.639256 0.479047 0.339497 0.      ]
    Weights of PCA eigenvalues (explained variance ratio):
     [0.468428 0.140796 0.117427 0.091353 0.079807 0.059806 0.042384 0.      ]
    Cumulative weights of PCA eigenvalues (cumulative explained variance ratio):
     [0.468428 0.609224 0.726651 0.818004 0.89781  0.957616 1.       1.      ]


### Error Tolerance of PCA Dimension Reduction

We can use the cumulative explained variance ratio to determine how much variance we capture of our data when performing PCA dimension reduction with different values of M (reduction dimensions). If we are satisfied with capturing greater than 80% of the variance from the original data set, then we can use M>=4 as a dimension reduction approximation. However, if we want to capture greater than 95% of the variance from the original data set, then we must use M>=6.

## Feature Correlation Analysis

Now that we have shown the comparision of SVD vs PCA for dimension reduction, we shift our focus to the correlations/covariances between the features in our data set. Particularly, we want to see which of the features are highest correlated with eachother and which are highest correlated with an athlete's overall finishing time.


```python
df.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Swim</th>
      <th>T1</th>
      <th>Bike</th>
      <th>T2</th>
      <th>Run</th>
      <th>Time</th>
      <th>Gender</th>
      <th>AgeGroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Swim</th>
      <td>1.000000</td>
      <td>0.477852</td>
      <td>0.540819</td>
      <td>0.313248</td>
      <td>0.424090</td>
      <td>0.691435</td>
      <td>0.091466</td>
      <td>-0.208273</td>
    </tr>
    <tr>
      <th>T1</th>
      <td>0.477852</td>
      <td>1.000000</td>
      <td>0.483319</td>
      <td>0.428836</td>
      <td>0.358845</td>
      <td>0.553294</td>
      <td>0.111193</td>
      <td>-0.161320</td>
    </tr>
    <tr>
      <th>Bike</th>
      <td>0.540819</td>
      <td>0.483319</td>
      <td>1.000000</td>
      <td>0.359894</td>
      <td>0.593732</td>
      <td>0.895452</td>
      <td>-0.133717</td>
      <td>-0.235186</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>0.313248</td>
      <td>0.428836</td>
      <td>0.359894</td>
      <td>1.000000</td>
      <td>0.329472</td>
      <td>0.446983</td>
      <td>0.108297</td>
      <td>-0.080900</td>
    </tr>
    <tr>
      <th>Run</th>
      <td>0.424090</td>
      <td>0.358845</td>
      <td>0.593732</td>
      <td>0.329472</td>
      <td>1.000000</td>
      <td>0.852911</td>
      <td>0.007329</td>
      <td>-0.203262</td>
    </tr>
    <tr>
      <th>Time</th>
      <td>0.691435</td>
      <td>0.553294</td>
      <td>0.895452</td>
      <td>0.446983</td>
      <td>0.852911</td>
      <td>1.000000</td>
      <td>-0.032630</td>
      <td>-0.257870</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>0.091466</td>
      <td>0.111193</td>
      <td>-0.133717</td>
      <td>0.108297</td>
      <td>0.007329</td>
      <td>-0.032630</td>
      <td>1.000000</td>
      <td>-0.022239</td>
    </tr>
    <tr>
      <th>AgeGroup</th>
      <td>-0.208273</td>
      <td>-0.161320</td>
      <td>-0.235186</td>
      <td>-0.080900</td>
      <td>-0.203262</td>
      <td>-0.257870</td>
      <td>-0.022239</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



From the above correlation chart, we observe that the bike performance is highest correlated with an athlete's finishing time with correlation of 0.895. The run is second highest correlated with overall time at a value of 0.853. The swim portion has the lowest correlation value of the three sub-disciplines when compared to the overall time, coming in at 0.691. In addition, the transition times (T1, T2) are also intuitively positively correlated with overall finishing time, but the magnitude of these correlations are noticeably smaller. Interestingly, we also see that among the sub-disciplines, the bike and the run performances are the highest correlated at 0.594 with the swim-bike correlation following at 0.541 and the swim-run lowest correlated at 0.424. Another interesting finding from the correlations shows that age is much more linked to overall time than gender is.

The key takeaway from this correlation analysis is that the bike portion of this distance of triathlon race is the most important when it comes to minimizing overall time, followed next by running time and then by swimming time. If one were to devise a training plan for an aspiring triathlete, this analysis shows that the most benefit to overall race performance should come from training for the bike, run, and swim in that order. A caveat to this is that this race distance (Ironman 70.3) has different race "proportions" for each of the sub-disciplines than other triathlon races. For instance, an Olympic distance triathlon has a higher percentage of the race distance allocated to the swim, which ultimately makes the swim more important in that race distance.

## Conclusion

This report has been an attempt to detail the development and evaluation of the PCA algorithm outlined in section 10.6 of the textbook. The first part of this report walked through the thought process and reasoning behind constructing the PCA algorithm. The next part involved the implmentation of the PCA algorithm as a Python class. We tested our PCA implementation against the SciKit Learn PCA implementation and found the same results on a test data set and sample. Following this, we loaded our target data set containing a set of triathlete finishing times and sub-discipline times for an Ironman 70.3 triathlon race. We proceeded to perform SVD and PCA analysis on this data set, and we found that the SVD dimension reduction performed better than PCA on this particular data set. We explained that this was due to the variance captured in the eigenvalues of the covariance matrix being "spread out" instead of concentrated in a few eigenvalues. This fact led to relatively low cumulative explained variance ratio for lower values of M (reduction dimensions) and therefore a requirement to use higher M values to capture respectable variance from the original data set when performing dimension reduction. We finished the discussion by analyzing the correlation matrix of the target data set and highlighting relationships present between features, particularly relationships between swim, bike, run, and overall performance.
