'''pca_cov.py
Performs principal component analysis using the singular value decomposition of the dataset
Ninh Giang Nguyen
CS 252: Mathematical Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import pandas as pd
import scipy.linalg
import pca
import scipy

from data_transformations import normalize, center


class PCA_SVD(pca.PCA):
    '''Perform principal component analysis using the singular value decomposition (SVD)

    NOTE: In your implementations, only the following "high level" `scipy`/`numpy` functions can be used:
    - `np.linalg.svd`
    The numpy functions that you have been using so far are fine to use.
    '''
    
    def fit(self, vars, normalize_dataset=False):
        '''Performs PCA on the data variables `vars` using SVD instead of the covariance matrix

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE:
        - This method should closely mirror the structure of your implementation in the `PCA` class, except there
        should NOT be a covariance matrix computed here!
        - Make sure you compute all the same instance variables that `fit` does in `PCA`.
        - Leverage the methods that you already implemented as much as possible to do computations.
        '''
        self.vars = vars
        self.A = self.data[vars].values
        self.A = np.array(self.A)
        self.orig_means = np.mean(self.A, axis = 0)
        
        if normalize_dataset:
            self.normalized = True
            self.orig_maxs = np.max(self.A, axis = 0)
            self.orig_mins = np.min(self.A, axis = 0)
            self.A = normalize(self.A)
        else:
            self.normalized = False
        
        self.A = center(self.A)
        U, S, VT = scipy.linalg.svd(self.A)
     
        self.e_vals = S**2/(self.A.shape[0] - 1)
        self.e_vecs = VT.T

   
        indices = np.argsort(self.e_vals)[::-1]
        self.e_vals = self.e_vals[indices]
        self.e_vecs = self.e_vecs[:, indices]

        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

