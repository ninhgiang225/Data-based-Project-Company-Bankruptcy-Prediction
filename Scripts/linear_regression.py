'''linear_regression.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS 252: Mathematical Data Analysis Visualization
Spring 2024
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis
import data


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable (true values) being predicted by linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A  = self.data.select_data(ind_vars)
        n = self.A.shape[0]

        self.y = self.data.select_data([dep_var])
        A1 = np.hstack([np.ones([n, 1]), self.A ])

        if method ==  "scipy":
            c = self.linear_regression_scipy(self.A, self.y)
        elif method ==  "normal":
            c = self.linear_regression_normal(A1, self.y)
        elif method ==  "qr":
            c = self.linear_regression_qr(A1, self.y)
        elif method ==  "gradient":
            c = self.linear_regression_gradient(self.A, self.y)
        self.slope = c[1:]
        self.intercept =c[0,0]
   
 
        y_pred = A1@c
        self.residuals = self.compute_residuals(y_pred)
        self.mse = self.compute_mse()
        self.R2 = self.r_squared(A1@c)
        self.p = p
    
        if p > 1:
            self.p = p
            self.A = self.make_polynomial_matrix(self.A, p)


    ### EXTENSION ###
    def linear_regression_gradient(self, X, y):
        # Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        num_samples, num_features = X.shape

        # Initialize coefficients
        c = np.random.randn(num_features)

        # Gradient descent
        for _ in range(1000):
            # Calculate predictions
            predictions = np.dot(X, c)

            # Calculate gradient
            error = predictions - y
            gradient = 1/num_samples * np.dot(X.T, error)

            # Update coefficients
            c -=  gradient
        return c

        
    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        A_copy = np.hstack([np.ones((A.shape[0], 1)), A])

        c,_,_,_ = scipy.linalg.lstsq(A_copy, y)
    
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''

        a = A.T @ A
        b = A.T @ y
        return np.linalg.inv(a) @ b

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        # Idea: Rc =Q.T @ y
        Q, R = self.qr_decomposition(A)
        c = scipy.linalg.solve_triangular(R, Q.T @ y) 
        return c


    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        Q = np.zeros((A.shape[0], A.shape[1]))
  
        # i column of A is the one that we want to convert it to orthogonal vector
        # j column of Q is the vector that are already mutual othgonal and we want to use them to convert i of A 
        for i in range (A.shape[1]):
            a = A[:,i].copy() #shape = (n,1)
            for j in range(i):
                a -= np.dot(a, Q[:,j]) * Q[:,j]
            Q[:,i] =  a / np.linalg.norm(a)
        

        R = Q.T @ A
        return Q, R

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        # NOTE: Need to make sure if passing in X:
        # - not polynomialized yet
        # - it NEEDS to NOT have intercept already
        if X is None:
            X = self.A

        if self.p > 1:
            X = self.make_polynomial_matrix(X, self.p)
            # N = len(X)
            # X = np.hstack([np.ones([N, 1]), A])

        if X.ndim == 1:
            X = X[:, np.newaxis]
        
        print(X.shape)
        print(self.slope.shape)
        print(self.intercept)
        return  X @ self.slope + self.intercept

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''

        sse = np.sum((self.compute_residuals(y_pred))**2)
        smd = np.sum((self.y - np.mean(self.y))**2)
        return 1 - sse/smd

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
 
        return self.y - y_pred

    def compute_mse(self):      
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        y_pred = self.predict()
        return np.sum(self.compute_residuals(y_pred)**2) / y_pred.shape[0]

    def scatter(self, ind_var, dep_var, title=""):         
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        # scatter the x and y real values
        N_sample_pts = 100
        x, y = super().scatter(ind_var, dep_var, 30, title)
        x_sample = np.linspace(np.min(x), np.max(x), N_sample_pts)

        # plot the regression line
        if self.p == 1:
            #y_pred = self.predict(x_sample[:, np.newaxis])
            y_pred = self.intercept + self.slope[0]*x_sample
            y_pred = y_pred[:,np.newaxis]
      
            plt.plot(x_sample, y_pred, color = "r")
            plt.title(title + " with R2 = " + str(self.R2) + "& mse = "+str(self.mse), size=20)

        if self.p > 1:
            # REMINDER: THIS N is num sample points, not num samples in dataset
            # A1 = np.hstack([np.ones([N_sample_pts, 1]), x_sample])
            # X_sample_poly = self.make_polynomial_matrix(x_sample[:, np.newaxis], self.p)
            y_pred_poly = self.predict(x_sample[:, np.newaxis])
            plt.plot(x_sample, y_pred_poly, color="r")


    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz)
        n= len(data_vars)
        for i, var_i in enumerate(data_vars):
            for j, var_j in enumerate(data_vars):
                if i != j:
    
                    self.linear_regression([var_i], var_j)
                    x_sample = np.linspace(np.min(self.A), np.max(self.A), self.A.shape[0])
                    y_pred = self.predict(x_sample)
                    axes[i][j].plot(x_sample, y_pred, color = "r")
                    axes[i][j].set_title(f"R2 = {self.R2:.4f}")

        
                if hists_on_diag and i==j:
                    
                    axes[i, j].remove()
                    axes[i, j] = fig.add_subplot(n, n, i*n+j+1)
                    if j < n-1:
                        axes[i, j].set_xticks([])
                    else:
                        axes[i, j].set_xlabel(data_vars[i])
                    if i > 0:
                        axes[i, j].set_yticks([])
                    else:
                        axes[i, j].set_ylabel(data_vars[i])
                    data = self.data.select_data([var_i])
                    plt.hist(data, bins = 30)

        self.show()


    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        self.p = p
        m = np.zeros([A.shape[0], p])
 
        for i in range(1, p+1):
            m[:, i-1] = np.squeeze(A**(i))

        return m        



# Quick look through this, idea to implement pol-matrix
    def poly_regression(self, ind_var, dep_var, p, color ="k"):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.p = p
        self.ind_vars = ind_var
        self.dep_var = dep_var
        
        self.A  = self.data.select_data([ind_var])
        n = self.A.shape[0]

        if p > 1:
            A1 = self.make_polynomial_matrix(self.A, p)
        else:
            A1 = self.A
            
        self.y = self.data.select_data([dep_var])
        A1 = np.hstack([np.ones([n, 1]), A1 ])


        c = self.linear_regression_qr(A1, self.y)

        self.slope = c[1:]
        self.intercept =c[0,0]
   
 
        y_pred = A1@c
        self.residuals = self.compute_residuals(y_pred)
        # self.mse = self.compute_mse()
        self.mse = -1
        self.R2 = self.r_squared(y_pred)
        


        # scatter the x and y real values
        # x, y = super().scatter(ind_var, dep_var, 30)
        # x_sample = np.linspace(np.min(x), np.max(x), self.y.shape[0])
        # self.p = p
        # # plot the regression line
        # if self.p == 1:
        #     y_pred = self.intercept + self.slope[0]*x_sample
        #     y_pred = y_pred[:,np.newaxis]
        #     plt.plot(x_sample, y_pred, color = "r")
        #     plt.title("Linear regression" + " with R2 = " + str(self.R2)+ "& mse = "+str(self.mse), size = 20, color = color)
        #     plt.plot(x_sample, y_pred, color = "r")

        
        # if self.p > 1:
        #     X_sample_poly = self.make_polynomial_matrix(x_sample[:,np.newaxis], self.p)
        #     X_sample_poly = np.hstack([np.ones((X_sample_poly.shape[0], 1)), X_sample_poly])
        #     c = self.linear_regression_qr(X_sample_poly, self.y)

        #     # Generate polynomial predictions
        #     y_pred_poly = np.dot(X_sample_poly, c)
        #     self.mse = np.mean(np.sum((self.y-y_pred_poly)**2))
        #     self.R2 = self.r_squared(y_pred_poly)
        #     plt.title(str(p)+"-Polynominal regression" + " with R2 = " + str(self.r_squared(self.R2)) + "& mse = "+str(self.mse) , size = 20,  color = color)
            
        #     # Plot the polynomial regression line
        #     plt.plot(x_sample, y_pred_poly, color="r")
        #     self.slope = c[1:]
        #     self.intercept = c[0,0]
        
            

        


    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept



# What should I do for this
    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        self.intercept = intercept
        self.slope = slope
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.p = p

