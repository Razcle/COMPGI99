import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



# ----------------- Generate Data ------------------------ #

def gen_data_non_linear(N,D,sigma):
    """
		Inputs
			N - integer number of data points
			D - number of dimensions
			sigma - noise
		Return
			X,Y,Y_true"""
    X = np.random.rand(N,D)*10 -5
    X = np.sort(X,0)
    X = np.concatenate((X,X**2),axis = 1)
    W_true = np.random.randn(2*D,1)
    Y_true = np.dot(X,W_true)
    Y = Y_true + np.random.randn(N,1)*sigma
    return X,Y,Y_true

def gen_data_linear(N,D,sigma):
    """Generate Linear Data with Gaussian Noise
        N - Number of Data
        D - Dimension
        simga - noise level """
    X = np.random.randn(N,D)
    w_true = np.random.randn(D+1,1)
    # Sort the X's
    X = np.sort(X,axis = 0)
    # Add Biases
    X = np.concatenate((X, np.ones((N, 1))), axis=1)
    # Get Y
    y = np.dot(X,w_true) + sigma*np.random.randn(N,1)
    return w_true, X, y



# ------------------- Kernels ---------------------------- #


def gaussian_kernel(x,y,sigmaSq = 100.0):
    return np.exp(-(np.linalg.norm(x-y)**2)/float(sigmaSq))

def polynomial_kernel(x,y,degree = 1):
    return (1+np.dot(x,y.T))**(degree)

def linear_kernel(x,y):
    return np.dot(x,y.T)

# ----------- REGRESSION FUNCTIONS ------------------------ #

def kernel_ridge_normal(X,Y,lmbda,kernel):
    """ Inputs:
        X - np array of data in shape num_points x num_dim
        Y - np array of labels
        lmda - regularisation constant
        kernel - kernel function
    return:
        K - Kernel Matrix as N x N np array
        alpha - dual regression vector
    """
    N = np.shape(X)[0]
    K = np.zeros((N,N))

    # Build K
    for i in range(N):
        for j in range(N):
            K[i,j] = kernel(X[i],X[j])

    #alpha = np.linalg.lstsq(K + lmbda * np.eye(N),Y)[0]

    kinv = np.linalg.inv(K + lmbda * np.eye(N))
    alpha = np.dot(kinv,Y)

    return K, alpha

def kernel_regression_with_sgd(X,Y,kernel,mu,lmbda,num_iter):
    """
    Perform num_iter iterations of sgd on this kernel regression task and
    return alpha as well as well as the loss at each stage of the descent.

    Inputs:
    	X
    	Y
    	Kernel
    	mu - learning rate
    	lmbda - regularisation parameter"""

    # Initialise alpha and errors
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    alpha = np.zeros((N,1))
    errors = np.zeros((num_iter,1))

    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = kernel(X[i],X[j])

    for step in range(num_iter):
        i = np.random.randint(N)
        x = X[i:i+1]
        y = Y[i]
        errors[step] = loss(alpha,K,Y)
        alpha = kernel_sgd_update_alpha(alpha,mu,y,i,K,lmbda)

    return K,alpha,errors


def kernel_regression_with_awsgd(X,Y,x_test,y_test,kernel,mu,eps,lmbda,num_iter,base_line = 0.0,verbose = False):
    """ Performs kernel regression with awsgd and returns the kernel matrix, alpha and
    errors as a function of iteration number.

    Inputs:
        X,Y - Data
        X_test,Y_test - Test Data
        kernel
        mu - model learning rate
        eps - sampling learning rate
        lmbda - regularisation paramter
        base_line
    """

    # Initialise alpha and errors
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    alpha = np.zeros((N,1))
    errors = np.zeros((num_iter,1))
    test_errors = np.zeros((N,1))
    taus = np.zeros((num_iter,1))
    tau = N/2
    sigma = N/5

    if x_test != []:
        X = np.concatenate((X,x_test),axis = 0)
        N_test = np.shape(x_test)[0]
    else:
        N_test = 0

    K = np.zeros((N+N_test,N+N_test))
    for i in range(N+N_test):
        for j in range(N+N_test):
            K[i,j] = kernel(X[i],X[j])

    K_train = K[0:N,0:N]
    K_test = K[N:,0:N]

    for step in range(num_iter):
        i,weight,z = get_weighted_index(N,sigma,tau)
        errors[step] = loss(alpha,K_train,Y)
        if x_test != []:
            test_errors[step] = loss(alpha,K_test,y_test)
        taus[step] = tau
        alpha,tau,base_line = kernel_awsgd_alpha_update(alpha,mu,eps,tau,Y,i,weight,z,base_line,K,lmbda)

    if verbose is False:
    	return K,alpha,errors
    else:
        return K,alpha,errors,taus,test_errors

def kernel_regression_with_minvar(X,Y,kernel,mu,lmbda,num_iter):
    """ Performs kernel regression with awsgd and returns the kernel matrix, alpha and
    errors as a function of iteration number.

    Inputs:
        X,Y - Data
        kernel
        mu - model learning rate
        eps - sampling learning rate
        lmbda - regularisation paramter
    """

    # Initialise alpha,sampling probs and errors
    N = np.shape(X)[0]
    D = np.shape(X)[1]

    alpha = np.zeros((N,1))
    errors = np.zeros((num_iter,1))
    probs = np.ones(N)*N**2
    dist = discrete_distribution(probs)


    # Initialise K
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = kernel(X[i],X[j])

    for step in range(num_iter):
        index = dist.sample()
        x = X[index:index+1]
        y = Y[index]
        errors[step] = loss(alpha,K,Y)
        alpha,delta = kernel_minvar_sgd_update_alpha(alpha,mu,y,index,K,lmbda)
        dist.update_prob(index,delta)


    return K,alpha,errors



# ----------- Utility Functions ---------------------------- #

def regressor(alpha,K,i):
    """
     Make a prediction for the y_value of x given the data seen so far
     Return
     	y_hat
    """
    y_hat = np.dot(K[i].T,alpha)

    return y_hat

def kernel_sgd_update_alpha(alpha,mu,y,i,K,lmbda):
    """Update the present value of alpha given what we've seen so far
    """
    alpha[i] = 0.0
    alpha = (1-mu*lmbda)*alpha
    delta = (regressor(alpha,K,i) - y)
    alpha[i] = -mu*delta
    return alpha

def kernel_minvar_sgd_update_alpha(alpha,mu,y,i,K,lda):
    """Update the present value of alpha given what we've seen so far
    """
    alpha[i] = 0.0
    alpha = (1-mu*lda)*alpha
    delta = (regressor(alpha,K,i) - y)
    upd = ((regressor(alpha,K,i) - y)*np.sqrt(K[i,i]) +
    np.sqrt(lda**2*np.dot(alpha.transpose(),np.dot(K,alpha))) + 2*lda*(regressor(alpha,K,i) - y)*
    regressor(alpha,K,i))
    alpha[i] = -mu*delta
    return alpha,upd

def loss(alpha,K,y):
    N = np.shape(alpha)[0]
    #alpha = np.reshape(alpha,(N,1))
    y_hat = np.dot(K,alpha)
    loss = np.dot((y-y_hat).T,(y-y_hat))
    return loss/float(N)

def get_weighted_index(N,sigma,mean = 50):
    """Returns an index by rounding a sample from a gaussian
    along with it's weight and the gaussian random variable"""
    s = np.random.randn()
    z = s*sigma + mean
    index = max(0,min(N-1,int(z)))
    if z < -0.5:
        weight = 0.0
    elif z > N + 0.5:
        weight =0.0
    else:
        weight = 1.0/norm.pdf(z,mean,sigma)
    return index,weight,z

def kernel_awsgd_alpha_update(alpha,mu,eps,tau,Y,i,weight,z,base_line,K,lmbda):
    """
    Do one kernel awsgd update
    Inputs:
        alpha - current dual vector
        mu - model learning rate
        eps - sampling learning rate
        tau - sampling parameter
        Y
        i - sampled index
        weight - importance weights
        z - random sample from gaussina
        base-line
        K - kernel matrix
        lmbda - regularisation parameter
    Outpus:
        alpha,tau,base_line
    """
    N = np.shape(Y)[0]
    y = Y[i]
    alpha = (1-mu*lmbda*float(weight))*alpha
    d = (regressor(alpha,K,i) - y)
    alpha [i] = -mu*d*float(weight)
    v = (np.linalg.norm(d)*float(weight))**2
    tau = tau - eps*(v - base_line)*(tau - z)
    base_line = v - 0.1*(base_line - v)
    return alpha,tau,base_line

class discrete_distribution(object):
    """ Discrete probability distribuiton on the range [1,N]
    """

    def __init__(self,probs):
        """Probs is a np array of probabilites"""
        self.Z = float(np.sum(np.abs(probs)))
        self.probs = probs/self.Z
        self.cumsum = np.cumsum(self.probs)

    def probability(self,index):
        return self.probs[index]

    def sample(self):
        seed = np.random.random()
        smp = 0
        while(seed > self.cumsum[smp]):
            smp = smp + 1
        return smp

    def update_prob(self,index,prob):
        prob = float(np.abs(prob))
        self.probs = self.probs*self.Z
        self.Z = self.Z - self.probs[index] + prob
        self.probs[index] = prob
        self.probs = self.probs/float(self.Z)
        self.cumsum = np.cumsum(self.probs)
