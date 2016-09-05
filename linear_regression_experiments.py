import numpy as np
from scipy.stats import norm
# ------------ Utility Functions ------------------ #

def GenData(N,D,sigma):
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

def lin_reg_grad(X,y,w):
    """Returns Gradient of Square Loss
    """

    N = np.shape(X)[0]
    gram = np.dot(X.transpose(),X)
    grad = (np.dot(gram,w) - np.dot(X.transpose(),y))/N
    return grad

def get_error(X,y,w):
    """Returns the Square Error"""
    N = np.shape(X)[0]
    res = np.dot(X,w) - y
    cost = np.dot(res.transpose(),res)/N
    return float(cost)

class discrete_distribution(object):
    """ Discrete probability distribuiton on the range [1,N]
    """

    def __init__(self,probs):
        """Probs is a np array of probabilites"""
        self.Z = float(np.sum(np.abs(probs)))
        self.probs = np.abs(probs)/self.Z
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
        self.probs = self.probs*self.Z
        self.Z = self.Z - self.probs[index] + np.abs(prob)
        self.probs[index] = np.abs(prob)
        self.probs = self.probs/float(self.Z)
        self.cumsum = np.cumsum(self.probs)

# --------------- Regression Functions --------------- #

def gradient_descent(X,y,w_0,mu,tol,max_iter):
    """"Performs Gradient descent on square loss objective
    and returns the erros and w"""
    errors = []
    error = 1000
    num_iter = 1
    w = w_0
    while(error > tol and num_iter < max_iter):
         w = w - mu*lin_reg_grad(X,y,w)
         num_iter = num_iter + 1
         error = get_error(X,y,w)
         errors.append(error)
    return w,errors

def stochastic_gradient_descent(X,y,w_0,mu,tol,max_iter):
    """Performs SGD and returns the errors and w"""
    errors = []
    error = 1000
    num_iter = 1
    w = w_0
    N = np.shape(X)[0]
    while(error > tol and num_iter < max_iter):
         index = np.random.randint(N)
         delta = np.dot(X[index,:],w) - y[index]
         w = w - mu*delta*X[index:index+1,:].transpose()
         num_iter = num_iter + 1
         error = get_error(X,y,w)
         errors.append(error)
    return w,errors

def min_var_sgd(X,y,w_0,mu,tol,max_iter):

    N = np.shape(X)[0]
    priorities = np.ones(N)*100
    delta_dist = discrete_distribution(priorities)
    errors = []
    error = 1000
    num_iter = 1
    w = w_0

    while(error > tol and num_iter < max_iter):
         index = delta_dist.sample()
         delta = np.dot(X[index,:],w) - y[index]
         w = w - mu*delta*X[index:index+1,:].transpose()
         num_iter = num_iter + 1
         delta_dist.update_prob(index,np.linalg.norm(delta)/10)
         error = get_error(X,y,w)
         errors.append(error)
    return w,errors,delta_dist


def AW_SGD(X,y,w_0,mu_1,mu_2,tol,max_iter):

    errors = []
    error = 1000
    num_iter = 1
    w = w_0
    N = np.shape(X)[0]
    tau = N/2.0
    taus = []
    sigma = N/6.0
    base = 0.0

    while(error > tol and num_iter < max_iter):
         taus.append(tau)
         z = -1.0
         while (z < -0.5 or z > N + 0.5):
             z = np.random.normal(tau,sigma)
             index = max(0,min(int(z),sigma))
             weight = 1.0/norm.pdf(z,tau,sigma)
         delta = np.dot(X[index,:],w) - y[index]
         delta = delta*X[index:index+1,:].transpose()
         w = w - mu_1*delta*weight
         v = float(np.dot(delta.transpose(),delta))
         if type(v) is float:
             tau = tau - mu_2*(v-base)*(z-tau)
             base = base - (1.0/num_iter)*(base - v)
         num_iter = num_iter + 1
         error = get_error(X,y,w)
         errors.append(error)
    return w,errors,taus
