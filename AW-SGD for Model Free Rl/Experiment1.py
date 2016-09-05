import itertools
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import seaborn
from max_binary_heap import *
import pickle

class blind_cliff_walk(object):
    """MDP for the blind cliff-walk problem
    Action space is 1/0
    Terminal State is -1
    """
    
    def __init__(self,max_state):
        self.max_state = max_state
        self.state = 0
    
    def step(self,action):

        if self.state % 2 == 0:
            if action == 1 and self.state < self.max_state-1:
                self.state = self.state + 1
                reward = 0
            elif action == 1:
                self.state = -1
                reward = 1
            elif action == 0:
                self.state = -1
                reward = 0
            else:
                print('not a valid action')
                
        else:
            if action == 0 and self.state < self.max_state -1:
                self.state = self.state + 1
                reward = 0
            elif action == 0:
                self.state = -1
                reward = 1
            elif action == 1:
                self.state = -1
                reward = 0
            else:
                print('not a valid action')
                
        return self.state,reward
        
def get_history(size):
    """Perform all possible actions on a cliff-walk of size "size" and return
    the history as a list of lists each of wich contains <S,A,R,S'>"""
    
    actions = [seq for seq in itertools.product((0,1), repeat=size)]
    history = []
    
    for action_group in actions:
        # Get a new MDP
        mdp = blind_cliff_walk(size)
        for action in action_group:
            # Take action and record transition
            state = mdp.state
            state_prime,reward = mdp.step(action)
            history.append([state,action,reward,state_prime])
            if state_prime == -1:
                mdp.state = 0
                break
    return history       
    
def Q(state,action,theta):
    
    if state == -1:
        return 0
    else:
        size = np.size(theta)/2
        if action == 0:
            return theta[state]
        else:
            return theta[state + size]

def get_max_action(state,theta):
    size = np.size(theta)/2
    if theta[state] > theta[state + size]:
        return 0
    else:
        return 1
    
def get_ground_truth(size):
    """ Returns the optimal action value function
    for the bcf model of size 'size' """
    
    discounts = [(1-1.0/size)**(x) for x in range(size)][::-1]
    discounts = discounts + discounts
    left_actions = [m %2 for m in range(size)]
    right_actions = copy(left_actions[1:])
    right_actions.append((right_actions[-1]+1)%2)
    
    gt = left_actions + right_actions
    
    gt = [x*y for x,y in zip(gt,discounts)]
    
    return gt

def mean_sq_er(ground_truth,theta):
    size = np.size(theta)/2.0
    return (((ground_truth - theta).sum())**2)/size

def experience_replay_bcf(size,rate,tol,theta_init):
    """ Perform Q-Learning on the blind cliff walk MDP
    with experience replay"""
    
    # Create Store for Thetas
    thetas = []
    errors = []
    error = 10
    count = 0
    
    # Get History
    history = get_history(size)
    
    # Initialise discount and theta
    theta = theta_init.copy()
    theta_true = get_ground_truth(size)
    gamma = 1 - 1.0/size
    
    #perform SGD/Q-learning
    while(error > tol and count < 100*(2**size)):
        #Sample transition
        index = np.random.randint(np.shape(history)[0])
        state,action,reward,state_prime = history[index]
        # Calcualte TD-error
        Q_1 = Q(state,action,theta)
        action_max = get_max_action(state_prime,theta)
        Q_prime = Q(state_prime,action_max,theta)
        delta = (reward + gamma*Q_prime - Q_1)
        #Updtate Theta
        if action == 0:
            theta[state] = theta[state] + rate*delta
        elif action == 1:
            theta[state + size] = theta[state + size] + rate*delta
        error = mean_sq_er(theta_true,theta)
        thetas.append(theta)
        errors.append(error)
        count = count + 1
        
    return thetas,errors,count


def greedy_experience_replay_bcf(size,rate,tol,theta_init):
    """ Perform Q-Learning on the blind cliff walk MDP
    with experience replay"""
    
    # Create Store for Thetas and deltas and errors
    delta_store = max_binary_heap()
    thetas = []
    errors = []
    error = 10
    count = 0
    
    # Get History
    history = get_history(size)
    np.random.shuffle(history)
    priorities = [100]*len(history)
    
    for i in range(len(history)):
        delta_store.insert(priorities[i],i)
    
    # Initialise discount and theta and theta_true
    theta = theta_init.copy()
    gamma = 1 - 1.0/size
    theta_true = get_ground_truth(size)
    
    #perform SGD/Q-learning
    while(error > tol and count < (100*2**size)):
        #Sample transition
        index = delta_store.get_max()[2]
        #print(index)
        state,action,reward,state_prime = history[index]
        # Calcualte TD-error
        Q_1 = Q(state,action,theta)
        action_max = get_max_action(state_prime,theta)
        Q_prime = Q(state_prime,action_max,theta)
        delta = (reward + gamma*Q_prime - Q_1)
        #Update Priorities
        delta_store.insert(delta,index)
        #Updtate Theta
        if action == 0:
            theta[state] = theta[state] + rate*delta
        elif action == 1:
            theta[state + size] = theta[state + size] + rate*delta
        error = mean_sq_er(theta_true,theta)
        thetas.append(theta)
        errors.append(error)
        count = count + 1
        
        
    return thetas,errors,count

if __name__ == '__main__':

# Calculate the number of iterations to converge to MSE 1e-3 for Greedy and Non-Greedy Experience Replay

    experiments = []
    experiments_gr = []

    for size in range(3):
        print('Help!')
        size = size + 11
        theta_init = np.random.normal(0,0.1,size*2)
        counts_gr = []
        counts = []
        for exper in range(10):
            print("------------ Running Experiment {} of Size {}---------------".format(exper,size))
        
            _,_,countgr = greedy_experience_replay_bcf(size,0.25,0.01,theta_init)
            _,_,count = experience_replay_bcf(size,0.25,0.01,theta_init)
            counts_gr.append(countgr)
            counts.append(count)
    
        experiments.append(counts)
        experiments_gr.append(counts_gr)
        
        
        with open('ExperimentResults_non_greedy_greedy{}'.format(size),'w') as f:
            pickle.dump(experiments,f)
            pickle.dump(experiments_gr,f)
        
        