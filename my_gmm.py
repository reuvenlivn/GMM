# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:15:24 2019

@author: reuve
"""

import numpy as np
import pylab as plt  
#from collections import namedtuple
from sklearn.datasets import load_iris

#Data preparation
#1. Load the iris dataset .
iris = load_iris()
data = iris.data

#2. randomly choose the starting centroids/means as three of the points from datasets
N_centers = 3
#n N_instance 
#d N_param
N_instance, N_param = data.shape

centers = data[np.random.choice(N_instance, N_centers, False), :]

#3. initialize the variances for each gaussians
Sigma= [np.eye(N_param)] * N_centers

#4. initialize the probabilities/weights for each gaussians, as equally distributed
weights = [1./N_centers] * N_centers

#5. Responsibility (membership) matrix is initialized to all zeros
Responsibility = np.zeros((N_instance, N_centers))


#Expectation
#6. Write a P function that calculates for each point the probability of belonging to each gaussian
def prob (sigma,center): 
    p_data = data - np.tile(np.transpose(center), (N_instance, 1))
    prob_res = np.sum(np.dot(p_data, np.linalg.inv(sigma))*p_data, 1)
    prob_res = np.exp(-0.5*prob_res) / np.sqrt(
            (np.power((2*np.pi), N_param))*np.absolute(np.linalg.det(sigma))
            )      
    return prob_res 

#7. Write the E-step (expectation) in which we multiply this P function for every 
#   point by the weight of the corresponding cluster
#   parameter R: the Responsibility (membership) matrix
def E_Step(R):
    
    for i in range(N_centers):
        R[:, i] = weights[i] * prob(Sigma[i], centers[i])
     
    #9. Normalize the responsibility matrix by the sum
    R = (R.T / np.sum(R, axis = 1)).T
       
    #8. Sum the log likelihood of all clusters
    # shuild be done after updating R
    log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
    
    #10. Calculate the number of points belonging to each Gaussian 
    N_Gaussian = np.sum(R, axis = 0)
    return log_likelihood, N_Gaussian, R


#Maximization
#11. Write the M-step (maximization) in which we calculate for each gaussian utilizing the new responsibilities:
#parameter: number of instance in the current gaussian
#update the centers, weights and Sigma for the current gaussian
def M_step(N_Gaussian):
    for i in range(N_centers):
        #a. the new mean(centers)
        centers[i] = 1. / N_Gaussian[i] * np.sum(Responsibility[:, i] * data.T, axis = 1).T
        distance = np.matrix(data - centers[i])
        
        #b. The new variance
        Sigma[i] = np.array(1 / N_Gaussian[i] * 
             np.dot(np.multiply(distance.T,  Responsibility[:, i]), distance))
        
        #c. The new weights
        weights[i] = 1. /  N_instance * N_Gaussian[i]


def print_clusters(data,labels):
#    plt.figure(figsize = (7,5))   
#    plt.scatter(data.T[2], data.T[3], c=labels, marker ="o", s=50);
#    plt.show()
    plt.title('GMM 2 3')
    plt.scatter(data.T[2], data.T[3], c=labels, marker ="o", s=40);
    plt.show()
    plt.title('GMM 0 1')
    plt.scatter(data.T[0], data.T[1], c=labels, marker ="o", s=40);
    plt.show()
    plt.title('GMM 1 3')
    plt.scatter(data.T[1], data.T[3], c=labels, marker ="o", s=40);
    plt.show()
    plt.title('GMM 0 3')
    plt.scatter(data.T[0], data.T[3], c=labels, marker ="o", s=40);
    plt.show()
    plt.title('GMM 1 2')
    plt.scatter(data.T[1], data.T[2], c=labels, marker ="o", s=40);
    plt.show()
    plt.title('GMM 0 2')
    plt.scatter(data.T[0], data.T[2], c=labels, marker ="o", s=40);
    plt.show()
    


#12. Iterate over Expectation-Maximization till we have 1000 iterations or till the 
#log-likelihood changes less than 0.0001
eps = 0.001 #0.0001
max_iters = 500
log_likelihoods = []
log_likelihood = 0

while len(log_likelihoods) < max_iters or np.abs(log_likelihood - log_likelihoods[-2]) > eps:   
    log_likelihood, N_Gaussian, Responsibility = E_Step(Responsibility)
    log_likelihoods.append(log_likelihood)
    M_step(N_Gaussian)

#print (log_likelihoods)   
# show
labels=np.argmax(Responsibility, axis=1)
print_clusters(data,labels)
    