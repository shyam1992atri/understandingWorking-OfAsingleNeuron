# understanding of the back propagation algorithm
# simulating and gate with a single neuron
# using sigmoid activation function

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        # y = 1/(1+np.exp(-x)) -- sigmoid function
        # this is the derivative of the sigmoid function
        # y'=(-np.exp(-x))/(1+np.exp(-x))**2 = y*(1-y)
        # since we are passing the output y in our loop we have 
        # to express the derivative in terms of output got from sigmoid function
        # hence the expression x*(1-x) which is eqivalet to y*(1-y)
        return x*(1-x)
    # if derivative is false we send the value of the sigmoid function
    return 1/(1+np.exp(-x))
    
# input dataset
# here 1 in the initial row is 
# the biasing term
X = np.array([  [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,0,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# this will initialise the initial weights randomly
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
    
    # this is the forward propogation strep 
    # where activation is calculated for all the hidden layers
    l0 = X
    # this function will compute the activation steps for the sigmoid function
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    # this step computes the error in each and every step
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    # this is the back propogation step
    # back propogated error is error times the derivative
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    # here we update our initial weights with the ones 
    # computed from the backpropogation step
	# input dataset transpose * back  propogating error from the step ahead
	# this syn0 will hold the learning outcome
    syn0 += np.dot(l0.T,l1_delta)
	

print ("Output After Training:")
print (l1)
print ("this is predection step")
print (1 / (1 + np.exp(-(np.dot(np.array([1, 1, 0]), syn0)))))
