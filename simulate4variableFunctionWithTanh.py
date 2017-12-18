# understanding of the back propagation algorithm
# simulating y=a*b+c*d with a single neuron
# using sigmoid activation function

import numpy as np

#tanh- tan hyperbolic function
def nonlin(x,deriv=False):
    if(deriv==True):
	# this is the tan hyperbolic function
	# y= (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))=tanhx
	# this is the derivative of tan hyperbolic function
	# y'= (2/(np.exp(x)+np.exp(-x)))**2 = (sech(x))^2 = 1-(tanh(x))^2 = 1-y**2
	# we are computing the derivative in terms of input
        return (1-x**2)
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
# input dataset
# here 1 in the initial row is 
# the biasing term
X = np.array([
[1,0,	0,	0,	0],
[1,1,	0,	0,	0],
[1,0,	1,	0,	0],
[1,1,	1,	0,	0],
[1,0,	0,	1,	0],
[1,1,	0,	1,	0],
[1,0,	1,	1,	0],
[1,1,	1,	1,	0],
[1,0,	0,	0,	1],
[1,1,	0,	0,	1],
[1,0,	1,	0,	1],
[1,1,	1,	0,	1],
[1,0,	0,	1,	1],
[1,1,	0,	1,	1],
[1,0,	1,	1,	1],
[1,1,	1,	1,	1],
])
    
# output dataset            
y = np.array([[
0,
0,
0,
1,
0,
0,
0,
1,
0,
0,
0,
1,
1,
1,
1,
1
]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# this will initialise the initial weights randomly
syn0 = 2*np.random.random((5,1)) - 1

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
	# updating weights is done as 
	# input dataset transpose * back  propogating error from the step ahead
	# this syn0 will hold the learning outcome
    syn0 += np.dot(l0.T,l1_delta)
	

print ("Output After Training:")
print (l1)
