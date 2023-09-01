import numpy as np
from numpy import random as rand
import math, scipy
from scipy.spatial.distance import cdist

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# PLEASE BE CAREFUL THAT ERROR CLASS NUMBERS START FROM 1 AND NOT 0. THUS, THE FIFTY ERROR CLASSES ARE
# NUMBERED AS 1 2 ... 50 AND NOT THE USUAL 0 1 ... 49. PLEASE ALSO NOTE THAT ERROR CLASSES 33, 36, 38
# NEVER APPEAR IN THE TRAINING SET NOR WILL THEY EVER APPEAR IN THE SECRET TEST SET (THEY ARE TOO RARE)

# Input Convention
# X: n x d matrix in csr_matrix format containing d-dim (sparse) bag-of-words features for n test data points
# k: the number of compiler error class guesses to be returned for each test data point in ranked order

# Output Convention
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of classes with the i-th row 
# containing k error classes which it thinks are most likely to be the correct error class for the i-th test point.
# Class numbers must be returned in ranked order i.e. the label yPred[i][0] must be the best guess for the error class
# for the i-th data point followed by yPred[i][1] and so on.

# CAUTION: Make sure that you return (yPred below) an n x k numpy nd-array and not a numpy/scipy/sparse matrix
# Thus, the returned matrix will always be a dense matrix. The evaluation code may misbehave and give unexpected
# dist if an nd-array is not returned. Please be careful that classes are numbered from 1 to 50 and not 0 to 49.

# def findErrorClass( X, k ):
# 	# Find out how many data points we have
# 	n = X.shape[0]
# 	# Load and unpack a dummy model to see an example of how to make predictions
# 	# The dummy model simply stores the error classes in decreasing order of their popularity
# 	npzModel = np.load( "/Users/tejasr/Documents/IITK/Semesters/sem 5/CS771/Assignments/assn2/sample_submit/model.npz" )
# 	model = npzModel[npzModel.files[0]]
# 	# Let us predict a random subset of the 2k most popular labels no matter what the test point
# 	shortList = model[0:2*k]
# 	# Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
# 	yPred = np.zeros( (n, k) )
# 	for i in range( n ):
# 		yPred[i,:] = rand.permutation( shortList )[0:k]
# 	return yPred

def matrix_dist(X, prototypes): #matrix version 
    dist= np.zeros((X.shape[0],1))
    M=X-prototypes
    dist= np.linalg.norm(M, axis=1)
    return dist

def euclidean_dist(X, prototype): #matrix version 
    dist= np.zeros((X.shape[0],1)) #(10000,1) 
    M=X-prototype
    dist= np.linalg.norm(M, axis=1)
    return dist

def mahalanobis_dist(X, prototype): #matrix version 
    # dist= np.zeros((X.shape[0],1)) #(10000,1) 
    # V = np.linalg.inv(np.cov(np.concatenate((X, prototype)).T))
    # dist= scipy.spatial.distance.mahalanobis(X, prototype, V)
    # inv_covmat = np.linalg.inv(cov)
    # left = np.dot(y_mu, inv_covmat)
    # mahal = np.dot(left, y_mu.T)
    # return mahal.diagonal()
    dist =  cdist(X,prototype,'mahalanobis')
    dist = np.diag(dist)
    return dist

# def mahalanobis_dist(X, prototype): #matrix version 
#     dist =  cdist(X,prototype,'mahalanobis')
#     dist = np.diag(dist)
#     return dist
  
def findErrorClass( X, k ):
		# Find out how many data points we have
	n = X.shape[0]
	W = np.load( "/Users/tejasr/Documents/IITK/Semesters/sem 5/CS771/Assignments/assn2/sample_submit/model_lwp.npy" )
	print("W has shape:",W.shape)
	# W has shape (50,225) and stores all the prototypes
	# Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
	yPred = np.zeros((n, k))
	err_class=50
	m= np.zeros((n, err_class)) #(10000,50)
	print("m has shape:",m.shape)
	for i in range(0, err_class):
		M= matrix_dist(X, W[i]) # M is a (10000,1) vector measuring distance from the ith prototype
		m[:, i]= M # the ith row, jth column of m denotes the distance of the ith example from the jth prototype
	for i in range(n):
		Top=[]
		for _ in range(k): #This for loop gets the top 5 values from the predicted errors: is there a better way? 
			a=np.argmin(m[i])
			Top.append(a+1)
			m[i][a]= math.inf
			# m[i][a]=-1
		if(i%50==0):
			print(Top)
		yPred[i,:]=np.array(Top)
	# for i in range( n ):
	# 	yPred[i,:] = rand.permutation( shortList )[0:k]
	return yPred

# def findErrorClass( X, k ):
# 	# Find out how many data points we have
# 	n = X.shape[0]
# 	W = np.load( "/Users/tejasr/Documents/IITK/Semesters/sem 5/CS771/Assignments/assn2/assn2MyCopy/sample_submit/sample_submit/modelv2.npy" )
# 	# Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
# 	yPred = np.zeros((n, k))

# 	# The model W has shape (226,50)
#     #we hide the bias term in X, so for a given data point x of dimensions (1, 226), you have a W model vector of dimension (226, 50)
#     # thus, x.W outputs a vector of shape (1,50), which is tadaaaaa
# 	for i in range(n):
# 		x=X[i]
# 		#still stored as a sparse representation 
# 		x=x.toarray() #converts to dense
#    		# x has shape (1,225)
# 		one=np.ones((x.shape[0],1)) #(1,1) size 
# 		xt=np.hstack((x,one))
#   		#xt has shape (1,226): I guess this just added the bias term 
# 		A=np.dot(xt,W)
# 		A1=sigmoid(A) # A is size (1,50)
# 		A2=np.squeeze(A1) # A2 is size (50,) : squeezes out the final dimension 
# 		Top=[]
		# for _ in range(k): #This for loop gets the top 5 values from the predicted errors: is there a better way? 
		# 	a=np.argmax(A2)
		# 	Top.append(a+1)
		# 	A2[a]=-1
		# # print("Top at i=",i,"is:", Top )
		# yPred[i,:]=np.array(Top)
# 	# for i in range( n ):
# 	# 	yPred[i,:] = rand.permutation( shortList )[0:k]
# 	return yPred