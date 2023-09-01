# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images
import cv2
import numpy as np 
import pickle
import functions
import keras
# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.
reverse_mapping= {0: 'ALPHA', 1: 'BETA', 2: 'CHI', 3: 'DELTA', 4: 'EPSILON', 5: 'ETA', 6: 'GAMMA', 7: 'IOTA', 8: 'KAPPA', 9: 'LAMDA', 10: 'MU', 11: 'NU', 12: 'OMEGA', 13: 'OMICRON', 14: 'PHI', 15: 'PI', 16: 'PSI', 17: 'RHO', 18: 'SIGMA', 19: 'TAU', 20: 'THETA', 21: 'UPSILON', 22: 'XI', 23: 'ZETA'}
# reverse_mapping= {0: 'ALPHA', 1: 'BETA', 2: 'CHI', 3: 'DELTA', 4: 'EPSILON', 5: 'ETA', 6: 'GAMMA', 7: 'IOTA', 8: 'KAPPA', 9: 'LAMDA', 10: 'MU', 11: 'NU', 12: 'OMEGA', 13: 'OMICRON', 14: 'PHI', 15: 'PI', 16: 'PSI', 17: 'RHO', 18: 'SIGMA', 19: 'TAU', 20: 'THETA', 21: 'UPSILON', 22: 'XI', 23: 'ZETA'}

def decaptcha( filenames ):
	X=[]
	# print(filenames[0:10])
	X=functions.load_images_from_filenames(filenames)
	X=functions.rgb_to_hsv(X)
	X= functions.segmentation(X) #Should be three times the size of X 
	X= np.array(X)
	print(X.shape)
	model= keras.models.load_model("model_final")
	y_pred = model.predict(X, batch_size= 50, verbose=1)
	y_pred=np.argmax(y_pred, axis=1)
	predictions=[]
	labels=[]
	for i in range(y_pred.shape[0]):
		predictions.append(reverse_mapping[y_pred[i]])
	for i in range(len(predictions)): # I will go from 0 to 5999
		if(i%3==0):
			s=""
			s+=predictions[i]+","
		elif(i%3==1):
			s+=predictions[i]+","
		elif(i%3==2):
			s+=predictions[i]
			labels.append(s)
	return labels # A list of strings in the required form 