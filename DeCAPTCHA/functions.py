
import cv2
import pandas
import numpy as np 
import os

def load_images_from_filenames(filenames):
    images = [] #filenames are given in pre sorted order 
    for filename in filenames:
        # print(filename)
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images

def rgb_to_hsv(X_rgb):
  X_hsv=[]
  for img in X_rgb:
    X_hsv.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
  return X_hsv

def convert_to_numpy(img):
    img= np.array(img)
    img= np.reshape(img, (img.shape[0],img.shape[1],1))
    return img

def processImage(img):
  #erode
  kernel = np.ones((5, 5), np.uint8)
  # Using cv2.erode() method 
  img = cv2.erode(img, kernel) 

  red = img[0][0][0]
  green = img[0][0][1]
  blue = img[0][0][2]

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      for k in range(3):
        if k==0:
          if img[i][j][k] < red:
            img[i][j][k]=0
          else:
            img[i][j][k] = img[i][j][k]-red
        elif k==1:
          if img[i][j][k] < green:
            img[i][j][k]=0
          else:
            img[i][j][k] = img[i][j][k]-green
        else:
          if img[i][j][k] < blue:
            img[i][j][k]=0
          else:
            img[i][j][k] = img[i][j][k]-blue

  #Removing the background to white
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      for k in range(3):
        img[i][j][k]=255-img[i][j][k]
  
  grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  filename= 'grayimage.png'
  cv2.imwrite(filename,grayimg)

  for i in range(grayimg.shape[0]):
    for j in range(grayimg.shape[1]):
      if(grayimg[i][j]!=255):
        grayimg[i][j]=0
  
  p1start=-1
  p1end=-1
  sizeOfImage = int(500/3)+1
  for j in range(grayimg.shape[1]):
    sum=0
    for i in range(grayimg.shape[0]):
      if(grayimg[i][j]!=255):
        sum+=1

    if(p1start==-1 and sum>=20):
      p1start=j
    elif(p1start!=-1 and sum<20):
      p1end=j
      break

  # print(p1end,p1start)
  img0 = np.zeros((150,p1end+5-p1start+5))
  img0 = grayimg[:,p1start-4:p1end+5]

  toadd = sizeOfImage - img0.shape[1]
  righthalf = int(toadd/2)
  lefthalf = int(toadd-righthalf)

  img0 = np.append(img0,255*np.ones((150,righthalf)),axis=1)
  img0 = np.append(255*np.ones((150,lefthalf)),img0,axis=1)

  p2start=-1
  p2end=-1
  for j in range(p1end+1,grayimg.shape[1]):
    sum=0
    for i in range(grayimg.shape[0]):
      if(grayimg[i][j]!=255):
        sum+=1
    if(p2start==-1 and sum>=20):
      p2start=j
    elif(p2start!=-1 and sum<20):
      p2end=j
      break

  img1 = np.zeros((150,p2end+5-p2start+5))
  img1 = grayimg[:,p2start-4:p2end+5]

  toadd = sizeOfImage - img1.shape[1]
  righthalf = int(toadd/2)
  lefthalf = int(toadd-righthalf)

  img1 = np.append(img1,255*np.ones((150,righthalf)),axis=1)
  img1 = np.append(255*np.ones((150,lefthalf)),img1,axis=1)

  p3start=-1
  p3end=-1
  for j in range(p2end+1,grayimg.shape[1]):
    sum=0
    for i in range(grayimg.shape[0]):
      if(grayimg[i][j]!=255):
        sum+=1
    if(p3start==-1 and sum>=20):
      p3start=j
    elif(p3start!=-1 and sum<20):
      p3end=j
      break

  img2 = np.zeros((150,p3end+5-p3start+5))
  img2 = grayimg[:,p3start-4:p3end+5]

  toadd = sizeOfImage - img2.shape[1]
  righthalf = int(toadd/2)
  lefthalf = int(toadd-righthalf)

  img2 = np.append(img2,255*np.ones((150,righthalf)),axis=1)
  img2 = np.append(255*np.ones((150,lefthalf)),img2,axis=1)

  for i in range(img0.shape[0]):
    for j in range(img0.shape[1]):
      img0[i][j]=255-img0[i][j]

  for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
      img1[i][j]=255-img1[i][j]
  
  for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
      img2[i][j]=255-img2[i][j]
  img0= convert_to_numpy(img0)
  img1= convert_to_numpy(img1)
  img2= convert_to_numpy(img2)
  return img0,img1,img2

def segmentation(X):
	ct=0
	segmented_images=[]
	for img in X:
		# print(ct)
		try:
			l1,l2,l3=processImage(img)
		except:
			print("Error in image "+ str(ct/3)+".png")
			
		else:
			# cv2.imwrite(os.path.join("new_data/" , str(ct)+".png"),l1)
			# ct+=1
			# cv2.imwrite(os.path.join("new_data/" , str(ct)+".png"),l2)
			# ct+=1
			# cv2.imwrite(os.path.join("new_data/" , str(ct)+".png"),l3)
			# ct+=1
			# print(l1.shape,l2.shape,l3.shape)
			segmented_images.append(l1)
			segmented_images.append(l2)
			segmented_images.append(l3)
	return segmented_images
	    
	
