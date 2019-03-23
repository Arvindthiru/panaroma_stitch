import argparse
import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

### reads the image from given location
def read_img(img):
	img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
	return img
### gets matches whose distance is less than specified ratio
def get_matches(allMatches):
	ms = []
	matches = []
	for m1,m2 in allMatches:
		change = m2.distance * 0.75
		if m1.distance < change :
			matches.append((m1.trainIdx,m1.queryIdx))
			ms.append(m1)
	return matches,ms

### detect features using SIFT	
def detect_sift(img1,img2):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1,des1 = sift.detectAndCompute(img1,None)
	kp2,des2 = sift.detectAndCompute(img2,None)
	kimg1 = cv2.drawKeypoints(img1,kp1,None)
	kimg2 = cv2.drawKeypoints(img2,kp2,None)
	cv2.imwrite("./results/sift_nevada5.jpg",kimg1)
	cv2.imwrite("./results/sift_nevada4.jpg",kimg2)
	print(len(kp1))
	print(len(kp2))
	print(kp1[0].pt)
	#print(kp1[0].pt)
	#print(des1[0])
	bf = cv2.BFMatcher()
	allMatches = bf.knnMatch(des1,des2,2)
	matches, ms = get_matches(allMatches)
	#print(ms)
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,ms[:40],None,flags=2)

	cv2.imwrite("./results/sift_match.jpg",img3)
	raise NotImplementedError



def main():

	#print("Argument is:",str(sys.argv[1]))
	file_path = sys.argv[1]
	files = os.listdir(file_path)
	print(files)
	img1 = read_img("./data/"+str(files[0]))
	img2 = read_img("./data/"+str(files[1]))
	print(np.shape(img1))
	print(np.shape(img2))
	detect_sift(img1,img2)


if __name__ == "__main__":
    main()