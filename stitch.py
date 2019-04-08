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
'''
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
'''
### Finds sum of squared difference
def get_ssd(d1,d2):
	length = len(d1)
	sum = 0
	d3 = d1 - d2
	sum = sum + np.linalg.norm(d3)
	return sum

### Returns matched keypoints
def get_mapped_keypoints(kp1,kp2,des1,des2):
	sd = 0
	mp1 = []
	mp2 = []
	len1 = len(des1)
	len2 = len(des2)
	j_val = 0
	for i in range(0,len1):
		min1 = sys.maxsize
		min2 = sys.maxsize
		for j in range(0,len2):
			sd = get_ssd(des1[i],des2[j])
			if(sd < min1):
				min2 = min1
				min1 = sd
				j_val = j
			elif(sd<min2):
				min2 = sd
		#print("Minimum from calculation: ")
		#print(min1,min2)
		ratio = min1/min2
		if(ratio < 0.5):
			print("Ratio:" + str(ratio))
			print("Matched keypoints for"+ str(i) +", "+str(j_val))
			mp1.append(kp1[i].pt)
			mp2.append(kp2[j_val].pt)
			print(kp1[i].pt,kp2[j_val].pt)
		#raise NotImplementedError
	return mp1,mp2
	
### detect features using SIFT	
def detect_sift(img1,img2):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1,des1 = sift.detectAndCompute(img1,None)
	kp2,des2 = sift.detectAndCompute(img2,None)
	kimg1 = cv2.drawKeypoints(img1,kp1,None)
	kimg2 = cv2.drawKeypoints(img2,kp2,None)
	cv2.imwrite("./results/sift_nevada5.jpg",kimg1)
	cv2.imwrite("./results/sift_nevada4.jpg",kimg2)
	#print(kp1[0].pt)
	print("Matching key points wait for 5-6 minutes")
	mp1,mp2 = get_mapped_keypoints(kp1,kp2,des1,des2)
	print(len(mp1))
	print(len(mp2))
	#print(des1[0])
	'''
	bf = cv2.BFMatcher()
	allMatches = bf.knnMatch(des1,des2,2)
	matches, ms = get_matches(allMatches)
	#print(ms)
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,ms[:40],None,flags=2)

	cv2.imwrite("./results/sift_match.jpg",img3)
	'''
	raise NotImplementedError



def main():

	print("Argument is:",str(sys.argv[1]))
	file_path = sys.argv[1]
	files = os.listdir(file_path)
	print(files)
	img1 = read_img("./data/"+str(files[0]))
	img2 = read_img("./data/"+str(files[1]))
	print(np.shape(img1))
	print(np.shape(img2))
	#raise NotImplementedError
	detect_sift(img1,img2)


if __name__ == "__main__":
    main()