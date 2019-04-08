import sys

min1=sys.maxsize
min2=sys.maxsize
a = [1,2,3,0,5,6]
for j in a:
	sd = j
	if(sd < min1):
		min2 = min1
		min1 = sd
	elif(sd<min2):
		min2 = sd
print("Minimum: ")
print(min1,min2)