import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
import os

def readFile(filename):
    rows = csv.reader(open(filename,"rb"))
    data = list(rows)
    list1 = []
    for i in range(len(data)):
    	temp = data[i][0]
    	list1.append(temp.split(';'))
         
    return list1
        
csvname = "allAnnotations.csv"
data =  readFile(csvname)

fileCommon = "signDatabasePublicFramesOnly/";
for i in range(1,len(data)):
	upperX=data[i][2]
	upperY=data[i][3]
	lowerX=data[i][4]
	lowerY=data[i][5]
	label=data[i][1]
	directory = "train/"+label
	print i
	if not os.path.exists(directory):
                os.makedirs(directory)
	path=fileCommon+data[i][0]
	if str(label)=="signalAhead":
		imgFile = cv2.imread(path)
		imgFile = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
		obj = imgFile[int(upperY):int(lowerY) , int(upperX):int(lowerX)]
		cv2.imwrite('train/{0}/{1}.png'.format(label,i),obj)
	else:
		imgFile = cv2.imread(path)
		imgFile = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
		obj = imgFile[int(upperY):int(lowerY) , int(upperX):int(lowerX)]
		cv2.imwrite('train/{0}/{1}.png'.format(label,i),obj)

        
