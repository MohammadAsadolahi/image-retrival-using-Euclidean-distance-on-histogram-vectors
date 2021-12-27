from tensorflow.keras import datasets
import numpy as np
(train_images, train_labels), (test_images, test_labels)= datasets.cifar10.load_data()
images=np.concatenate((train_images,test_images))
labels=np.concatenate((train_labels,test_labels))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def plotHistogram(histogram):
  plt.bar(range(len(histogram)), list(histogram), align='center')
  plt.xticks([i for i in range(256)], [i for i in range(256)])
  plt.show()

def plotChannel(channel):
  plt.imshow(channel,cmap='gray',vmin = 0, vmax = 255)
  plt.show()

def vectorDistance(v1,v2):
  return np.mean(np.square(v1-v2))

def getHist(channel):
  unique, counts = np.unique(channel, return_counts=True)
  points=dict(zip(unique, counts))
  for i in range(256):
    if i not in points:
      points[i]=0
  hist=[]
  for i in range(256):
    hist.append(points[i])
  return np.array(hist)

def imageDistance(img1,img2):
  splitedImage1=np.split(img1,[0,1,2],axis=2)
  r1=getHist(np.squeeze(splitedImage1[1]))
  g1=getHist(np.squeeze(splitedImage1[2]))
  b1=getHist(np.squeeze(splitedImage1[3]))
  
  splitedImage2=np.split(img2,[0,1,2],axis=2)
  r2=getHist(np.squeeze(splitedImage2[1]))
  g2=getHist(np.squeeze(splitedImage2[2]))
  b2=getHist(np.squeeze(splitedImage2[3]))
  
  redDistance=vectorDistance(r1,r2)
  greenDistance=vectorDistance(g1,g2)
  blueDistance=vectorDistance(b1,b2)
  return redDistance+greenDistance+blueDistance

# for referenceIndex in for i in [29,4,6,9,3,27,0,7,8,1]:

referenceIndex=29
distanceMatrix=[]
for i in range(1000):
  distanceMatrix.append(imageDistance(images[referenceIndex],images[i]))
distanceMatrix=sorted(range(len(distanceMatrix)), key=lambda k: distanceMatrix[k])
top10=distanceMatrix[1:11:]
match=0
plt.imshow(images[referenceIndex])
plt.show()
print(f"++++++++++++++++++ mathced images for picture {referenceIndex}.jpg: ++++++++++++++++++")
for i in top10:
  plt.imshow(images[i])
  plt.show()
  if labels[referenceIndex]==labels[i]:
    match+=1
print(f"item: {referenceIndex}.jpg  amount of: {match*10}% of similar pictures are form item's category")
print(f"\n\n\n\n")

# for plot red channle of firt airplane picture:
#
# splited=np.split(images[29],[0,1,2],axis=2)
# red_channel=np.squeeze(splited[1])
# green_channel=np.squeeze(splited[2])
# blue_channel=np.squeeze(splited[3])
# plotChannel(red_channel)
# plotHistogram(getHist(red_channel))
