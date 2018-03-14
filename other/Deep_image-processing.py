import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd

frame             = cv2.imread("C:\\Udemy\\Practical Data Science in Python\\x.jpg")
frame             = cv2.resize(frame,(800,600), interpolation = cv2.INTER_CUBIC)


hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([22,0,0])
upper_blue = np.array([180,222,360])
    
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res  = cv2.bitwise_and(frame,frame, mask= mask)


kernel = np.ones((55,55),np.uint8)
params = cv2.SimpleBlobDetector_Params()
     

params.minThreshold = 0;
params.maxThreshold = 256;
     

params.filterByArea = True
params.minArea = 2



detector = cv2.SimpleBlobDetector_create(params)
reversemask=255-mask
keypoints = detector.detect(reversemask)
im_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

x = np.empty((len(keypoints),2))
i = 0


 
for k in keypoints:
    print(1)
    x[i,0] = k.pt[0]
    x[i,1] = k.pt[1]
    #cv2.circle(frame, (int(k.pt[0]), int(k.pt[1])), int(k.size), (0, 0, 255), -1)
    i = i + 1
    



ds      = DBSCAN(eps=70, min_samples=2, metric='euclidean', algorithm='auto', p=None).fit(x)
labels = ds.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Number of Nuts: %d' % n_clusters_)

final = pd.DataFrame(labels)
data  = pd.DataFrame(x)
final = pd.concat((final,data),axis=1)
final.columns = ['index', 'x','y']
average = final.groupby("index").mean().values

average = average.astype(np.int64)


for row in range(average.shape[0]):
    cv2.circle(frame,(average[row,0],average[row,1]), 10, (0,0,255), -1)

cv2.imshow('frame',frame)
cv2.imshow('res',mask)
