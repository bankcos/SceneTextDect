import cv2
import numpy as np
import csv

#read image
img = cv2.imread("D:\\project\\test2.jpg",cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Image") 
cv2.imshow("Image", img)  

detector = cv2.SIFT()

#writer = csv.writer(file('D:\\a.csv', 'wb'))
#writer.writerow(['Column1'])

#writer.writerow(keypoints)
#生成关键点
keypoints = detector.detect(gray,None)
img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
res = detector.compute(gray,keypoints)

print res


cv2.imshow('SIFT',img);
cv2.waitKey(0)
cv2.destroyAllWindows()


