import numpy as np
import cv2

name = 'v12'

video = cv2.VideoCapture(name + '.mp4')
totalVids = []

success, frame = video.read()

for i in range(100):
    success, frame = video.read()
    totalVids.append(frame)
totalVids = np.array(totalVids)

average = np.mean(totalVids, 0)
print(np.amax(average))

average = average.astype(np.uint8)
cv2.imwrite(name + '_averaged.png', average)
cv2.imshow("image", average)
cv2.waitKey(0)
