import os, PIL
import numpy as np 
import cv2
from PIL import Image

v1 = cv2.VideoCapture('v01.mp4')
# v2 = cv2.VideoCapture('v02.mp4')
# v3 = cv2.VideoCapture('v03.mp4')
# v4 = cv2.VideoCapture('v04.mp4')
# v5 = cv2.VideoCapture('v05.mp4')
# v6 = cv2.VideoCapture('v06.mp4')
# v7 = cv2.VideoCapture('v07.mp4')
# v8 = cv2.VideoCapture('v08.mp4')
# v9 = cv2.VideoCapture('v09.mp4')
# v10 = cv2.VideoCapture('v10.mp4')
# v11 = cv2.VideoCapture('v11.mp4')
# v12 = cv2.VideoCapture('v12.mp4')

numused = 0
average1 = np.zeros((1080, 1920, 3), dtype=np.uint16)
h = average1.shape[0]
w = average1.shape[1]
c = average1.shape[2]

while(numused<50):
    ret1, frame1 = v1.read()
    # ret2, frame2 = v2.read()
    # ret3, frame3 = v3.read()
    # ret4, frame4 = v4.read()
    # ret5, frame5 = v5.read()
    # ret6, frame6 = v6.read()
    # ret7, frame7 = v7.read()
    # ret8, frame8 = v8.read()
    # ret9, frame9 = v9.read()
    # ret10, frame10 = v10.read()
    # ret11, frame11 = v11.read()
    # ret12, frame12 = v12.read()
    # np_frame = cv2.imread('v01.mp4', frame1)
    # image1.append(np_frame)

    if ret1 == False:
        break
    if numused > 0: 
        for j in range (h):
            for k in range (w):
                average1[j][k][0] += frame1[j][k][0]
                average1[j][k][1] += frame1[j][k][1]
                average1[j][k][2] += frame1[j][k][2]

        # cv2.imwrite('v1_frame_'+str(i-10)+'.png', frame1)
        # cv2.imwrite('v2_frame_'+str(i-10)+'.png', frame2)
        # cv2.imwrite('v3_frame_'+str(i-10)+'.png', frame3)
        # cv2.imwrite('v4_frame_'+str(i-10)+'.png', frame4)
        # cv2.imwrite('v5_frame_'+str(i-10)+'.png', frame5)
        # cv2.imwrite('v6_frame_'+str(i-10)+'.png', frame6)
        # cv2.imwrite('v7_frame_'+str(i-10)+'.png', frame7)
        # cv2.imwrite('v8_frame_'+str(i-10)+'.png', frame8)
        # cv2.imwrite('v9_frame_'+str(i-10)+'.png', frame9)
        # cv2.imwrite('v10_frame_'+str(i-10)+'.png', frame10)
        # cv2.imwrite('v11_frame_'+str(i-10)+'.png', frame11)
        # cv2.imwrite('v12_frame_'+str(i-10)+'.png', frame12)
    numused+=1

for j in range (average1.shape[0]):
    for k in range (average1.shape[1]):
        average1[j][k][0] /= numused - 2
        average1[j][k][1] /= numused - 2
        average1[j][k][2] /= numused - 2
        if average1[j][k][0] > 255:
            average1[j][k][0] = 255
        if average1[j][k][1] > 255:
            average1[j][k][1] = 255
        if average1[j][k][2] > 255:
            average1[j][k][2] = 255

average1 = average1.astype(np.uint8)
cv2.imshow("average image", average1)
cv2.imwrite('v1_average_image.png', average1)
cv2.waitKey(0)

v1.release()
# v2.release()
# v3.release()
# v4.release()
# v5.release()
# v6.release()
# v7.release()
# v8.release()
# v9.release()
# v10.release()
# v11.release()
# v12.release()

cv2.destroyAllWindows()