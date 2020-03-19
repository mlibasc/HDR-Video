import numpy as np 
import cv2

v1 = cv2.VideoCapture('v01.mp4')
v2 = cv2.VideoCapture('v02.mp4')
v3 = cv2.VideoCapture('v03.mp4')
v4 = cv2.VideoCapture('v04.mp4')
v5 = cv2.VideoCapture('v05.mp4')
v6 = cv2.VideoCapture('v06.mp4')
v7 = cv2.VideoCapture('v07.mp4')
v8 = cv2.VideoCapture('v08.mp4')
v9 = cv2.VideoCapture('v09.mp4')
v10 = cv2.VideoCapture('v10.mp4')
v11 = cv2.VideoCapture('v11.mp4')
v12 = cv2.VideoCapture('v12.mp4')

#counter variable and arrays for averaging each video
numused = 0
average1 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average2 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average3 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average4 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average5 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average6 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average7 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average8 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average9 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average10 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average11 = np.zeros((1080, 1920, 3), dtype=np.uint16)
average12 = np.zeros((1080, 1920, 3), dtype=np.uint16)

h = average1.shape[0]
w = average1.shape[1]
c = average1.shape[2]

#number of frames averaged
while(numused<4):

    ret1, frame1 = v1.read()
    ret2, frame2 = v2.read()
    ret3, frame3 = v3.read()
    ret4, frame4 = v4.read()
    ret5, frame5 = v5.read()
    ret6, frame6 = v6.read()
    ret7, frame7 = v7.read()
    ret8, frame8 = v8.read()
    ret9, frame9 = v9.read()
    ret10, frame10 = v10.read()
    ret11, frame11 = v11.read()
    ret12, frame12 = v12.read()

    if ret1 == False:
        break
    #skip first frame 
    if numused > 0: 
        #sum all frames
        for j in range (h):
            for k in range (w):
                for l in range(3):
                    average1[j][k][l] = np.add(average1[j][k][l], frame1[j][k][l])
                    average2[j][k][l] = np.add(average2[j][k][l], frame2[j][k][l])
                    average3[j][k][l] = np.add(average3[j][k][l], frame3[j][k][l])
                    average4[j][k][l] = np.add(average4[j][k][l], frame4[j][k][l])
                    average5[j][k][l] = np.add(average5[j][k][l], frame5[j][k][l])
                    average6[j][k][l] = np.add(average6[j][k][l], frame6[j][k][l])
                    average7[j][k][l] = np.add(average7[j][k][l], frame7[j][k][l])
                    average8[j][k][l] = np.add(average8[j][k][l], frame8[j][k][l])
                    average9[j][k][l] = np.add(average9[j][k][l], frame9[j][k][l]) 
                    average10[j][k][l] = np.add(average10[j][k][l], frame10[j][k][l])
                    average11[j][k][l] = np.add(average11[j][k][l], frame11[j][k][l])
                    average12[j][k][l] = np.add(average12[j][k][l], frame12[j][k][l])

    numused+=1

#average all frames
for j in range (average1.shape[0]):
    for k in range (average1.shape[1]):
        for l in range (3):
            average1[j][k][l] = np.divide(average1[j][k][l], (numused-1)) 
            average2[j][k][l] = np.divide(average2[j][k][l], (numused-1))
            average3[j][k][l] = np.divide(average3[j][k][l], (numused-1))
            average4[j][k][l] = np.divide(average4[j][k][l], (numused-1))
            average5[j][k][l] = np.divide(average5[j][k][l], (numused-1))
            average6[j][k][l] = np.divide(average6[j][k][l], (numused-1))
            average7[j][k][l] = np.divide(average7[j][k][l], (numused-1))
            average8[j][k][l] = np.divide(average8[j][k][l], (numused-1))
            average9[j][k][l] = np.divide(average9[j][k][l], (numused-1))
            average10[j][k][l] = np.divide(average10[j][k][l], (numused-1))
            average11[j][k][l] = np.divide(average11[j][k][l], (numused-1))
            average12[j][k][l] = np.divide(average12[j][k][l], (numused-1))

            #account for overflow
            if average1[j][k][l] > 255:
                average1[j][k][l] = 255
            if average2[j][k][l] > 255:
                average2[j][k][l] = 255
            if average3[j][k][l] > 255:
                average3[j][k][l] = 255
            if average4[j][k][l] > 255:
                average4[j][k][l] = 255
            if average5[j][k][l] > 255:
                average5[j][k][l] = 255
            if average6[j][k][l] > 255:
                average6[j][k][l] = 255
            if average7[j][k][l] > 255:
                average7[j][k][l] = 255
            if average8[j][k][l] > 255:
                average8[j][k][l] = 255
            if average9[j][k][l] > 255:
                average9[j][k][l] = 255
            if average10[j][k][l] > 255:
                average10[j][k][l] = 255
            if average11[j][k][l] > 255:
                average11[j][k][l] = 255
            if average12[j][k][l] > 255:
                average12[j][k][l] = 255

average1 = average1.astype(np.uint8)
average2 = average2.astype(np.uint8)
average3 = average3.astype(np.uint8)
average4 = average4.astype(np.uint8)
average5 = average5.astype(np.uint8)
average6 = average6.astype(np.uint8)
average7 = average7.astype(np.uint8)
average8 = average8.astype(np.uint8)
average9 = average9.astype(np.uint8)
average10 = average10.astype(np.uint8)
average11 = average11.astype(np.uint8)
average12 = average12.astype(np.uint8)

cv2.imshow("v1 image", average1)
cv2.imshow("v2 image", average2)
cv2.imshow("v3 image", average3)
cv2.imshow("v4 image", average4)
cv2.imshow("v5 image", average5)
cv2.imshow("v6 image", average6)
cv2.imshow("v7 image", average7)
cv2.imshow("v8 image", average8)
cv2.imshow("v9 image", average9)
cv2.imshow("v10 image", average10)
cv2.imshow("v11 image", average11)
cv2.imshow("v12 image", average12)

cv2.imwrite('v1_average_image.png', average1)
cv2.imwrite('v2_average_image.png', average2)
cv2.imwrite('v3_average_image.png', average3)
cv2.imwrite('v4_average_image.png', average4)
cv2.imwrite('v5_average_image.png', average5)
cv2.imwrite('v6_average_image.png', average6)
cv2.imwrite('v7_average_image.png', average7)
cv2.imwrite('v8_average_image.png', average8)
cv2.imwrite('v9_average_image.png', average9)
cv2.imwrite('v10_average_image.png', average10)
cv2.imwrite('v11_average_image.png', average11)
cv2.imwrite('v12_average_image.png', average12)

cv2.waitKey(0)

v1.release()
v2.release()
v3.release()
v4.release()
v5.release()
v6.release()
v7.release()
v8.release()
v9.release()
v10.release()
v11.release()
v12.release()