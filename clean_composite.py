import numpy as np
import cv2
import math

numImages = 12
width = 1920
height = 1080
gamma = 0.69
widthtrap = 256/4
k = 2
epsilon = 0.001

shutters = [1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4]

newimage = []
finalImage = np.zeros((height, width, 3)).astype(np.float32)

imageNames = []
certainty = np.zeros(256)

#Fill certainty
first = int(256 / 2 - widthtrap / 2)
second = int(256 / 2  + widthtrap / 2)

for i in range(first):
    certainty[i] = i
for i in range(second - first):
    certainty[i + first] = first
for i in range(256 - second):
    certainty[second + i] = 256 - second - i
certainty[0] = epsilon

print(certainty)

#for i in range(len(certainty)):
#    if i == 0:
#        certainty[i] = epsilon
#    else:
#        certainty[i] = gamma * i ** ((gamma - 1) / gamma)



def compute_hdr(data):
    total = 0
    divisor = 0
    for i in range(len(data)):
        total += np.multiply(certainty[data[i]], np.power(data[i], np.divide(1,gamma)))
        divisor += certainty[data[i]]

    return np.power(np.divide(total, divisor), gamma)


for i in range(numImages):
    imageNames.append("v" + str(i+1) + "_averaged.png")

for i in range(numImages):
    temp = cv2.imread(imageNames[i])
    temp = cv2.resize(temp, (width, height))
    newimage.append(temp)
newimage = np.array(newimage)


for i in range(height):
    for j in range(width):
        for k in range(3):

            # get list of pixels
            pix = []
            for n in range(numImages):
                pix.append(newimage[n][i][j][k])
            pix = np.array(pix)
            pix.astype(np.float32)

            # now i have a list of pixels
            finalImage[i][j][k] = compute_hdr(pix)
            if(finalImage[i][j][k] > 255):
                finalImage[i][j][k] = 255


maxval = np.amax(finalImage)
print(np.amax(finalImage))
finalImage = np.multiply(finalImage, 255)
finalImage = np.divide(finalImage, maxval)
print(np.amax(finalImage))
finalImage = finalImage.astype(np.uint8)
cv2.imwrite('hdr_' + str(width) + '_' + str(gamma) + '_'+str(height)+'.png', finalImage)
#cv2.imshow('new image', finalImage)
#cv2.waitKey(0)
