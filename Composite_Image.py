import numpy as np
import cv2

newimage = np.zeros((1080,1920, 3), dtype=np.uint16)
image1_ = cv2.imread('v1_average_image.png')
image2_ = cv2.imread('v2_average_image.png')
image3_ = cv2.imread('v3_average_image.png')
image4_ = cv2.imread('v4_average_image.png')
image5_ = cv2.imread('v5_average_image.png')
image6_ = cv2.imread('v6_average_image.png')
image7_ = cv2.imread('v7_average_image.png')
image8_ = cv2.imread('v8_average_image.png')
image9_ = cv2.imread('v9_average_image.png')
image10_ = cv2.imread('v10_average_image.png')
image11_ = cv2.imread('v11_average_image.png')
image12_ = cv2.imread('v12_average_image.png')

#resize the images
scale_percent = 50
width = int(image1_.shape[1] * scale_percent / 100)
height = int(image1_.shape[0] * scale_percent / 100)
dim = (width, height)
image1 = cv2.resize(image1_, dim, interpolation=cv2.INTER_AREA)
image2 = cv2.resize(image2_, dim, interpolation=cv2.INTER_AREA)
image3 = cv2.resize(image3_, dim, interpolation=cv2.INTER_AREA)
image4 = cv2.resize(image4_, dim, interpolation=cv2.INTER_AREA)
image5 = cv2.resize(image5_, dim, interpolation=cv2.INTER_AREA)
image6 = cv2.resize(image6_, dim, interpolation=cv2.INTER_AREA)
image7 = cv2.resize(image7_, dim, interpolation=cv2.INTER_AREA)
image8 = cv2.resize(image8_, dim, interpolation=cv2.INTER_AREA)
image9 = cv2.resize(image9_, dim, interpolation=cv2.INTER_AREA)
image10 = cv2.resize(image10_, dim, interpolation=cv2.INTER_AREA)
image11 = cv2.resize(image11_, dim, interpolation=cv2.INTER_AREA)
image12 = cv2.resize(image12_, dim, interpolation=cv2.INTER_AREA)
newimage = cv2.resize(newimage, dim, interpolation=cv2.INTER_AREA)

#summing RGB values separately
for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
        for k in range(image1.shape[2]):
            newimage[i][j][k] = np.add(newimage[i][j][k], image1[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image2[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image3[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image4[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image5[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image6[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image7[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image8[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image9[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image10[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image11[i][j][k])
            newimage[i][j][k] = np.add(newimage[i][j][k], image12[i][j][k])
#averaging 
for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
        for k in range(image1.shape[2]):
            newimage[i][j][k] = np.divide(newimage[i][j][k], 12)
            if newimage[i][j][k] > 255:
                newimage[i][j][k] = 255

newimage = newimage.astype(np.uint8)
cv2.imshow('new image', newimage) 
cv2.imwrite('newimage.png', newimage)
cv2.waitKey(0)