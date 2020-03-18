import os, PIL
import numpy as np 
from PIL import Image

allfiles = os.listdir(os.getcwd())
imlist = [filename for filename in allfiles if filename[-4:] in [".png",".PMG"]]

w = 1920
h = 1080
N = len(imlist)

arr = np.zeros((h,w,3), np.float)

for im in imlist:
    imarr = np.array(Image.open(im), dtype=np.float)
    arr = arr + imarr/N

arr = np.array(np.round(arr), dtype=np.uint8)

out = Image.fromarray(arr, mode="RBG")
out.save("v1_Average.png")
out.show()
