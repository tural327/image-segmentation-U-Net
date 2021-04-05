import cv2
import numpy as np
import glob
import pickle

img_h = 128
img_w = 128
img_c = 3
lc = "stage1_train/*"
folder = sorted(glob.glob(lc))

X = np.zeros((len(folder), img_h, img_w, img_c), dtype=np.uint8)
Y = np.zeros((len(folder), img_h, img_w, 1), dtype=np.bool)
path = "stage1_train/"
for loc in folder:
    lc1 = loc.split("/")[1]
    file_mask = path + lc1 + "/masks/*.png"
    index = folder.index(loc)
    img = path + lc1 + "/images/" + lc1 + ".png"
    img_read = cv2.imread(img)
    img_size = cv2.resize(img_read, (img_h, img_w))
    img_array = np.array(img_size)
    X[index] = img_array
    mask = []
    for mask_ in sorted(glob.glob(file_mask)):
        img_mask = cv2.imread(mask_)
        img_mask1 = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        img_mask_size = cv2.resize(img_mask1, (img_h, img_w))
        mask_array = np.array(img_mask_size).reshape(img_h, img_w, 1)
        mask.append(mask_array)

    Y[index] = sum(mask)


pickle_out = open('X.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('Y.pickle','wb')
pickle.dump(Y, pickle_out)
pickle_out.close()

