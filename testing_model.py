import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import numpy as np
import glob
import pickle

model = load_model('model.h5')
lc = "stage2_test_final/*"
folder = sorted(glob.glob(lc))
try:
    X_test = pickle.load(open("X_test.pickle", 'rb'))

except:
    X_test = np.zeros((len(folder), 128,128,3),dtype=np.uint8)
    lc = "stage2_test_final/*"
    folder = sorted(glob.glob(lc))
    path = "stage2_test_final/"
    for loc in folder:
        lc1 = loc.split("/")[1]
        file_mask = path + lc1+"/masks/*.png"
        index = folder.index(loc)
        img = path + lc1 + "/images/" + lc1 + ".png"
        img_read = cv2.imread(img)
        img_size = cv2.resize(img_read,(128,128))
        img_array = np.array(img_size)
        X_test[index] = img_array

    pickle_out = open('X_test.pickle', 'wb')
    pickle.dump(X_test, pickle_out)
    pickle_out.close()


a = random.randint(0,len(folder))
b = random.randint(0,len(folder))

y_pred_a = model.predict(np.array(X_test[a]).reshape(1,128,128,3))
y_pred_b = model.predict(np.array(X_test[b]).reshape(1,128,128,3))



plt.subplot(2, 2, 1)
plt.imshow(X_test[a])
plt.title("Testing image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(y_pred_a[0])
plt.title("predicted mask")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(X_test[b])
plt.title("Testing image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(y_pred_b[0])
plt.title("predicted mask")
plt.axis("off")

plt.show()
