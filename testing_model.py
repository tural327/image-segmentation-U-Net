import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob
import cv2
import random

test_lc = "stage2_test_final/*"
folder = sorted(glob.glob(test_lc))
model = tf.keras.models.load_model('model.h5')

try:
    X_test = pickle.load(open("X_test.pickle", 'rb'))

except:
    test_lc = "stage2_test_final/*"
    folder = sorted(glob.glob(test_lc))
    X_test = np.zeros((len(folder), 128,128,3),dtype=np.uint8)
    path = "stage2_test_final/"
    for loc in folder:
        lc1 = loc.split("/")[1]
        index = folder.index(loc)
        img = path + lc1 + "/images/" + lc1 + ".png"
        img_read = cv2.imread(img)
        img_size = cv2.resize(img_read,(128,128))
        img_array = np.array(img_size)
        X_test[index] = img_array

    pickle_out = open('X_test.pickle','wb')
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

a = random.randint(0,len(folder))
b = random.randint(0,len(folder))

x_test_for_prediction_a = np.array(X_test[a]).reshape(1,128,128,3)
y_pred_a = model.predict(x_test_for_prediction_a)
x_test_for_prediction_b = np.array(X_test[b]).reshape(1,128,128,3)
y_pred_b = model.predict(x_test_for_prediction_b)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(X_test[a])
ax1.get_xaxis().set_visible(False)
ax1.title.set_text('Testing')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(y_pred_a[0])
ax2.get_xaxis().set_visible(False)
ax2.title.set_text('Predicted Mask')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(X_test[b])
ax3.title.set_text('Testing')
ax3.get_xaxis().set_visible(False)
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(y_pred_b[0])
ax4.title.set_text('Predicted Mask')
ax4.get_xaxis().set_visible(False)

plt.axis('off')
plt.show()


print(a)

