import PIL
import numpy as np
import glob
import tensorflow as tf
from sklearn.preprocessing import scale

image1 = glob.glob('./images_dir/hamburger/*.jpg')
image2 = glob.glob('./images_dir/steak/*.jpg')
image3 = glob.glob('./images_dir/risotto/*.jpg')
image4 = glob.glob('./images_dir/chicken/*.jpg')
image5 = glob.glob('./images_dir/sushi/*.jpg')
image6 = glob.glob('./images_dir/icecream/*.jpg')
image7 = glob.glob('./images_dir/spaghetti/*.jpg')
image8 = glob.glob('./images_dir/ramen/*.jpg')
image9 = glob.glob('./images_dir/tiramisu/*.jpg')
image10 = glob.glob('./images_dir/pizza/*.jpg')
images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]

data = np.empty((0, 128, 128, 3))
# labels = [i] * 1000 for i in range(3)

for i in range(1000):
    img = PIL.Image.open(image1[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "1" + str(i) + "th working"
for i in range(1001):
    if i == 415:
        continue
    img = PIL.Image.open(image2[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "2" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image3[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "3" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image4[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "4" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image5[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "5" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image6[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "6" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image7[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "7" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image8[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "8" + str(i) + "th working"

for i in range(1000):
    img = PIL.Image.open(image9[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "9" + str(i) + "th working"
for i in range(1000):
    img = PIL.Image.open(image10[i])
    arr = np.array(img.resize((128, 128)))
    arr = arr.flatten()
    arr = scale(arr)
    arr = np.reshape(arr, (128, 128, 3))
    data = np.append(data, [arr], axis=0)
    print "10" + str(i) + "th working"

data.tofile('test.dat')
