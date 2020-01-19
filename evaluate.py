import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale
from tensorflow.contrib.layers import flatten
import glob
import sys
import PIL
################defining layers####################
labelsFullPath = '/tmp/output_labels.txt'


def network(x, keep_prob):

        # Layer 1: Convolutional. Input = 128x128x3. Output = 128 128 64.
    conv1_W = tf.get_variable("conv1_W", shape=(3, 3, 3, 64), initializer=tf.contrib.layers.xavier_initializer())

    conv1_b = tf.Variable(tf.zeros(64), name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # Pooling. Input = 128 128 64 Output = 64 64 64.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 64 64 128.
    conv2_W = tf.get_variable("Conv2_W", shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
    conv2_b = tf.Variable(tf.zeros(128))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # Pooling. Input = 64 64 128. Output = 32 32 128.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Output = 32 32 256.
    conv3_W = tf.get_variable("Conv3_W", shape=(3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
    conv3_b = tf.Variable(tf.zeros(256))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b

    # Activation.
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # Pooling. Input = 32 32 256. Output = 16 16 256.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 4: Convolutional. Input 16 16 256 Output = 16 16 512.
    conv4_W = tf.get_variable("Conv4_W", shape=(3, 3, 256, 512), initializer=tf.contrib.layers.xavier_initializer())
    conv4_b = tf.Variable(tf.zeros(512))
    conv4 = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b

    # Activation.
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.dropout(conv4, keep_prob)
    # Pooling. Input = 16 16 512. Output = 8 8 512.
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 5: Convolutional. Input 8 8 512 Output = 8 8 1024.
    conv5_W = tf.get_variable("Convn5_W", shape=(3, 3, 512, 1024), initializer=tf.contrib.layers.xavier_initializer())
    conv5_b = tf.Variable(tf.zeros(1024))
    conv5 = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b

    # Activation.
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.dropout(conv5, keep_prob)
    # Pooling. Input = 8 8 1024. Output = 4 4 1024.
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Layer 6: Convolutional. Output = 4 4 2048.

    conv6_W = tf.get_variable("Conv6_W", shape=(3, 3, 1024, 2048), initializer=tf.contrib.layers.xavier_initializer())
    conv6_b = tf.Variable(tf.zeros(2048))
    conv6 = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b

    # Activation.
    conv6 = tf.nn.relu(conv6)
    conv6 = tf.nn.dropout(conv6, keep_prob)
    # Pooling. Input = 4 4 2048. Output = 2 2 2048
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv7_W = tf.get_variable("Conv7_W", shape=(3, 3, 2048, 2048), initializer=tf.contrib.layers.xavier_initializer())
    conv7_b = tf.Variable(tf.zeros(2048))
    conv7 = tf.nn.conv2d(conv6, conv7_W, strides=[1, 1, 1, 1], padding='SAME') + conv7_b

    # Activation.
    conv7 = tf.nn.relu(conv7)
    conv7 = tf.nn.dropout(conv7, keep_prob)
    # Pooling. Input = 2 2 2048. Output = 1 1 2048
    conv7 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0 = flatten(conv7)

# Layer 6: Fully Connected. Input = 8192. Output = 1024.
    fc1_W = tf.get_variable("Fc1_W", shape=(2048, 1024), initializer=tf.contrib.layers.xavier_initializer())
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # Layer 4: Fully Connected. Input = 1024 Output = 512.
    fc2_W = tf.get_variable("Fc2_W", shape=(1024, 512), initializer=tf.contrib.layers.xavier_initializer())
    fc2_b = tf.Variable(tf.zeros(512))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    # Layer 5: Fully Connected. Input = 512. Output = 10.
    fc3_W = tf.get_variable("Fc3_W", shape=(512, 10), initializer=tf.contrib.layers.xavier_initializer())
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


BATCH_NUM = 5
ROW_SIZE = 100
COL_SIZE = 100

data = np.fromfile('test.dat', dtype=int)


data = np.reshape(data, (10000, 128, 128, 3))
labels = np.array([])

for i in range(10 * 1000):
    labels = np.append(labels, [i / 1000])
data_size = len(data)
split_size = int(0.7 * data_size)

X_train, y_train = data[:split_size], labels[:split_size]
X_test, y_test = data[split_size:], labels[split_size:]

assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

EPOCHS = 100
BATCH_SIZE = 100

x = tf.placeholder(tf.float32, (None, 128, 128, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
keep_prob = tf.placeholder(tf.float32)

rate = 0.0001


logits = network(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tmpsaver = tf.train.Saver()


def run_inference_on_image(imagePath):
    answer = None

    with tf.Session() as sess:

        tmpsaver.restore(sess, './tmpsave')
        image = glob.glob(imagePath)
        img = PIL.Image.open(image[0])
        arr = np.array(img.resize((128, 128)))
        arr = arr.flatten()
        arr = scale(arr)
        arr = np.reshape(arr, (128, 128, 3))
        predictions = sess.run(logits,
                               {x: [arr], y: np.array([1]), keep_prob: 1.0})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer


if __name__ == '__main__':
    imagePath = sys.argv[1]
    run_inference_on_image(imagePath)
