import tensorflow as tf
import get_data
import model
import model2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


data_X, data_Y = get_data.get_data(predict=True)
X, Y, y = model2.model(tf.contrib.layers.l2_regularizer(0.0001))
data_Y = np.reshape(data_Y, [-1, 8])
saver = tf.train.Saver()
save_path = r'./checkpoint_64w_average_op/'
with tf.Session() as sess:
    try:
        last_chk_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, last_chk_path)
    except:
        print("Failed to restore checkpoint. Exit.")
        exit()
    image = Image.open(r'./dataset/data_iamge/pic04983.jpg')
    # image = Image.open(r'./test/huge02.jpg')
    image = np.array(image)
    image = image.reshape(1, 224*224*3)
    c = sess.run(y, feed_dict={X: image})
    c = c[0][:]
    plt.imshow(np.reshape(image, [224, 224, 3]))
    plt.plot(c[0], c[1], 'r*')
    plt.plot(c[2], c[3], 'r*')
    plt.plot(c[4], c[5], 'r*')
    plt.plot(c[6], c[7], 'r*')
    a1_x = c[0] - 10
    a1_y = c[1] - 10
    a2_x = c[2] + 10
    a2_y = a1_y
    a3_x = a2_x
    a3_y = c[7] + 10
    a4_x = a1_x
    a4_y = a3_y
    plt.plot([a1_x, a2_x, a3_x, a4_x, a1_x], [a1_y, a2_y, a3_y, a4_y, a1_y], color='green')
    plt.show()
