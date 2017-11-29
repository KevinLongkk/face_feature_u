import tensorflow as tf
import get_data
import model
import model2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


X, Y, y = model2.model(tf.contrib.layers.l2_regularizer(0.0001))
cameraCapture = cv2.VideoCapture(0)
saver = tf.train.Saver()
save_path = r'./checkpoint_64w_average_op/'
def find_face(image):
    c = sess.run(y, feed_dict={X: image})
    c = c[0][:]
    a1_x = int(c[0] - 10)
    a1_y = int(c[1] - 10)
    a2_x = int(c[2] + 10)
    a2_y = a1_y
    a3_x = a2_x
    a3_y = int(c[7] + 10)
    a4_x = a1_x
    a4_y = a3_y
    cv2.line(image_show, (a1_x * 2, a1_y * 2),
             (a2_x * 2, a2_y * 2), (255, 155, 155), 5)
    cv2.line(image_show, (a2_x * 2, a2_y * 2),
             (a3_x * 2, a3_y * 2), (255, 155, 155), 5)
    cv2.line(image_show, (a3_x * 2, a3_y * 2),
             (a4_x * 2, a4_y * 2), (255, 155, 155), 5)
    cv2.line(image_show, (a4_x * 2, a4_y * 2),
             (a1_x * 2, a1_y * 2), (255, 155, 155), 5)
    # cv2.circle(image_show, (int(c[0]*2), int(c[1]*2)), radius=5, color=(100, 200, 100))
    # cv2.circle(image_show, (int(c[2]*2), int(c[3]*2)), radius=5, color=(100, 200, 100))
    # cv2.circle(image_show, (int(c[4]*2), int(c[5]*2)), radius=5, color=(100, 200, 100))
    # cv2.circle(image_show, (int(c[6]*2), int(c[7]*2)), radius=5, color=(100, 200, 100))
with tf.Session() as sess:
    try:
        last_chk_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, last_chk_path)
    except:
        print("Failed to restore checkpoint. Exit.")
        exit()
    success, frame = cameraCapture.read()
    i = 0
    while success and cv2.waitKey(1) == -1:
        image_show = cv2.resize(frame, dsize=(448, 448))
        k = cv2.resize(image_show, dsize=(224, 224))
        image = np.resize(k, [1, 224*224*3])
        find_face(image)
        cv2.imshow(' ', image_show)
        success, frame = cameraCapture.read()

    # plt.imshow(np.reshape(image, [224, 224, 3]))
    # plt.plot(c[0], c[1], 'r*')
    # plt.plot(c[2], c[3], 'r*')
    # plt.plot(c[4], c[5], 'r*')
    # plt.plot(c[6], c[7], 'r*')
    # a1_x = c[0] - 10
    # a1_y = c[1] - 10
    # a2_x = c[2] + 10
    # a2_y = a1_y
    # a3_x = a2_x
    # a3_y = c[7] + 10
    # a4_x = a1_x
    # a4_y = a3_y
    # plt.plot([a1_x, a2_x, a3_x, a4_x, a1_x], [a1_y, a2_y, a3_y, a4_y, a1_y], color='green')
    # plt.show()

