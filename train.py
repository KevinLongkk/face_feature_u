import python35.face_feature.get_data as get_data
import python35.face_feature.model as model
import python35.face_feature.model2 as model2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


_TRIAN_STEP = 300000
def train():
    data_X, data_Y = get_data.get_data()
    print(data_Y.shape)
    X, Y, y = model2.model(tf.contrib.layers.l2_regularizer(0.0001))
    data_Y = np.reshape(data_Y, [-1, 8])
    # print(data_Y[0:4][:])

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.reduce_mean(tf.square(Y-y)) + tf.add_n(tf.get_collection('losses'))
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Y)
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step)
    average_variable = tf.train.ExponentialMovingAverage(0.99, global_step)
    average_variable_op = average_variable.apply(tf.trainable_variables())

    saver = tf.train.Saver()
    save_path = r'./checkpoint_64w_average_op/'
    with tf.Session() as sess:
        try:
            last_cke_path = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, last_cke_path)
        except:
            print("Failed to restore checkpoint. Initializing variables instead.")
            sess.run(tf.initialize_all_variables())
        for i in range(_TRIAN_STEP):
            randix = np.random.randint(len(data_X), size=24)
            xs = data_X[randix]
            ys = data_Y[randix]
            _, _, losss = sess.run([loss, average_variable_op, cross_entropy], feed_dict={X: xs, Y: ys})
            if i % 100 == 0:
                print(losss)
                print(sess.run([global_step, y], feed_dict={X: [data_X[0]]}))
                saver.save(sess, save_path=save_path, global_step=global_step)
                print('Save checkpoint')


if __name__ == '__main__':
    train()
# plt.imshow(np.reshape(_X[51], [224, 224, 3]))
# c = _coordinate[408:416]
# plt.plot(c[0], c[1], 'r*')
# plt.plot(c[2], c[3], 'r*')
# plt.plot(c[4], c[5], 'r*')
# plt.plot(c[6], c[7], 'r*')
# plt.show()
