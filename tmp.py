import tensorflow as tf
a = tf.constant([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
b = tf.reshape(a, [2*2, 2])
c = tf.reshape(b, [-1, 2*2])
with tf.Session() as sess:
    print sess.run(c)
