import tensorflow as tf

a = tf.constant([[2,2,2],[2,2,2]])
b = tf.expand_dims(a, 0)
with tf.Session() as sess:
    print sess.run(b.shape)
