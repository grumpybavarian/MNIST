import tensorflow as tf

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train():
    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # sess.run(init)

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, batch_accuracy, cross_entropy_ = sess.run([train_step, accuracy, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

        print(batch_accuracy)

    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Test Accuracy: {}".format(test_accuracy))

if __name__ == '__main__':
    train()