import tensorflow as tf

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    """
    creates weight variable
    :param shape: 
    :return: 
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    creates bias variable
    :param shape: 
    :return: 
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def train():
    input = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.reshape(input, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    layer1 = conv2d(x, W_conv1) + b_conv1
    layer1 = tf.nn.relu(layer1)
    layer1 = max_pool_2x2(layer1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    layer2 = conv2d(layer1, W_conv2) + b_conv2
    layer2 = tf.nn.relu(layer2)
    layer2 = max_pool_2x2(layer2)
    layer2 = tf.reshape(layer2, [-1, 7*7*64])

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    fc1_layer = tf.nn.relu(tf.matmul(layer2, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    fc_layer = tf.nn.dropout(fc1_layer, keep_prob=keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y = tf.matmul(fc_layer, W_fc2) + b_fc2
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            batch = mnist.train.next_batch(100)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={input: batch[0], y_: batch[1], keep_prob: 0.5})
                print("Step {}: Accuracy: {}".format(i, train_accuracy))
            train_step.run(feed_dict={input: batch[0], y_: batch[1], keep_prob: 0.5})

        test_accuracy = accuracy.eval(feed_dict={input: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("Test Accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    train()


"""
conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  """