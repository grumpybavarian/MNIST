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

def conv_relu_pool(layer_input, num_filters, keep_prob):
    layer = tf.layers.conv2d(inputs=layer_input,
                             filters=num_filters,
                             kernel_size=[5, 5],
                             padding='SAME',
                             activation=tf.nn.relu)
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2)
    return layer

def dense_layer(layer_input, num_neurons, keep_prob):
    layer = tf.layers.dense(inputs=layer_input,
                            units=num_neurons,
                            activation=tf.nn.relu,
                            use_bias=True)
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    return layer


def train(num_conv_layers=2, num_dense_layers=2):
    input = tf.placeholder(tf.float32, shape=[None, 784])
    network = tf.reshape(input, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

    for i in range(num_conv_layers):
        network = conv_relu_pool(network, 64, keep_prob)

    network = tf.flatten(network)

    for i in range(num_dense_layers):
        network = dense_layer(network, 1024, keep_prob)

    y = network
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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


