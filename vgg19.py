import tensorflow as tf


class VGG:
    # We will create a very simple class implement VGG model using tensorflow api.
    # The model architecture is described in great detail in following paper : https://arxiv.org/pdf/1409.1556.pdf

    # Initially, we create a class with 2 hyper-parameter, learning rate & batch size
    def __init__(self, learning_rate, minibatch_size):
        self.lr = learning_rate
        self.minibatch = minibatch_size

    # You will initialize weights/filters for 2 type of layer : fully connected(dense) layer and convolution layer.
    # Parameters are generated in a truncated normal distribution, this is the recommended initializer for layer's
    # weights.
    def init_dense(self, n_in, n_out, name):
        W = tf.get_variable(name=name + "_W", shape=[n_in, n_out],
                            initializer=tf.initializers.truncated_normal)

        b = tf.get_variable(name=name + "_b", shape=[n_out],
                            initializer=tf.initializers.truncated_normal)

        return W, b

    def init_conv2d(self, n_f, n_c_in, n_c_out, name):
        W = tf.get_variable(name=name + "_W", shape=[n_f, n_f, n_c_in, n_c_out],
                            initializer=tf.initializers.truncated_normal)

        b = tf.get_variable(name=name + "_b", shape=[n_c_out],
                            initializer=tf.initializers.truncated_normal)

        return W, b

    # Each CONV2D block would be a CONV2D layer with relu activation
    def conv2d(self, bottom, n_c_in, n_c_out, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.init_conv2d(3, n_c_in, n_c_out, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            b = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(b)
            return relu

    def dense(self, bottom, n_in, n_out, name):
        with tf.variable_scope(name):
            W, b = self.init_dense(n_in, n_out, name)
            x = tf.reshape(bottom, [-1, n_in])
            fc = tf.nn.bias_add(tf.matmul(x, W), b)
            return fc

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # Implement the below model function to implement VGG19 model:
    def model(self, images):
        conv1_1 = self.conv2d(images, 3, 64, "conv2d1_1")
        conv1_2 = self.conv2d(conv1_1, 64, 64, "conv2d1_2")
        pool1 = self.max_pool(conv1_2, 'pooling1')

        conv2_1 = self.conv2d(pool1, 64, 128, "conv2d2_1")
        conv2_2 = self.conv2d(conv2_1, 128, 128, "conv2d2_2")
        pool2 = self.max_pool(conv2_2, 'pooling2')

        conv3_1 = self.conv2d(pool2, 128, 256, "conv2d3_1")
        conv3_2 = self.conv2d(conv3_1, 256, 256, "conv2d3_2")
        conv3_3 = self.conv2d(conv3_2, 256, 256, "conv2d3_3")
        conv3_4 = self.conv2d(conv3_3, 256, 256, "conv2d3_4")
        pool3 = self.max_pool(conv3_4, 'pooling3')

        conv4_1 = self.conv2d(pool3, 256, 512, "conv2d4_1")
        conv4_2 = self.conv2d(conv4_1, 512, 512, "conv2d4_2")
        conv4_3 = self.conv2d(conv4_2, 512, 512, "conv2d4_3")
        conv4_4 = self.conv2d(conv4_3, 512, 512, "conv2d4_4")
        pool4 = self.max_pool(conv4_4, 'pooling4')

        conv5_1 = self.conv2d(pool4, 512, 512, "conv2d5_1")
        conv5_2 = self.conv2d(conv5_1, 512, 512, "conv2d5_2")
        conv5_3 = self.conv2d(conv5_2, 512, 512, "conv2d5_3")
        conv5_4 = self.conv2d(conv5_3, 512, 512, "conv2d5_4")
        pool5 = self.max_pool(conv5_4, 'pooling5')

        fc6 = self.dense(pool5, 25088, 4096, "dense6")
        relu6 = tf.nn.relu(fc6)

        fc7 = self.dense(relu6, 4096, 4096, "dense7")
        relu7 = tf.nn.relu(fc7)

        fc8 = self.dense(relu7, 4096, 1000, "dense8")

        logits = tf.nn.softmax(fc8, name="logits")
        return logits

    # tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels): computes the softmax entropy loss.
    # tf.reduce_mean: computes the mean of elements across dimensions of a tensor. Use this to sum the losses over all
    # the examples to get the overall cost.
    def compute_loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss

    # TODO : Implement and evaluate training step
    # You will need to create a placeholder for the input data.
    # Then compute the cost (as cost) of above models with that inputs.
    # Then define the optimizer using :
    #           optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
    # Then create a tf session to compute such graph
    def train(self, x_train, y_train, x_test, y_test, epochs):
        pass
