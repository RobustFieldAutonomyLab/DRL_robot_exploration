import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def weight_variable(shape):
    initial = tf.compat.v1.random.truncated_normal(shape, stddev=0.01)
    return tf.compat.v1.Variable(initial)


def bias_variable(shape):
    initial = tf.compat.v1.constant(0.01, shape=shape)
    return tf.compat.v1.Variable(initial)


def conv2d(x, W, stride):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")


def create_CNN(num_action):
    # network weights
    W_conv1 = weight_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    W_fc1 = weight_variable([7 * 7 * 64, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, num_action])
    b_fc2 = bias_variable([num_action])

    # input layer
    s = tf.compat.v1.placeholder("float", [None, 84, 84, 1])

    # hidden layers
    h_conv1 = tf.compat.v1.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_conv2 = tf.compat.v1.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.compat.v1.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.compat.v1.layers.flatten(h_conv3)

    h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    keep_rate = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)
    hidden = tf.compat.v1.nn.dropout(h_fc1, keep_rate)

    # readout layer
    readout = tf.matmul(hidden, W_fc2) + b_fc2

    return s, readout, keep_rate


def create_LSTM(num_action, num_cell, scope):
    # network weights
    W_conv1 = weight_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    W_conv4 = weight_variable([7, 7, 64, 512])
    b_conv4 = bias_variable([512])
    W_fc2 = weight_variable([512, num_action])
    b_fc2 = bias_variable([num_action])

    # training setup
    trainLength = tf.compat.v1.placeholder(shape=None, dtype=tf.int32)

    # input layer
    s = tf.compat.v1.placeholder("float", [None, 84, 84, 1])
    batch_size = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])

    # hidden layers
    h_conv1 = tf.compat.v1.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_conv2 = tf.compat.v1.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.compat.v1.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv4 = tf.compat.v1.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

    # define rnn layer
    rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
        num_units=num_cell, state_is_tuple=True)
    convFlat = tf.reshape(tf.compat.v1.layers.flatten(
        h_conv4), [batch_size, trainLength, num_cell])
    state_in = rnn_cell.zero_state(batch_size, tf.float32)
    rnn, rnn_state = tf.compat.v1.nn.dynamic_rnn(
        inputs=convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=state_in, scope=scope)
    rnn = tf.reshape(rnn, shape=[-1, num_cell])

    keep_rate = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)
    hidden = tf.compat.v1.nn.dropout(rnn, keep_rate)

    # readout layer
    readout = tf.matmul(hidden, W_fc2) + b_fc2

    return s, readout, keep_rate, trainLength, batch_size, state_in, rnn_state
