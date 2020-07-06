from __future__ import print_function
import os
import tensorflow as tf
import cv2
import numpy as np
from Networks import experience_buffer
from Networks import create_LSTM
import robot_simulation as robot

# training environment parameters
TRAIN = True
ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1e4  # timesteps to observe before training
EXPLORE = 2e6  # frames over which to anneal epsilon
REPLAY_MEMORY = 1e3  # number of previous transitions to remember
BATCH = 4  # size of minibatch
h_size = 512  # size of hidden cells of LSTM
trace_length = 8  # memory length
FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate

reward_dir = "../results/"+"rnn_"+str(ACTIONS)
network_dir = "../saved_networks/" + "rnn_" + str(ACTIONS)
log_dir = "../log/" + "rnn_" + str(ACTIONS)
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)
if not os.path.exists(network_dir):
    os.makedirs(network_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_location_ave = reward_dir + "/average_reward_rnn_"+str(ACTIONS)+".csv"

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)


def padd_eps(eps_buff):
    if len(eps_buff) < trace_length:
        s = np.zeros([1, 84, 84, 1])
        a = np.zeros([ACTIONS])
        r = 0
        s1 = np.zeros([1, 84, 84, 1])
        d = True
        for i in range(0, trace_length - len(eps_buff)):
            eps_buff.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
    return eps_buff


def trainNetwork(s, readout, drop, tl, bs, si, rnn_state, sess):
    # define the cost function
    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.compat.v1.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cost)

    # open a test
    robot_explo = robot.Robot(0, TRAIN)

    # store the previous observations in replay memory
    myBuffer = experience_buffer(REPLAY_MEMORY)

    # setup training
    step_t = 0
    drop_rate = INITIAL_RATE
    total_reward = np.empty([0, 0])
    average_reward = np.empty([0, 0])
    state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # Reset the recurrent layer's hidden state

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # get the first state by doing nothing and preprocess the image to 84x84x1
    x_t = robot_explo.begin()
    x_t = cv2.resize(x_t, (84, 84))
    s_t = np.reshape(x_t, (1, 84, 84, 1))
    a_t_coll = []
    episodeBuffer = []

    while 1:
        # e-greedy scale down epsilon
        if drop_rate > FINAL_RATE and step_t > OBSERVE:
            drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE

        # choose an action by uncertainty
        readout_t, state1 = sess.run([readout, rnn_state],
                                     feed_dict={s: s_t, drop: drop_rate, tl: 1, bs: 1, si: state})
        readout_t = readout_t[0]
        readout_t[a_t_coll] = None
        a_t = np.zeros([ACTIONS])

        # print("Policy Action")
        action_index = np.nanargmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index = robot_explo.step(action_index)
        x_t1 = cv2.resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 84, 84, 1))
        s_t1 = x_t1
        finish = terminal

        # store the transition in D
        episodeBuffer.append(np.reshape(np.array([s_t, a_t, r_t, s_t1, terminal]), [1, 5]))

        if step_t > OBSERVE:
            # Reset the recurrent layer's hidden state
            state_train = (np.zeros([BATCH, h_size]), np.zeros([BATCH, h_size]))

            # sample a minibatch to train on
            trainBatch = myBuffer.sample(BATCH, trace_length)

            # get the batch variables
            s_j_batch = np.vstack(trainBatch[:, 0])
            a_batch = np.vstack(trainBatch[:, 1])
            r_batch = np.vstack(trainBatch[:, 2]).flatten()
            s_j1_batch = np.vstack(trainBatch[:, 3])

            readout_j1_batch = readout.eval(
                feed_dict={s: s_j1_batch, drop: 0, tl: trace_length, bs: BATCH, si: state_train})[0]
            end_multiplier = -(np.vstack(trainBatch[:, 4]).flatten() - 1)
            y_batch = r_batch + GAMMA * np.max(readout_j1_batch) * end_multiplier

            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch,
                drop: 0.8,
                tl: trace_length,
                bs: BATCH,
                si: state_train}
            )

        step_t += 1
        total_reward = np.append(total_reward, r_t)

        # save progress
        if step_t % 500000 == 0:
            # save neural networks
            saver.save(sess, network_dir, global_step=step_t)
            # save average reward data
            np.savetxt(file_location_ave, average_reward, delimiter=",")

        if step_t > 1e4:
            new_average_reward = np.average(total_reward[len(total_reward) - 10000:])
            average_reward = np.append(average_reward, new_average_reward)

        print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

        if step_t >= EXPLORE:
            break

        if finish:
            bufferArray = np.array(padd_eps(episodeBuffer))
            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            episodeBuffer = []

            if re_locate:
                x_t = robot_explo.rescuer()
            if complete:
                x_t = robot_explo.begin()
            x_t = cv2.resize(x_t, (84, 84))
            s_t = np.reshape(x_t, (84, 84, 1))
            a_t_coll = []
            continue

        # avoid collision next time
        if collision_index:
            a_t_coll.append(action_index)
            continue
        a_t_coll = []
        state = state1
        s_t = s_t1


def start_training():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    s, readout, drop, tl, bs, si, rnn_state = create_LSTM(ACTIONS, h_size)
    trainNetwork(s, readout, drop, tl, bs, si, rnn_state, sess)


if __name__ == "__main__":
    start_training()
