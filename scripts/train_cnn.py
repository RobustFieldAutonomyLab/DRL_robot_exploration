from __future__ import print_function
import tensorflow as tf
import os
import cv2
import random
import numpy as np
from collections import deque
from Networks import create_CNN
import robot_simulation as robot

# training environment parameters
TRAIN = True
ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 10000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
REPLAY_MEMORY = 10000  # number of previous transitions to remember
BATCH = 64  # size of minibatch

# exploration and exploitation trad-off parameters
# e-greedy
FINAL_EPSILON = 0  # final value of epsilon
INITIAL_EPSILON = 0.9
reward_dir = "../results/"+"cnn_"+str(ACTIONS)
netword_dir = "../saved_networks/"+"cnn_"+str(ACTIONS)
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)
if not os.path.exists(netword_dir):
    os.makedirs(netword_dir)
file_location_ave = reward_dir + "/average_reward_cnn_"+str(ACTIONS)+".csv"


def trainNetwork(s, readout, keep_per, sess):
    # define the cost function
    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cost)

    # open a test
    robot_explo = robot.Robot(0, TRAIN)

    # store the previous observations in replay memory
    D = deque()

    # setup training
    step_t = 0
    epsilon = INITIAL_EPSILON
    total_reward = np.empty([0, 0])
    average_reward = np.empty([0, 0])
    reward_std = np.empty([0, 0])

    # saving and loading networks
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(netword_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t = robot_explo.begin()
    x_t = cv2.resize(x_t, (84, 84))
    s_t = np.reshape(x_t, (84, 84, 1))
    a_t_coll = []

    while 1:
        # e-greedy scale down epsilon
        if epsilon > FINAL_EPSILON and step_t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t], keep_per: 0.2})[0]
        readout_t[a_t_coll] = None
        a_t = np.zeros([ACTIONS])

        # print("Policy Action")
        action_index = np.nanargmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index = robot_explo.step(action_index)
        x_t1 = cv2.resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (84, 84, 1))
        s_t1 = x_t1
        finish = terminal
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if step_t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch, keep_per: 0.2})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch,
                keep_per: 0.2}
            )

        step_t += 1
        total_reward = np.append(total_reward, r_t)

        # save progress every 500000 iterations
        if step_t % 500000 == 0:
            saver.save(sess, netword_dir, global_step=step_t)

        if step_t > 10000:
            new_average_reward = np.average(total_reward[len(total_reward) - 10000:])
            average_reward = np.append(average_reward, new_average_reward)
            # plotting
            # if step_t % 1000 == 0:
            #     xaxis = step_t - 10000
            #     yaxis = new_average_reward
                # ss.write(dict(x=xaxis, y=yaxis))

        print("TIMESTEP", step_t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

        if step_t % 10000 == 0:
            np.savetxt(file_location_ave, average_reward, delimiter=",")

        if step_t >= EXPLORE:
            break

        if finish:
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
        s_t = s_t1


def start_training():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    s, readout, keep_per = create_CNN(ACTIONS)
    trainNetwork(s, readout, keep_per, sess)


if __name__ == "__main__":
    start_training()
