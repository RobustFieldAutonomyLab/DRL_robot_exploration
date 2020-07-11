import os
import tensorflow as tf
from skimage.transform import resize
import random
import numpy as np
from tf_networks import create_LSTM
from tensorboardX import SummaryWriter
import robot_simulation as robot

# training environment parameters
TRAIN = True
PLOT = False

ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1e4  # timesteps to observe before training
EXPLORE = 2e6  # frames over which to anneal epsilon
REPLAY_MEMORY = 1000  # number of previous transitions to remember
BATCH = 8  # size of minibatch
h_size = 512  # size of hidden cells of LSTM
trace_length = 8  # memory length
FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate

reward_dir = "../results/"+"rnn_"+str(ACTIONS)
network_dir = "../saved_networks/"+"rnn_"+str(ACTIONS)
log_dir = "../log/" + "rnn_" + str(ACTIONS)
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)
if not os.path.exists(network_dir):
    os.makedirs(network_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_location_ave = reward_dir + "/average_reward.csv"


class experience_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])


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


def trainNetwork(s, readout, keep_per, tl, bs, si, rnn_state, sess):
    # define the cost function
    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.compat.v1.reduce_sum(
        tf.multiply(readout, a), reduction_indices=1)
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cost)

    # open a test
    robot_explo = robot.Robot(0, TRAIN, PLOT)

    # tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # initialize an training environment
    myBuffer = experience_buffer(REPLAY_MEMORY)
    step_t = 0
    drop_rate = INITIAL_RATE
    total_reward = np.empty([0, 0])
    average_reward = np.empty([0, 0])
    init_state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    finish_all_map = False

    # saving and loading networks
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.compat.v1.train.get_checkpoint_state(network_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # get the first state by doing nothing and preprocess the image to 84x84x1
    x_t = robot_explo.begin()
    x_t = resize(x_t, (84, 84))
    s_t = np.reshape(x_t, (1, 84, 84, 1))
    state = init_state
    a_t_coll = []
    episodeBuffer = []

    while TRAIN and step_t <= EXPLORE:
        # scale down dropout rate
        if drop_rate > FINAL_RATE and step_t > OBSERVE:
            drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE

        # choose an action by uncertainty
        readout_t, state1 = sess.run([readout, rnn_state],
                                     feed_dict={s: s_t, keep_per: 1 - drop_rate, tl: 1, bs: 1, si: state})
        readout_t = readout_t[0]
        readout_t[a_t_coll] = None
        a_t = np.zeros([ACTIONS])
        action_index = np.nanargmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index, _ = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 84, 84, 1))
        s_t1 = x_t1
        finish = terminal

        # store the transition in D
        episodeBuffer.append(np.reshape(np.array([s_t, a_t, r_t, s_t1, terminal]), [1, 5]))

        if step_t > OBSERVE:
            # Reset the recurrent layer's hidden state
            state_train = (np.zeros([BATCH, h_size]),
                           np.zeros([BATCH, h_size]))

            # sample a minibatch to train on
            trainBatch = myBuffer.sample(BATCH, trace_length)

            # get the batch variables
            s_j_batch = np.vstack(trainBatch[:, 0])
            a_batch = np.vstack(trainBatch[:, 1])
            r_batch = np.vstack(trainBatch[:, 2]).flatten()
            s_j1_batch = np.vstack(trainBatch[:, 3])

            readout_j1_batch = readout.eval(
                feed_dict={s: s_j1_batch, keep_per: 0.2, tl: trace_length, bs: BATCH, si: state_train})[0]
            end_multiplier = -(np.vstack(trainBatch[:, 4]).flatten() - 1)
            y_batch = r_batch + GAMMA * np.max(readout_j1_batch) * end_multiplier

            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch,
                keep_per: 0.2,
                tl: trace_length,
                bs: BATCH,
                si: state_train}
            )
            new_average_reward = np.average(total_reward[len(total_reward) - 10000:])
            writer.add_scalar('average reward', new_average_reward, step_t)
            average_reward = np.append(average_reward, new_average_reward)

        step_t += 1
        total_reward = np.append(total_reward, r_t)

        # save progress
        if step_t % 2e4 == 0 or step_t % 2e5 == 0 or step_t % 2e6 == 0:
            saver.save(sess, network_dir+'/rnn', global_step=step_t)
            np.savetxt(file_location_ave, average_reward, delimiter=",")

        print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

        if finish:
            bufferArray = np.array(padd_eps(episodeBuffer))
            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            episodeBuffer = []
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, _ = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = np.reshape(x_t, (1, 84, 84, 1))
            a_t_coll = []
            state = init_state
            continue

        # update the old values
        if collision_index:
            a_t_coll.append(action_index)
            continue
        a_t_coll = []
        state = state1
        s_t = s_t1

    while not TRAIN and not finish_all_map:
        # choose an action by uncertainty
        readout_t, state1 = sess.run([readout, rnn_state],
                                     feed_dict={s: s_t, keep_per: 1, tl: 1, bs: 1, si: state})
        readout_t = readout_t[0]
        readout_t[a_t_coll] = None
        a_t = np.zeros([ACTIONS])
        action_index = np.nanargmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index, finish_all_map = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 84, 84, 1))
        s_t1 = x_t1
        finish = terminal

        step_t += 1
        print("TIMESTEP", step_t, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

        if finish:
            a_t_coll = []
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, finish_all_map = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = np.reshape(x_t, (1, 84, 84, 1))
            continue

        # update the old values
        if collision_index:
            a_t_coll.append(action_index)
            continue
        a_t_coll = []
        s_t = s_t1


def start_training():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)
    s, readout, keep_per, tl, bs, si, rnn_state = create_LSTM(ACTIONS, h_size)
    trainNetwork(s, readout, keep_per, tl, bs, si, rnn_state, sess)


if __name__ == "__main__":
    start_training()
