import os
from skimage.transform import resize
import numpy as np
import random
from Networks import create_LSTM
import torch
from torch.utils.tensorboard import SummaryWriter
import robot_simulation as robot

# training environment parameters
TRAIN = True
ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1e2  # timesteps to observe before training
EXPLORE = 2e6  # frames over which to anneal epsilon
REPLAY_MEMORY = 1e3  # number of previous transitions to remember
BATCH = 4  # size of minibatch
h_size = 512  # size of hidden cells of LSTM
trace_length = 8  # memory length
FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate
TARGET_UPDATE = 15000
max_grad_norm = 0.5

# torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network_dir = "../saved_networks/" + "rnn_" + str(ACTIONS)
log_dir = "../log/" + "rnn_" + str(ACTIONS)
if not os.path.exists(network_dir):
    os.makedirs(network_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class experience_buffer():
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        element_size = np.size(sampled_episodes[0][0])
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, element_size])


def padd_eps(eps_buff):
    if len(eps_buff) < trace_length:
        s = np.zeros([1, 1, 84, 84])
        a = np.zeros([ACTIONS])
        r = 0
        s1 = np.zeros([1, 1, 84, 84])
        d = True
        for i in range(0, trace_length - len(eps_buff)):
            eps_buff.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
    return eps_buff


def cost(pred, target, action):
    readout_action = torch.sum(pred*action, dim=1)
    loss = torch.sqrt(torch.mean((readout_action-target)**2))
    return loss


def train(data, action, y, state, model, optimizer):
    data = torch.tensor(data).to(device, dtype=torch.float)
    action = torch.tensor(action).to(device, dtype=torch.float)
    y = torch.tensor(y).to(device, dtype=torch.float)
    model.train()
    optimizer.zero_grad()
    out, _ = model(data, 0.8, state, trace_length, BATCH, h_size)
    loss = cost(out, y, action)
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-max_grad_norm, max_grad_norm)
    optimizer.step()
    return loss.item()


def test(data, prob, state_in, model, trace_length, BATCH, h_size):
    data = torch.tensor(data).to(device, dtype=torch.float)
    model.eval()
    pred, state_out = model(data, prob, state_in, trace_length, BATCH, h_size)
    return pred, state_out


def start_training():
    # initialize networks
    policy_net = create_LSTM(ACTIONS).to(device)
    target_net = create_LSTM(ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)
    init_state = (torch.zeros(1, 1, h_size).to(device), torch.zeros(1, 1, h_size).to(device))

    # initialize an training environment
    robot_explo = robot.Robot(0, TRAIN)
    step_t = 0
    drop_rate = INITIAL_RATE
    total_reward = np.empty([0, 0])

    # store the previous observations in replay memory
    myBuffer = experience_buffer(REPLAY_MEMORY)

    # tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # load model
    policy_model_name = network_dir + '/Model_Policy.pt'
    if os.path.exists(policy_model_name):
        check_point_p = torch.load(policy_model_name)
        policy_net.load_state_dict(check_point_p)
        print("Successfully loaded")
    else:
        print("Could not find old network weights")

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t = robot_explo.begin()
    x_t = resize(x_t, (84, 84))
    s_t = np.reshape(x_t, (1, 1, 84, 84))
    state = init_state
    a_t_coll = []
    episodeBuffer = []

    while step_t <= EXPLORE:
        # scale down dropout rate
        if drop_rate > FINAL_RATE and step_t > OBSERVE:
            drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE

        # choose an action by uncertainty
        readout_t, state1 = test(s_t, drop_rate, state, policy_net, 1, 1, h_size)
        readout_t = readout_t.cpu().detach().numpy()
        readout_t[a_t_coll] = None
        a_t = np.zeros([ACTIONS])
        action_index = np.nanargmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 1, 84, 84))
        s_t1 = x_t1
        finish = terminal

        # store the transition in D
        episodeBuffer.append(np.reshape(np.array([s_t, a_t, r_t, s_t1, terminal]), [1, 5]))

        if step_t > OBSERVE:
            # updata target network
            if step_t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Reset the recurrent layer's hidden state
            state_train = (torch.zeros(1, BATCH, h_size).to(device), torch.zeros(1, BATCH, h_size).to(device))

            # sample a minibatch to train on
            trainBatch = myBuffer.sample(BATCH, trace_length)

            # get the batch variables
            s_j_batch = np.vstack(trainBatch[:, 0])
            a_batch = np.vstack(trainBatch[:, 1])
            r_batch = np.vstack(trainBatch[:, 2]).flatten()
            s_j1_batch = np.vstack(trainBatch[:, 3])

            readout_j1_batch, _ = test(s_j1_batch, 0., state_train, target_net, trace_length, BATCH, h_size)
            readout_j1_batch = readout_j1_batch.cpu().detach().numpy()
            end_multiplier = -(np.vstack(trainBatch[:, 4]).flatten() - 1)
            y_batch = r_batch + GAMMA * np.max(readout_j1_batch) * end_multiplier

            # perform gradient step
            temp_loss = train(s_j_batch, a_batch, y_batch, state_train, policy_net, optimizer)
            new_average_reward = np.average(total_reward[len(total_reward) - 10000:])
            writer.add_scalar('Train/avg_reward', new_average_reward, step_t)
            writer.add_scalar('Train/loss', temp_loss, step_t)

        step_t += 1
        total_reward = np.append(total_reward, r_t)

        # save progress
        if step_t % 500000 == 0:
            # save neural networks
            torch.save(policy_net.state_dict(), network_dir + '/MyModel.pt')

        print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

        if finish:
            bufferArray = np.array(padd_eps(episodeBuffer))
            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            episodeBuffer = []
            if re_locate:
                x_t = robot_explo.rescuer()
            if complete:
                x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = np.reshape(x_t, (1, 1, 84, 84))
            a_t_coll = []
            continue

        # avoid collision next time
        if collision_index:
            a_t_coll.append(action_index)
            continue
        a_t_coll = []
        s_t = s_t1


if __name__ == "__main__":
    start_training()
