import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
#from tensorboardX import SummaryWriter
from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 5000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions_prob = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions_prob = [], []

    def make_action(self, state, test=False):
        probs = self.model(torch.tensor(state).unsqueeze(0))
        c = D.Categorical(probs)
        action = c.sample()
        self.saved_actions_prob.append(c.log_prob(action))
        return action.item()

    def update(self):
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        R = 0
        rd = []
        loss = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rd.append(R)
        rd.reverse()
        rd = torch.tensor(rd)
        rd = (rd - rd.mean()) / rd.std()
        for prob, r in zip(self.saved_actions_prob, rd):
            loss -= prob * r
        
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        #writer = SummaryWriter()
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                #self.saved_actions_prob.append(action)
                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            #writer.add_scalar('pg_reward', avg_reward, epoch)
            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))

            if avg_reward > 100: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
