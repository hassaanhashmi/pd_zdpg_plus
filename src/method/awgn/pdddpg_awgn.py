import copy
import numpy as np
import torch
from torch.optim import SGD
from torch.nn import MSELoss
from model.actor_awgn import Actor
from model.critic_awgn import Critic


class PD_DDPG():
    r"""
    Primal-Dual Deep Deterministic Policy Gradient (Actor-Critic)
    """

    def __init__(self, env, num_users, pow_max, priority_weights, lr_x, lr_actor,lr_critic, lr_lr, lr_lri, lr_decay):
        self.env = env
        self.num_users = num_users
        self.pow_max = pow_max
        self.var_scale = np.sqrt(pow_max)
        self.priority_weights = priority_weights
        self.lr_decay = lr_decay
        #initialize x, actor-critic, lambda_s & lambda_r
        self.metrics_x = np.ones(shape=(self.num_users,1))
        self.actor = []
        self.actor_params = []
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        for i in range(self.num_users):
            self.actor.append(Actor(1))
            self.actor[i].apply(self.init_weights_actor)
            self.actor_params += list(self.actor[i].parameters())
        self.actor_optim = SGD(self.actor_params, lr=self.lr_actor)
        self.critic = Critic(self.num_users, self.num_users)
        self.critic_init = 1e-6
        self.critic.apply(self.init_weights_critic)
        self.critic_optim = SGD(self.critic.parameters(), lr=self.lr_critic)
        self.critic_loss = MSELoss()
        self.lamda_r = np.ones(shape=(self.num_users,1))
        self.lamda_ri = 1.0

        self.lr_x = lr_x
        self.lr_lr = lr_lr
        self.lr_lri = lr_lri

        self.vec_H = np.zeros(shape=(self.num_users, 1))
        self.sample_action_vals = torch.zeros(size=(self.num_users, 1)).double()

    def init_weights_actor(self, m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
    
    def init_weights_critic(self, m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(self.critic_init)
            m.bias.data.fill_(self.critic_init)
    
    def get_actions(self):
        net_input = torch.from_numpy(self.vec_H)
        for i, actor in enumerate(self.actor):
            self.sample_action_vals[i] = self.pow_max*actor(net_input[i])
        return self.sample_action_vals.detach().numpy()

    def update_x(self):
        update_step = self.priority_weights - self.lamda_r
        return self.lr_x*update_step

    def update_ac(self, vec_f, fi):
        vec_H_in = torch.tensor(np.squeeze(self.vec_H))
        lamdas = torch.tensor(np.concatenate((np.squeeze(self.lamda_r, axis=1), 
                                              np.expand_dims(self.lamda_ri, axis=0))))
        #Update Critic
        critic_tar = np.concatenate((np.squeeze(vec_f, axis=1), np.expand_dims(fi, axis=0)))
        critic_tar = torch.from_numpy(critic_tar)
        self.critic_optim.zero_grad()
        critic_out = self.critic([vec_H_in, torch.squeeze(self.sample_action_vals.detach())])
        critic_loss = self.critic_loss(critic_out, critic_tar)
        critic_loss.backward()
        self.critic_optim.step()
        #Update Actors
        self.actor_optim.zero_grad()
        actor_loss = -self.critic([vec_H_in, torch.squeeze(self.sample_action_vals)])
        actor_loss = torch.dot(actor_loss, lamdas)
        actor_loss.backward()
        self.actor_optim.step()


    def update_lamda_ri(self, fi_h):
        return self.lr_lri*fi_h

    def update_lamda_r(self, f_h):
        update_step = f_h - self.metrics_x
        return self.lr_lr*update_step


    def reset_env(self):
        self.vec_H = self.env.reset()

    def step(self):
        #update x
        self.metrics_x += self.update_x()
        self.metrics_x = np.maximum(0,self.metrics_x)
        actions = self.get_actions()
        #probe 1
        g_x, fi_h, f_h, _ = self.env.step(actions, self.metrics_x, self.vec_H)
        #for plotting
        g_vec_f, _, _, _ = self.env.step(actions, f_h, self.vec_H)
        #update theta
        self.update_ac(f_h, fi_h)
        #probe 2
        actions_plus = self.get_actions()
        _, fi_h_plus, f_h_plus, self.vec_H = self.env.step(actions_plus, self.metrics_x, self.vec_H)
        #update lambda_ri and lambda_r
        self.lamda_ri = np.maximum(0,self.lamda_ri - self.update_lamda_ri(fi_h_plus))
        self.lamda_r = np.maximum(0,self.lamda_r - self.update_lamda_r(f_h_plus))
        self.sample_action_vals.detach_()
        return g_vec_f, g_x, fi_h