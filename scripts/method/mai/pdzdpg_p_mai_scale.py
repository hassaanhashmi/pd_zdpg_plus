import numpy as np
import torch
from torch.optim import SGD
import gym
from model.actor_scale import Actor


class PD_ZDPG_Plus():
    def __init__(self, env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c, nn_scale):
        self.env = env
        self.num_users = num_users
        self.pow_max = pow_max
        self.priority_weights = priority_weights
        self.mu_r = mu_r
        #initialize x, theta, lambda_s & lambda_r
        self.metrics_x = np.zeros(shape=(self.num_users,1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = Actor(self.num_users, nn_scale).to(self.device)
        self.policy.apply(self.init_weights)
        self.lr_th = lr_th
        self.optim = SGD(self.policy.parameters(), lr=self.lr_th)
        self.lamda_r = np.ones(shape=(self.num_users,1))
        self.lamda_ri = 1.0
        self.mean_r = np.zeros(self.num_users)
        self.cov_r = np.eye(self.num_users)
        self.U_r = np.random.multivariate_normal(self.mean_r, self.cov_r, size=1).T
        self.lr_x = lr_x
        self.lr_lr = lr_lr
        self.lr_lri = lr_lri
        self.slack = self.mu_r*c
        self.vec_H = np.zeros(shape=(self.num_users, 1))
        self.fi = 0

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def sample_ur(self):
        self.U_r = np.random.multivariate_normal(self.mean_r, self.cov_r, size=1).T
    
    def get_actions(self):
        vec_actions = np.zeros_like(self.vec_H)
        net_in = torch.tensor(np.squeeze(self.vec_H)).to(self.device)
        vec_actions[:,0] = self.policy(net_in).cpu().detach().numpy().T
        return self.pow_max*vec_actions
    
    def get_actions_mu(self):
        return self.get_actions() + self.mu_r*self.U_r

    def delta_fi(self, fi_h, fi_uh):
        return (fi_uh - fi_h)/self.mu_r
        
    def delta_f(self, f_h, f_uh):
        return (f_uh - f_h)/self.mu_r

    def update_x(self):
        update_step = self.priority_weights - self.lamda_r
        return self.lr_x*update_step

    def update_theta(self, delta_f, delta_fi):
        self.optim.zero_grad()
        net_in = torch.from_numpy(np.squeeze(self.vec_H)).to(self.device)
        delta_f_lamd_r = np.dot(delta_f.T, self.lamda_r)[0,0]
        delta_fi_lamd_ri = delta_fi*self.lamda_ri
        update_step = torch.from_numpy((delta_f_lamd_r + delta_fi_lamd_ri)*np.squeeze(self.U_r)).to(self.device)
        out = -self.pow_max*torch.dot(update_step,self.policy(net_in))
        out.backward()
        self.optim.step()
    
    def update_lamda_ri(self, fi_uh):
        return self.lr_lri*fi_uh

    def update_lamda_r(self, f_uh):
        update_step = f_uh - self.metrics_x - self.slack
        return self.lr_lr*update_step


    def reset_env(self):
        self.vec_H = self.env.reset()

    def step(self):
        #update x
        self.metrics_x += self.update_x()
        self.metrics_x = np.maximum(0,self.metrics_x)
        #sample U_r
        self.sample_ur()
        #probe 1
        actions = self.get_actions()
        g_x, fi_h, f_h, _ = self.env.step(actions, self.metrics_x, self.vec_H)
        #for plotting
        g_vec_f, _, _, _ = self.env.step(actions, f_h, self.vec_H)
        #probe 2
        actions_mu = np.maximum(0, self.get_actions_mu())
        _, fi_uh, f_uh, _ = self.env.step(actions_mu, self.metrics_x, self.vec_H)
        #calculate delta's
        delta_fi = self.delta_fi(fi_h, fi_uh)
        delta_f = self.delta_f(f_h, f_uh)
        #update theta
        self.update_theta(delta_f, delta_fi)
        #probe 3
        actions_mu = self.get_actions_mu()
        _, fi_uh, f_uh, self.vec_H = self.env.step(actions_mu, self.metrics_x, self.vec_H)
        #update lambda_ri and lambda_r
        self.lamda_ri = np.maximum(0,self.lamda_ri - self.update_lamda_ri(fi_uh))
        self.lamda_r = np.maximum(0,self.lamda_r - self.update_lamda_r(f_uh))
        self.fi = fi_h
        return g_vec_f, g_x