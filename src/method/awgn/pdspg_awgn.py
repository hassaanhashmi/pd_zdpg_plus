import copy
import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Normal
from model.actor_awgn import Actor


class PD_SPG():
    r"""
    Primal-Dual Stochastic Policy Gradient
    """

    def __init__(self, env, num_users, pow_max, priority_weights, lr_x, lr_th, lr_lr, lr_lri, batch_size):
        self.env = env
        self.num_users = num_users
        self.pow_max = pow_max
        self.var_scale = np.sqrt(pow_max)
        self.priority_weights = priority_weights
        #initialize x, theta, lambda_s & lambda_r
        self.metrics_x = np.ones(shape=(self.num_users,1))
        # uncoupled neural networks initialized to 0.0 values
        self.policies = []
        self.optimizers = []
        self.lr_th = lr_th
        for i in range(self.num_users):
            self.policies.append(Actor())
            self.policies[i].apply(self.init_weights)
            self.optimizers.append(Adam(self.policies[i].parameters(), lr=self.lr_th))
        self.lamda_r = np.ones(shape=(self.num_users,1))
        self.lamda_ri = 1.0
        self.lr_x = lr_x
        self.lr_lr = lr_lr
        self.lr_lri = lr_lri
        self.batch_size = batch_size
        self.vec_H = np.zeros(shape=(self.num_users, self.batch_size))
        self.saved_log_probs = []
        self.f_h = np.zeros_like(self.vec_H)
        self.fi_h = np.zeros(self.batch_size)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def sample_ur(self):
        self.U_r = np.random.multivariate_normal(self.mean_r, self.cov_r, size=1).T
    
    def get_actions(self):
        vec_actions = np.zeros_like(self.vec_H)
        net_in = torch.from_numpy(self.vec_H)
        for i, policy in enumerate(self.policies):
            net_out = policy(net_in[i,:].reshape(-1,1))
            mean_vec, std_vec = self.pow_max*net_out[:, 0] , np.sqrt(self.var_scale)*net_out[:, 1]
            m = Normal(mean_vec,std_vec)
            actions = torch.clip(m.sample(), min=0, max=self.pow_max) #truncated normal
            self.saved_log_probs.append(m.log_prob(actions))
            vec_actions[i,:] = actions.detach().numpy()
        return vec_actions

    def delta_fi(self, fi_h, fi_uh):
        return (fi_uh - fi_h)/self.mu_r
        
    def delta_f(self, f_h, f_uh):
        return (f_uh - f_h)/self.mu_r

    def update_x(self):
        update_step = self.priority_weights - self.lamda_r
        return self.lr_x*update_step

    def update_theta(self):
        f_lamd_r = np.dot(self.f_h.T, self.lamda_r)
        fi_lamd_ri = self.fi_h*self.lamda_ri
        update_step = torch.from_numpy(np.squeeze(f_lamd_r) + fi_lamd_ri)
        for i, log_prob in enumerate(self.saved_log_probs):
            self.optimizers[i].zero_grad()
            loss = -torch.dot(update_step, log_prob.double())/self.batch_size
            loss.backward()
            self.optimizers[i].step()
    
    def update_lamda_ri(self, fi_h):
        return self.lr_lri*fi_h

    def update_lamda_r(self, f_h):
        update_step = f_h - self.metrics_x
        return self.lr_lr*update_step

    def reset_env(self):
        for n in range(self.batch_size):
            self.vec_H[:,n] = np.squeeze(self.env.reset())

    def step(self):
          #update x
          self.metrics_x += self.update_x()
          self.metrics_x = np.maximum(0,self.metrics_x)
          #draw action values from Gaussian Policies
          actions = self.get_actions()
          for s in range(self.batch_size):
              #probe 1
              g, a, b, _ = self.env.step(actions[:,s], self.metrics_x, self.vec_H[:,s])
              g_x, self.fi_h[s], self.f_h[:,s] = np.squeeze(g), np.squeeze(a), np.squeeze(b)
          #for plotting
          g_vec_f, _, _, _ = self.env.step(np.mean(actions, axis=1).reshape(-1,1), np.mean(self.f_h, axis=1).reshape(-1,1), self.vec_H)
              #update theta
          self.update_theta()
          actions_plus = self.get_actions()
          for s in range(self.batch_size):
              #probe 2
              _, a, b, c = self.env.step(actions_plus[:,s], self.metrics_x, self.vec_H[:,s])
              self.fi_h[s], self.f_h[:,s], self.vec_H[:,s] = np.squeeze(a), np.squeeze(b), np.squeeze(c)
          #update lambda_ri and lambda_r
          self.lamda_ri = np.maximum(0,self.lamda_ri - self.update_lamda_ri(np.mean(self.fi_h)))
          self.lamda_r = np.maximum(0,self.lamda_r - self.update_lamda_r(np.mean(self.f_h, axis=1).reshape(-1,1)))
          del self.saved_log_probs[:]
          fi_h_ = np.mean(self.fi_h)
          return g_vec_f, g_x, fi_h_