import copy
import numpy as np
import torch
from model.actor_scale import Actor


class PD_ZDPG():
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
        self.lamda_ri = 1
        self.lamda_r = np.ones(shape=(self.num_users,1))
        self.mean_s = np.zeros(self.num_users)
        self.cov_s = np.eye(self.num_users)
        self.U_s = np.random.multivariate_normal(self.mean_s, self.cov_s, size=1).T
        self.U_r = []
        self.lr_x = lr_x
        self.lr_th = lr_th
        self.lr_lri = lr_lri
        self.lr_lr = lr_lr
        self.slack = self.mu_r*c
        self.vec_H = np.zeros(shape=(self.num_users, 1))

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def sample_gaussian_vectors(self):
        self.U_s = np.random.multivariate_normal(self.mean_s, self.cov_s, size=1).T
        self.U_r = []
        with torch.no_grad():
            for param in self.policy.parameters():
              if param.requires_grad:
                self.U_r.append(torch.randn_like(param.data).to(self.device))

    def get_actions(self):
        vec_actions = np.zeros_like(self.vec_H)
        net_input = torch.from_numpy(self.vec_H.T).to(self.device)
        vec_actions[:,0]=self.policy(net_input).cpu().detach().numpy()
        return self.pow_max*vec_actions
    
    def get_actions_mu(self):
        vec_actions = np.zeros_like(self.vec_H)
        net_input = torch.from_numpy(self.vec_H.T.astype(np.float64)).to(self.device)
        temp_policy = copy.deepcopy(self.policy)
        for i, (name, param) in enumerate(temp_policy.named_parameters()):
            if param.requires_grad:
                param.data += self.mu_r*self.U_r[i]
                dict(temp_policy.named_parameters())[name].data.copy_(param.data)
        vec_actions[:,0]=temp_policy(net_input).cpu().detach().numpy()
        return self.pow_max*vec_actions

    def delta_fi(self, fi_uh, fi_h):
        return (fi_uh - fi_h)/self.mu_r
        
    def delta_f(self, f_uh, f_h):
        return (f_uh - f_h)/self.mu_r

    def update_x(self):
        update_step = self.priority_weights - self.lamda_r
        return self.lr_x*update_step
    
    def update_theta(self, delta_f, delta_fi):
      delta_f_lamd_r = np.dot(delta_f.T, self.lamda_r)[0,0]
      delta_fi_lamd_ri = delta_fi*self.lamda_ri
      update_step = torch.tensor(delta_f_lamd_r + delta_fi_lamd_ri).to(self.device)
      for i, (name, param) in enumerate(self.policy.named_parameters()):
        if param.requires_grad:
            param.data += self.lr_th*update_step*self.U_r[i]
            dict(self.policy.named_parameters())[name].data.copy_(param.data)


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
        #sample U_s and U_r
        self.sample_gaussian_vectors()
        #get NN outputs
        actions = self.get_actions()
        actions_mu = self.get_actions_mu()
        #probe 1
        g_x, fi_h, f_h, _ = self.env.step(actions, self.metrics_x, self.vec_H)
        #for plotting
        g_vec_f, _, _, _ = self.env.step(actions, f_h, self.vec_H)
        #probe 2
        _, fi_uh, f_uh, _ = self.env.step(actions_mu, self.metrics_x, self.vec_H)
        #calculate delta's
        delta_fi = self.delta_fi(fi_uh, fi_h)
        delta_f = self.delta_f(f_uh, f_h)
        #update theta
        self.update_theta(delta_f, delta_fi)
        #probe 3
        actions_mu = self.get_actions_mu()
        _, fi_uh_plus, f_uh_plus, self.vec_H = self.env.step(actions_mu, self.metrics_x, self.vec_H)
        #update lambda_ri and lambda_r
        self.lamda_r = np.maximum(0,self.lamda_r - self.update_lamda_r(f_uh_plus))
        self.lamda_ri = np.maximum(0,self.lamda_ri - self.update_lamda_ri(fi_uh_plus))
        return g_vec_f, g_x


