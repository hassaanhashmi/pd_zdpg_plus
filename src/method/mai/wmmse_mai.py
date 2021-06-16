import numpy as np


class WMMSE_Policy():
    r"""
    WMMSE Policy for MAI channel
    """

    def __init__(self, env, num_users, pow_max, priority_weights, noise_var, wmmse_iter):
        self.env = env
        self.num_users = num_users
        self.pow_max = pow_max
        self.priority_weights = priority_weights
        self.noise_var = noise_var
        self.wmmse_iter = wmmse_iter
        #initialize power and sumrate
        self.sumrate = 0.0
        self.p_wmmse = np.zeros(shape=(self.num_users, 1))
        self.vec_H = np.zeros(shape=(self.num_users, 1))
    
    def update_p_wmmse(self):
        h2 = np.ones(self.num_users)*self.vec_H
        m = h2.shape[0]
        h = np.sqrt(h2)
        h_diag = np.array([h.diagonal()]).T
        v = np.ones((m,1))*np.sqrt(self.pow_max)/m
        u = (h_diag*v)/(h2.dot(v**2) + self.noise_var)
        w = 1/(1 - u*h_diag*v)
        for i in range(self.wmmse_iter):
            A = self.priority_weights*w*u*h_diag
            v = A/(h2.dot(self.priority_weights**w*u**2))
            vAlt = np.sqrt(self.pow_max) * A / np.linalg.norm(A)
            if np.linalg.norm(v)**2 > self.pow_max:
                v = vAlt
            u = (h_diag*v)/(h2.dot(v**2) + self.noise_var)
            w = 1/(1 - u*h_diag*v)
        self.p_wmmse = v**2
        return self.p_wmmse

    def update_policy_sumrate(self, vec_f):
        self.sumrate = np.dot(vec_f.T, self.priority_weights)
        return self.sumrate

    def clear_values(self):
        self.p_wmmse *= 0.0
        self.sumrate = 0
    
    def reset_env(self):
      self.vec_H = self.env.reset()

    def step(self):
        metrics_x = np.zeros(shape=(self.num_users,1))
        self.clear_values()
        p_wmmse = self.update_p_wmmse()
        _, _, vec_f, self.vec_H = self.env.step(self.p_wmmse, metrics_x, self.vec_H)
        sumrate = self.update_policy_sumrate(vec_f)
        return sumrate