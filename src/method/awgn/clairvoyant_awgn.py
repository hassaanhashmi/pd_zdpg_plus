import numpy as np

class Clairvoyant_Policy():
    r"""
    Clairvoyant Policy for AWGN channel
    """

    def __init__(self, env, num_users, pow_max, priority_weights, noise_var, lr_lu):
        self.env = env
        self.num_users = num_users
        self.pow_max = pow_max
        self.priority_weights = priority_weights
        self.noise_var = noise_var
        #initialize lambda and sumrate
        self.lamda_u = 1.0
        self.lr_lu = lr_lu
        self.vec_H = np.zeros(shape=(self.num_users, 1))
        
    def update_pu(self):
        pu = np.zeros(shape=(self.num_users, 1))
        for i in range(self.num_users):
            pu[i] = np.maximum(0, self.priority_weights[i]/self.lamda_u - np.divide(self.noise_var,self.vec_H[i]))
        return pu
    
    def update_sumrate(self, vec_f):
        sumrate = 0.0
        for i in range(self.num_users):
            sumrate += vec_f[i] * self.priority_weights[i]
        return sumrate
    
    def update_lamda_u(self, fi):
        self.lamda_u = np.maximum(0, self.lamda_u - self.lr_lu*fi)
    
    def reset_env(self):
        self.vec_H = self.env.reset()

    def step(self):
        metrics_x = np.zeros(shape=(self.num_users,1))
        pu = self.update_pu()
        _, fi, vec_f, self.vec_H = self.env.step(pu, metrics_x, self.vec_H)
        sumrate = self.update_sumrate(vec_f)
        self.update_lamda_u(fi)
        return sumrate
