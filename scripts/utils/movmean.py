import numpy as np

'''
Centered Moving Average of a column vector x over
window k
'''
def movmean(x,k):
    '''
    Centered Moving Average of a column vector x over
    window k
    '''
    y = np.zeros(shape=x.shape[0])
    for i in range(x.shape[0]):
        if i < (k)//2:
            y[i] = np.mean(x[:i+(k-1)//2+1])
        else:
            y[i] = np.mean(x[i-k//2:i+(k-1)//2+1])
    return y