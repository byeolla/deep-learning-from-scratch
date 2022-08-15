#%%
import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
# %%
def sum_squares_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
# %%
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label = True)

print(x_train.shape) ; print(t_train.shape)
# %%
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask] ; t_batch = t_train[batch_mask]
#%%
# #%%
# y1 = [.1, .05, .6, .0, .05, .1,  .0, .1, .0, .0]
# y2 = [.1, .05, .1, .0, .05, .1,  .0, .6, .0, .0]
# t = [0, 0, 1, 0, 0, 0, 0 ,0, 0, 0]
# # %%
# cross_entropy_error(np.array(y1), np.array(t))
# #%%
# cross_entropy_error(np.array(y2), np.array(t))
# # %%
