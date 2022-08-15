#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
def function_1(x):
    return 0.01*x**2+ 0.1*x

def function_2(x, y):
    return x

def numerical_diff(f, x):
    h = 1e-4
    
    return (f(x+h)-f(x-h)) / (2*h)
#%%
x = np.linspace(-2, 2, 7) ; y = np.linspace(-2, 2, 7)


z = function_2(x)
# %%
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
# %%
numerical_diff(function_1, 5)
numerical_diff(function_1, 10)
# %%
