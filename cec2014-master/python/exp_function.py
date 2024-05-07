import numpy as np
import jax.numpy as jnp
import math
import cec2014

def f1(x):
    bias = 100
    num_point=len(x)
    result=np.zeros(num_point)
    for i in range(num_point):
        t=np.array(x[i],dtype=np.float64)
        result[i]=cec2014.cec14(t,5)
    result=np.transpose(result)
    
    return result

def f7(x):
    bias = 700
    sum1=0
    sum2=1
    D=len(x[0])
    for i in range(D):
        sum1 += x[:,i]**2
    for i in range(D):
        sum2 *= jnp.cos(x[:,i]/math.sqrt(i+1))
    return 1+sum1/4000-sum2 + bias

def f8(x):
    bias = 800
    D=len(x[0])
    sum=0
    for i in range(D):
        sum += x[:,i]**2 - 10*jnp.cos(2*math.pi*x[:,i]) + 10
    return sum + bias
