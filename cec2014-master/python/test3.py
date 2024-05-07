import jax
from evosax import LES
#from evosax.strategies.cma_es import CMA_ES, EvoParams
import jax.numpy as jnp
import numpy as np
import exp_function
import cec2014

def function(x):
    D=len(x[0])
    sum=100
    
    for i in range(D):
        sum+=(10**6)**((i) / (D - 1)) * x[:,i]**2
    return sum
# Instantiate the search strategy
rng = jax.random.PRNGKey(0)
strategy = LES(popsize=20, num_dims=10)
#es_params = EvoParams(mu_eff=jnp.array(5.9388046, dtype=jnp.float32), c_1=jnp.array(0.11884386, dtype=jnp.float32), c_mu=jnp.array(0.37442228, dtype=jnp.float32), c_sigma=jnp.array(0.61356556, dtype=jnp.float32), d_sigma=jnp.array(2.1797051, dtype=jnp.float32), c_c=jnp.array(0.5837605, dtype=jnp.float32), chi_n=jnp.array(1.2542727, dtype=jnp.float32), c_m=1.0, sigma_init=1.0, init_min=-100.0, init_max=100.0, clip_min=-3.4028235e+38, clip_max=3.4028235e+38)
es_params= strategy.default_params
state = strategy.initialize(rng, es_params)

# Run ask-eval-tell loop - NOTE: By default minimization!
for t in range(100):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, es_params)
    if t%10==0:
        print(x)
   
    fitness = exp_function.f1(x)# Your population evaluation fct 
    state = strategy.tell(x, fitness, state, es_params)
    

# Get best overall population member & its fitness
print(state.best_member, state.best_fitness)