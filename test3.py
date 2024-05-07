import jax
from evosax import LES
import jax.numpy as jnp
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
strategy = LES(popsize=20, num_dims=2)
es_params = strategy.default_params
state = strategy.initialize(rng, es_params)

# Run ask-eval-tell loop - NOTE: By default minimization!
for t in range(100):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, es_params)
   
    fitness = exp_function.f1(x)# Your population evaluation fct 
    state = strategy.tell(x, fitness, state, es_params)
    print(state.best_fitness)

# Get best overall population member & its fitness
state.best_member, state.best_fitness