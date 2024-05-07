# We import the algorithm (You can use from pyade import * to import all of them)
import pyade
from pyade import *
import numpy as np

# You may want to use a variable so its easier to change it if we want
algorithm = pyade.de
algorithm2= pyade.lshade

# We get default parameters for a problem with two variables
params = algorithm.get_default_params(dim=2) 
params2 = algorithm2.get_default_params(dim=2) 
# We define the boundaries of the variables
params['bounds'] = np.array([[-75, 75]] * 2) 
params2['bounds'] = np.array([[-75, 75]] * 2)
# We indicate the function we want to minimize
params['func'] = lambda x: x[0]**2 + x[1]**2 + x[0]*x[1] - 500 
params2['func'] = lambda x: x[0]**2 + x[1]**2 + x[0]*x[1] - 500 
# We run the algorithm and obtain the results
solution, fitness = algorithm.apply(**params)
solution2, fitness2 = algorithm2.apply(**params2)
print(solution)
print(fitness)

print(solution2)
print(fitness2)