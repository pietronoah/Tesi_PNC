from casadi import *
from numpy.core.fromnumeric import shape

# Creo delle variabii di tipo SX, 
# ovvero variabili simboliche che possono contenere
# operazioni unary e binary
x = SX.sym("x")
y = SX.sym("y")
z = SX.sym("z")
var = vertcat(x,y,z)

# Compongo una funzione
f1 = 1 - (x**3)*y - (y**2)*x + 12*y - 4*x + z
#print(f1)

dfdx = gradient(f1, x)
dfdy = gradient(f1, y)
#print(dfdx, dfdy)

dfdxdx = gradient(dfdx, x)
dfdydy = gradient(dfdy, y)
#print(dfdxdx, dfdydy)

f = Function("f", [x,y,z], [f1])
#print(f(0.5,1,1))

#print(gradient(f1, var))
#print(hessian(f1, var))

H = hessian(f1, var)

h = Function("h",[x,y,z],[H[0]])
h1 = h(2,3,2)
print(h1)

#print(h(x,y,z).shape())
#b = SX.zeros(shape(h))
#print((h(0.5,4,2.5)))
#print(solve(h))