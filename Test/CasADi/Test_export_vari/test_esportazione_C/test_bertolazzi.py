from casadi import *

x    = SX.sym('x',2)
expr = SX.sym('expr',2)
expr[0] = x[0]*exp(x[1])**x[1];
expr[1] = log(1+sin(x[1])*x[0])
f    = Function('f', [x],[expr])
g    = Function('pluto',[x], [jacobian(expr,x)]) 
f.generate('gen.c')
g.generate('gen1.c')