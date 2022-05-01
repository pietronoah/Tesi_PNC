from casadi import *

x = MX.sym('x',2) 
y =MX.sym('y')
f = Function('pippo',[x,y], [x,sin(y)*x],['x','y'] ,['r','q']) 
f.generate ('gen.c')