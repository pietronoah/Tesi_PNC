from casadi import *

x = SX.sym("x")
y = SX.sym("y")
z = SX.sym("z")

f = Function("f",[x,y,z],[x*x + 2*y + z*z*z + arccos(arctan(x*y))])
print(f)

# You can also add the option to add name to the single INPUTS and OUTPUTS
g = Function('g',[x,y],[x*y,sin(y)*x],['x','y'],['r','q'])
print(g)
# This way I rapresented a funtion tha goes from R2 to R2

# Now I try to generate a C code
f.generate("f.c")

r0, q0 = g(5,3)
print('r0:',r0)
print('q0:',q0)
