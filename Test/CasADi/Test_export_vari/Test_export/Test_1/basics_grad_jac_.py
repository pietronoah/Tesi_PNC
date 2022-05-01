from casadi import *

# Creo delle variabii di tipo SX, 
# ovvero variabili simboliche che possono conteneri
# operazioni unary e binary
x = SX.sym("x")
y = SX.sym("y")
z = SX.sym("z")

# Compongo una funzione


f = Function('f',[x,y,z],[x*x + 2*y + z*z*z + arccos(arctan(x*y))],['x','y','z'] ,['r'])

f = x*x + 2*y + z*z*z + arccos(arctan(x*y))

print("f: ",f)

# Utilizzo la funzione gradient per crearne il gradiente
g = gradient(f,x)
#print("gradient_f: ", gradient(f,x))

# Utilizzo la fuzione hessiano per calcolarne l'essiano, ovvero la derivata seconda
[H,G] = hessian(f,x)
print("gradient_f: ", H, " and hessian_f is: ", H)

# Now try a substitution
print(substitute(f,x,3))

