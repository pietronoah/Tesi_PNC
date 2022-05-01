from casadi import *

x    = SX.sym('x',2) # Vettore di due variabili x_0, x_1
expr = SX.sym('expr',1) # Espressione data da due funzioni 

# Definisco le due espressioni
expr[0] = pow(x[0],3) - 3*pow(x[1],2) - 2

# Definisco le due funzioni che poi utilizzerò per genrare il codice C
f    = Function('Function', [x],[expr])

[h,g] = hessian(expr[0],x)

g    = Function('Function_jacobian',[x], [g]) 
h    = Function('Function_hessian',[x], [h]) 


#print(f)
#print(g)
#print(h([1,1]))

C = CodeGenerator('gen.c')
C.add(f)
C.add(g)
C.add(h)
C.generate()
