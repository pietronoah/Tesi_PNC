from casadi import *

x    = SX.sym('x',2) # Vettore di due variabili x_0, x_1
expr = SX.sym('expr',2) # Espressione data da due funzioni 

# Definisco le due espressioni
expr[0] = pow(x[0],2) + x[1] - 14
expr[1] = pow(x[0],3) - 3*pow(x[1],2) - 2

# Definisco le due funzioni che poi utilizzerò per genrare il codice C
f    = Function('Function', [x],[expr])
g    = Function('Function_jacobian',[x], [jacobian(expr,x)]) 
#h    = Function('Function_hessian',[x], [hessian(expr,x)[0]]) 

# Qui utilizzo un CodeGeneratotr che mi permette di esportaare un unico file .c contenente tutte le funzioni che voglio
C = CodeGenerator('gen.c')
C.add(f)
C.add(g)
#C.add(h)
C.generate()