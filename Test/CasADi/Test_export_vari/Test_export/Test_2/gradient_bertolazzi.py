from casadi import *

x    = SX.sym('x',2) # Vettore di due variabili x_0, x_1
expr = SX.sym('expr',2) # Espressione data da due funzioni 

#Definisco le due espressioni
expr[0] = x[0]*exp(x[1])**x[1]
expr[1] = log(1+sin(x[1])*x[0])

##Definisco le due funzioni che poi utilizzerò per genrare il codice C
f    = Function('Function', [x],[expr])
g    = Function('Function_jacobian',[x], [jacobian(expr,x)]) 
f.generate('gen.c')
g.generate('gen1.c')