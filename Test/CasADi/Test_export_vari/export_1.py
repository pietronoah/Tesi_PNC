from casadi import *

#This is a test file that must be able, given a function, to return a c file that contain the gradient of the function, and one that contain the hessian of the function


x    = SX.sym('x',2) # Vettore di due variabili x_0, x_1
expr = SX.sym('expr',1) # Espressione data da una funzione

#Definisco le due espressioni
#expr[0] = (x[0])^3 + 2(x[1])^2 - 7*x[0]*(x[1])^2
expr[0] = x[0]*x[1]+x[0]

##Definisco le due funzioni che poi utilizzerò per genrare il codice C
f    = Function('base_function', [x],[expr])
g    = Function('function_jacobian',[x], [jacobian(expr,x)]) 
h    = Function('function_hessian',[x], [hessian(expr,x)[0]]) 
f.generate('base_function.c')
g.generate('function_jacobian.c')
h.generate('function_hessian.c')