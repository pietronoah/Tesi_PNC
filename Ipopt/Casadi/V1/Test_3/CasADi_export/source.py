# This file is about hs071 problem. The objective of this is to find the minimum point value using ipopt and CasADi

# First I import the casadi library for python
from casadi import *

# Create the variables vector and the expressions of the function and of the constraints

n = 5
m = 3

x             = SX.sym('x',5) # There are 5 variables, from x[0] to x[4]
objective     = SX.sym('obj',1) # Objective function
constrains    = SX.sym('cons',3) # Constrain functions

# Definition of the objective function
# Utilizzo un vettore unico perchè tanto la funzione obiettivo è sempre una unica espressione
objective     = pow(x[0]-1,2) + pow(x[0]-x[1],2) + pow(x[1]-x[2],2) + pow(x[2]-x[3],4) + pow(x[3]-x[4],4)

# Definition of constrain functions
constrains[0] = x[0] + pow(x[1],2) + pow(x[2],3)
constrains[1] = x[1] -  pow(x[2], 2) + x[3]
constrains[2] = x[0] * x[4]

# Functions that I will use in order to generate the c code used inside the c++ ipopt iterface

# Objective funtion
[h_f,j_f] = hessian(objective,x) # h_f is the hessian of the function f and g_f is the gradient of the function f

obj_f       = Function('obj_f', [x],[objective])
obj_f_grad   = Function('obj_f_grad',[x], [j_f]) 
obj_f_hes   = Function('obj_f_hes',[x], [h_f]) 



# Now let's try to compute the jacobain of the entire vector of contrains
con_g  = Function('con_g',[x], [constrains])

j_g1 = jacobian(constrains,x)
j_g = j_g1
con_g_jac  = Function('con_g_jac',[x], [j_g])

h_g = jacobian(j_g,x)

# Now I reorder rows of the hessian matrix to have better handling inside ipopt code

for i in range(m):
    for j in range(n):
        if i == 0 and j == 0:
            h_g1 = h_g[0,:]
        else:
            h_g1 = vertcat(h_g1,h_g[i+m*j,:])



con_g_hes  = Function('con_g_hes',[x], [h_g1])






# Genero il codice C aggiungendo tutte le funzioni

C = CodeGenerator('source.c')
C.add(obj_f)
C.add(obj_f_grad)
C.add(obj_f_hes)
C.add(con_g)
C.add(con_g_jac)
C.add(con_g_hes)
C.generate()







