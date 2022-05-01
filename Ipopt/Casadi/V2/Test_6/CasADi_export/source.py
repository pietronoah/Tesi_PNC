# This file is about hs071 problem. The objective of this is to find the minimum point value using ipopt and CasADi

# First I import the casadi library for python
from casadi import *


# Create the variables vector and the expressions of the function and of the constraints

n = 7
m = 2

n_variables  = Function('n_variables',[], [n])
n_constrains  = Function('n_constrains',[], [m])



x    = SX.sym('x',n) # There are 4 variables, from x[0] to x[3]
objective = SX.sym('obj',1) # Objective function
constrains = SX.sym('cons',m) # Constrain functions

# Definition of the objective function
# Utilizzo un vettore unico perchè tanto la funzione obiettivo è sempre una unica espressione
objective = pow(x[0]-10,2) + 5*pow(x[1]-12,2) + pow(x[2],4) + 3*pow(x[3]-11,2) + 10*pow(x[4],6) + 7*pow(x[5],2) + pow(x[6],4) - 4*x[5]*x[6] -10*x[5] - 8*x[6]


# Definition of constrain functions
constrains[0] = 2*pow(x[0],2) + 3*pow(x[1],4) +x[2] +4*pow(x[3],2) + 5*x[4]
constrains[1] = -4*pow(x[0],2) -pow(x[1],2) + 3*x[0]*x[1] -2*pow(x[2],2) - 5*x[5] + 11*x[6]

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


# Adesso provo a calcolare il numero di elementi non zero delle matrici

# Elementi non zero dello jacobiano
j_g_nze = 0
for i in range(m):
    for j in range(n):
        if (str(j_g[i,j]) != "00" and str(j_g[i,j]) != "0"):
            j_g_nze += 1

con_g_jac_nze  = Function('con_g_jac_nze',[], [j_g_nze])


#Elementi non zero della matrice hessiana
ipopt_h_f = h_f
for i in range(m):
    ipopt_h_f = ipopt_h_f + h_g1[i*n:(i+1)*n,:]


h_f_nze = 0
for i in range(n):
    for j in range(i+1):
        if (str(ipopt_h_f[i,j]) != "00" and str(ipopt_h_f[i,j]) != "0"):
            h_f_nze += 1

obj_f_hes_nze  = Function('obj_f_hes_nze',[], [h_f_nze])




# Set variable and constrains boundaries
x_l    = SX.sym('x_l',n)
x_u    = SX.sym('x_u',n)
for i in range(n):
    x_l[i] = -5
    x_u[i] = 5


g_l    = SX.sym('g_l',m)
g_u    = SX.sym('g_u',m)

g_l[0] = 127
g_u[0] = 127

g_l[1] = 0
g_u[1] = 0

# Convert into functions
x_l  = Function('x_l_casadi',[], [x_l])
x_u  = Function('x_u_casadi',[], [x_u])

g_l  = Function('g_l_casadi',[], [g_l])
g_u  = Function('g_u_casadi',[], [g_u])


# Set starting point for x
x_start    = SX.sym('x_start',n)
x_start[0] = 1
x_start[1] = 2
x_start[2] = 0
x_start[3] = 4
x_start[4] = 0
x_start[5] = 1
x_start[6] = 1
x_start_point  = Function('x_start_point',[], [x_start])


# Genero il codice C aggiungendo tutte le funzioni

C = CodeGenerator('source.c')
C.add(obj_f)
C.add(obj_f_grad)
C.add(obj_f_hes)
C.add(con_g)
C.add(con_g_jac)
C.add(con_g_hes)
C.add(n_variables)
C.add(n_constrains)
C.add(con_g_jac_nze)
C.add(obj_f_hes_nze)
C.add(x_l)
C.add(x_u)
C.add(g_l)
C.add(g_u)
C.add(x_start_point)
C.generate() 








