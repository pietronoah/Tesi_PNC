# This file is about hs071 problem. The objective of this is to find the minimum point value using ipopt and CasADi

# First I import the casadi library for python
from casadi import *


# Create the variables vector and the expressions of the function and of the constraints

n = 9
m = 14

n_variables  = Function('n_variables',[], [n])
n_constrains  = Function('n_constrains',[], [m])



x    = SX.sym('x',n) # There are 4 variables, from x[0] to x[3]
objective = SX.sym('obj',1) # Objective function
constrains = SX.sym('cons',m) # Constrain functions

# Definition of the objective function
# Utilizzo un vettore unico perchè tanto la funzione obiettivo è sempre una unica espressione
objective = -0.5*(x[0]*x[3]-x[1]*x[2]+x[2]*x[8]-x[4]*x[8]+x[4]*x[7]-x[5]*x[6])


# Definition of constrain functions
constrains[0] = 1 - pow(x[2],2) - pow(x[3],2)
constrains[1] = 1 - pow(x[4],2) - pow(x[5],2)
constrains[2] = 1 - pow(x[8],2)
constrains[3] = 1 - pow(x[0],2) - pow(x[1]-x[8],2)
constrains[4] = 1 - pow(x[0]-x[4],2) - pow(x[1]-x[5],2)
constrains[5] = 1 - pow(x[0]-x[6],2) - pow(x[1]-x[7],2)
constrains[6] = 1 - pow(x[2]-x[6],2) - pow(x[3]-x[7],2)
constrains[7] = 1 - pow(x[2]-x[4],2) - pow(x[3]-x[5],2)
constrains[8] = 1 - pow(x[6],2) - pow(x[7]-x[8],2)
constrains[9] = x[0]*x[3]-x[1]*x[2]
constrains[10] = x[2]*x[8]
constrains[11] = -x[4]*x[8]
constrains[12] = x[4]*x[7]-x[5]*x[6]
constrains[13] = x[8]

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
    x_l[i] = 0
    x_u[i] = 2


g_l    = SX.sym('g_l',m)
g_u    = SX.sym('g_u',m)

g_l[0] = 0
g_u[0] = 2e19

g_l[1] = 0
g_u[1] = 2e19

g_l[2] = 0
g_u[2] = 2e19

g_l[3] = 0
g_u[3] = 2e19

g_l[4] = 0
g_u[4] = 2e19

g_l[5] = 0
g_u[5] = 2e19

g_l[6] = 0
g_u[6] = 2e19

g_l[7] = 0
g_u[7] = 2e19

g_l[8] = 0
g_u[8] = 2e19

g_l[9] = 0
g_u[9] = 2e19

g_l[10] = 0
g_u[10] = 2e19

g_l[11] = 0
g_u[11] = 2e19

g_l[12] = 0
g_u[12] = 2e19

g_l[13] = 0
g_u[13] = 2e19

# Convert into functions
x_l  = Function('x_l_casadi',[], [x_l])
x_u  = Function('x_u_casadi',[], [x_u])

g_l  = Function('g_l_casadi',[], [g_l])
g_u  = Function('g_u_casadi',[], [g_u])


# Set starting point for x
x_start    = SX.sym('x_start',n)
x_start[0] = 1
x_start[1] = 1
x_start[2] = 1
x_start[3] = 1
x_start[4] = 1
x_start[5] = 1
x_start[6] = 1
x_start[7] = 1
x_start[8] = 1
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








