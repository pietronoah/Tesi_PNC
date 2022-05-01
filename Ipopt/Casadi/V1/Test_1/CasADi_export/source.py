# This file is about hs071 problem. The objective of this is to find the minimum point value using ipopt and CasADi

# First I import the casadi library for python
from casadi import *

# Create the variables vector and the expressions of the function and of the constraints

n = 4

x    = SX.sym('x',4) # There are 4 variables, from x[0] to x[3]
objective = SX.sym('obj',1) # Objective function
constrains = SX.sym('cons',2) # Constrain functions

# Definition of the objective function
# Utilizzo un vettore unico perchè tanto la funzione obiettivo è sempre una unica espressione
objective = x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]

# Definition of constrain functions
constrains[0] = x[0]*x[1]*x[2]*x[3]
constrains[1] = pow(x[0],2)+pow(x[1],2)+pow(x[2],2)+pow(x[3],2)

# For ipopt I need only the gradient of the constrain functions, and for the objectve function I need the gradient and the hessian of it.


# Functions that I will use in order to generate the c code used inside the c++ ipopt iterface

# Objective funtion
[h_f,j_f] = hessian(objective,x) # h_f is the hessian of the function f and g_f is the gradient of the function f

obj_f       = Function('obj_f', [x],[objective])
obj_f_grad   = Function('obj_f_grad',[x], [j_f]) 
#obj_f_hes   = Function('obj_f_hes',[x], [h_f]) 


# Dato che l'hessiano della funzione è una matrice simmetrica, riporto solamente la parte inferiore della matrice, nel particolare ordine considerato da ipopt



# Creo all'interno di un loop questo vettore con il corretto ordine dei termini 
""" h_f1 = []

for i in range(n):
    for j in range(i+1):
        #h_f1 = h_f1 + str([h_f[i+(j*n)]])
        h_f1.append(([h_f[i+(j*n)]]))

h_f2 = SX.sym('h_f2',(len(h_f1)-1))
for i in range(len(h_f1)):
    h_f2[i] = h_f1[i]  """


""" hessian_f = SX.sym('h_f2',10)
for i in range(len(h_f1)):
    hessian_f[i] = h_f1[i] """

obj_f_hes   = Function('obj_f_hes',[x], [h_f]) 



# Now let's try to compute the jacobain of the entire vector of contrains
con_g  = Function('con_g',[x], [constrains])

j_g1 = jacobian(constrains,x)
j_g = j_g1.T
con_g_jac  = Function('con_g_jac',[x], [j_g])

h_g = jacobian(j_g,x)
con_g_hes  = Function('con_g_hes',[x], [h_g])


#print(j_g)
print(sparse(obj_f_hes))




""" C = CodeGenerator('source.c')
C.add(obj_f)
C.add(obj_f_grad)
C.add(obj_f_hes)
C.add(con_g)
C.add(con_g_jac)
C.add(con_g_hes)
C.generate() """


""" print(con_g_hes([2,2,2,2])[0])
print(con_g_hes([2,2,2,2])[1])
print(con_g_hes([2,2,2,2])[2])
print(con_g_hes([2,2,2,2])[3])
print(con_g_hes([2,2,2,2])[4])
print(con_g_hes([2,2,2,2])[5])
print(con_g_hes([2,2,2,2])[6])
print(con_g_hes([2,2,2,2])[7])
print(con_g_hes([2,2,2,2])[8])
print(con_g_hes([2,2,2,2])[9])
print(con_g_hes([2,2,2,2])[10])
print(con_g_hes([2,2,2,2])[11])
print(con_g_hes([2,2,2,2])[12])
print(con_g_hes([2,2,2,2])[13])
print(con_g_hes([2,2,2,2])[14])
print(con_g_hes([2,2,2,2])[15]) """







