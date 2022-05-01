from casadi import *

x = SX.sym("x")
y = SX.sym("y")
z = SX.sym("z")

f_description_op = open("Test/CasADi/read_file_test/text.txt", "r")
#print(f_description.read())

f_description = f_description_op.read()
f_description = f_description.splitlines() # This allow me to splite lines of the read file
#print(f_description)

variables = [] # This list will contain all function variables

n_var = int(f_description[0])

for i in range(n_var):
    var = SX.sym(f_description[1].split()[i])
    # Globals is used to convert a string to a variable name
    variables.append(var)

f1 = SX.sym("f1")
f1 = eval(f_description[3])
print(type(f1))

var_vect = vertcat(variables[0],variables[1])
if n_var > 2:
    for i in range(2, n_var):
        var_vect = vertcat(var_vect, variables[i])

var_list = []
for i in range(n_var):
    var_list.append(var_vect[i])

print(var_list)

print(var_vect)

f = Function("f", var_list,[f1])
print(f)
print(f(0.5,1))







#print(type(variables[0]))
#f = Function(f_description[2],variables,function,['x','y'],['r'])

#f_description_op.close()

#print(f)

#r0 = f(5,3)

#j = derivative(f)

#f_result = open("funtion_results.txt","w+")