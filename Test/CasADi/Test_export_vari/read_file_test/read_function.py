from casadi import *

f_description_op = open("read_file_test/text.txt", "r")
#print(f_description.read())

f_description = f_description_op.read()
f_description = f_description.splitlines() # This allow me to splite lines of the read file
print(f_description)

variables = [] # This list will contain all function variables
function = [SX.sym(f_description[3])]

n_var = int(f_description[0])
for i in range(n_var):
    globals()[f_description[1].split()[i]] = SX.sym(f_description[1].split()[i]) # Globals is used to convert a string to a variable name
    variables.append(SX.sym(f_description[1].split()[i]))

f = Function(f_description[2],variables,function,['x','y'],['r'])

f_description_op.close()

print(f)

r0 = f(5,3)

#j = derivative(f)

#f_result = open("funtion_results.txt","w+")