from jinja2 import *

ipopt_nlp = open('templates/ipopt_nlp.txt','r');
ipopt_nlp_text = ipopt_nlp.read()

ipopt_main = open('templates/ipopt_main.txt','r');
ipopt_main_text = ipopt_main.read()

ipopt_nlp_hpp = open('templates/ipopt_nlp_hpp.txt','r');
ipopt_nlp_hpp_text = ipopt_nlp_hpp.read()

cmake_template = open('templates/cmake_template.txt','r');
cmake_template_text = cmake_template.read()

from source import *

t1 = Template(ipopt_nlp_text)
t11 = t1.render(objective1=objective1,constrains1=constrains1,project_name=project_name,n=n,m=m,j_g_nze=j_g_nze,h_f_nze=h_f_nze,x_l=x_l,x_u=x_u,g_l=g_l,g_u=g_u,x_start=x_start)

t2 = Template(ipopt_main_text)
t22 = t2.render(project_name=project_name)

t3 = Template(ipopt_nlp_hpp_text)
t33 = t3.render()

nlp_name = project_name + '_nlp.cpp'
file1 = open(nlp_name,'w');
file1.write(t11)

main_name = project_name + '_main.cpp'
file2 = open(main_name,'w');
file2.write(t22)

nlp_hpp_name = project_name + '_nlp.hpp'
file3 = open(nlp_hpp_name,'w');
file3.write(t33)

t4 = Template(cmake_template_text)
t44 = t4.render(project_name=project_name,main_name=main_name,nlp_name=nlp_name)

file4 = open('CMakeLists.txt','w');
file4.write(t44)


