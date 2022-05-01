// In this file I will try to use the CasADi created file to evaluate functions 

#include "test_4_nlp.hpp"

#include <cassert>
#include <iostream>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;


dual2nd f(dual2nd* x)
{
  return pow(x[0]-10,3) + pow(x[1]-20,3);
}


dual2nd g_0(dual2nd* x)
{
  return pow(x[0]-5,2)+pow(x[1]-5,2);
}

dual2nd g_1(dual2nd* x)
{
  return pow(x[1]-5,2)+pow(x[0]-6,2);
}






using namespace Ipopt;

// constructor
HS071_NLP::HS071_NLP()
{}

//destructor
HS071_NLP::~HS071_NLP()
{}

// returns the size of the problem
bool HS071_NLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, IndexStyleEnum& index_style)
{

  
  // The problem described in HS071_NLP.hpp has 4 variables, x[0] through x[3]
  n = 2;

  // one equality constraint and one inequality constraint
  m = 2;

  // in this example the jacobian is dense and contains 8 nonzeros
  nnz_jac_g = 4;

  // the hessian is also dense and has 16 total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = 2;

  // use the C style indexing (0-based)
  index_style = TNLP::C_STYLE;

  return true;
}

// returns the variable bounds
bool HS071_NLP::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                Index m, Number* g_l, Number* g_u)
{
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.
  assert(n == 2);
  assert(m == 2);


  
    x_l[0] = 13;
    x_u[0] = 100;
  
    x_l[1] = 0;
    x_u[1] = 100;
  

  // the variables have upper bounds of 5
  
    g_l[0] = 100;
    g_u[0] = 2e+19;
  
    g_l[1] = -2e+19;
    g_u[1] = 82.81;
  

  return true;
}

// returns the initial point for the problem
bool HS071_NLP::get_starting_point(Index n, bool init_x, Number* x,
                                   bool init_z, Number* z_L, Number* z_U,
                                   Index m, bool init_lambda,
                                   Number* lambda)
{
  // Here, we assume we only have starting values for x, if you code
  // your own NLP, you can provide starting values for the dual variables
  // if you wish
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  
    x[0] = 20.1;
  
    x[1] = 5.84;
  

  return true;
}

// returns the value of the objective function
bool HS071_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  assert(n == 2);

  dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }
  obj_value = Number(f(p));


  return true;
}

// return the gradient of the objective function grad_{x} f(x)
bool HS071_NLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
  assert(n == 2);

  // Gradiente restituito come vettore colonna

  dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }

  auto d_f = gradient(f, wrt(p[0], p[1]), at(p));

  for (int i = 0; i < n; i++) {
    grad_f[i] = Number(d_f[i]);
  }


  return true;
}

// return the value of the constraints: g(x)
bool HS071_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  assert(n == 2);
  assert(m == 2);


  dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }

  
  g[0] = Number(g_0(p));
  
  g[1] = Number(g_1(p));
  

  return true;
}

// return the structure or values of the jacobian
bool HS071_NLP::eval_jac_g(Index n, const Number* x, bool new_x,
                           Index m, Index nele_jac, Index* iRow, Index *jCol,
                           Number* values)
{
  if (values == NULL) {
    // return the structure of the jacobian

    dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
    for(int i = 0; i < n; i++) {
      p[i] = rand() % 10000 + 1;
    }

    
    auto d_g_0 = gradient(g_0, wrt(p[0], p[1]), at(p));
    
    auto d_g_1 = gradient(g_1, wrt(p[0], p[1]), at(p));
    

    int index = 0;
    
    for(int i = 0; i < n; i++) {
      if (d_g_0[i] != 0) {
        iRow[index] = 0;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_1[i] != 0) {
        iRow[index] = 1;
        jCol[index] = i;
        index++;
      }
    }
    


  }
  else {
    // return the values of the jacobian of the constraints

    dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
    for(int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    
    auto d_g_0 = gradient(g_0, wrt(p[0], p[1]), at(p));
    
    auto d_g_1 = gradient(g_1, wrt(p[0], p[1]), at(p));
    

    int index = 0;
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_0[i]) != 0) {
        values[index] = Number(d_g_0[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_1[i]) != 0) {
        values[index] = Number(d_g_1[i]);
        index++;
      }
    }
    
  }

  return true;
}

//return the structure or values of the hessian
bool HS071_NLP::eval_h(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values)
{
  if (values == NULL) {
   
    dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
    for(int i = 0; i < n; i++) {
      p[i] = rand() % 10000 + 1;
    }

    double* values1 = (double*) calloc(n*n,sizeof(double));

    
    
    
    values1[0] = Number(derivatives(f, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] = Number(derivatives(f, wrt(p[1], p[0]), at(p))[2]);
    values1[2] = Number(derivatives(f, wrt(p[1], p[1]), at(p))[2]);


    
    
    
    
    values1[0] += Number(derivatives(g_0, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_0, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_0, wrt(p[1], p[1]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_1, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_1, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_1, wrt(p[1], p[1]), at(p))[2]);

    


    int idx = 0;
    for (int i = 0; i < n*n; i++) {
      if (values1[i] != 0) {
        int line_index = 0;
        int k = i;
        while (k > line_index) {
          k -= (line_index + 1);
          line_index++;
        }
        iRow[idx] = line_index;
        jCol[idx] = k;
        idx++;
      }
    }  

    assert(idx == nele_hess); // nele_hess == 10
  }
  else {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only


    dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
    for(int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    double* values1 = (double*) calloc(n*n,sizeof(double));

    
    
    
    values1[0] = Number(derivatives(f, wrt(p[0], p[0]), at(p))[2]) * obj_factor;
    
    values1[1] = Number(derivatives(f, wrt(p[1], p[0]), at(p))[2]) * obj_factor;
    values1[2] = Number(derivatives(f, wrt(p[1], p[1]), at(p))[2]) * obj_factor;


    
    
    
    
    values1[0] += Number(derivatives(g_0, wrt(p[0], p[0]), at(p))[2]) * lambda[0];
    
    values1[1] += Number(derivatives(g_0, wrt(p[1], p[0]), at(p))[2]) * lambda[0];
    values1[2] += Number(derivatives(g_0, wrt(p[1], p[1]), at(p))[2]) * lambda[0];

    
    
    
    
    values1[0] += Number(derivatives(g_1, wrt(p[0], p[0]), at(p))[2]) * lambda[1];
    
    values1[1] += Number(derivatives(g_1, wrt(p[1], p[0]), at(p))[2]) * lambda[1];
    values1[2] += Number(derivatives(g_1, wrt(p[1], p[1]), at(p))[2]) * lambda[1];

    




    int index = 0;
    for(int i = 0; i < n*n; i++) {
      if(values1[i] != 0) {
        values[index] = values1[i];
        index++;
      }
    }




  }

  return true;
}

void HS071_NLP::finalize_solution(SolverReturn status,
                                  Index n, const Number* x, const Number* z_L, const Number* z_U,
                                  Index m, const Number* g, const Number* lambda,
                                  Number obj_value,
				  const IpoptData* ip_data,
				  IpoptCalculatedQuantities* ip_cq)
{
  // here is where we would store the solution to variables, or write to a file, etc
  // so we could use the solution.

  // For this example, we write the solution to the console
  std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
  for (Index i=0; i<n; i++) {
     std::cout << "x[" << i << "] = " << x[i] << std::endl;
  }

  std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
  for (Index i=0; i<n; i++) {
    std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
  }
  for (Index i=0; i<n; i++) {
    std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
  }

  std::cout << std::endl << std::endl << "Objective value" << std::endl;
  std::cout << "f(x*) = " << obj_value << std::endl;

  std::cout << std::endl << "Final value of the constraints:" << std::endl;
  for (Index i=0; i<m ;i++) {
    std::cout << "g(" << i << ") = " << g[i] << std::endl;
  }
}