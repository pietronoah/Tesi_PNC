// In this file I will try to use the CasADi created file to evaluate functions 

#include "test_5_nlp.hpp"

#include <cassert>
#include <iostream>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;


dual2nd f(dual2nd* x)
{
  return -0.5*(x[0]*x[3]-x[1]*x[2]+x[2]*x[8]-x[4]*x[8]+x[4]*x[7]-x[5]*x[6]);
}


dual2nd g_0(dual2nd* x)
{
  return 1 - pow(x[2],2) - pow(x[3],2);
}

dual2nd g_1(dual2nd* x)
{
  return 1 - pow(x[4],2) - pow(x[5],2);
}

dual2nd g_2(dual2nd* x)
{
  return 1 - pow(x[8],2);
}

dual2nd g_3(dual2nd* x)
{
  return 1 - pow(x[0],2) - pow(x[1]-x[8],2);
}

dual2nd g_4(dual2nd* x)
{
  return 1 - pow(x[0]-x[4],2) - pow(x[1]-x[5],2);
}

dual2nd g_5(dual2nd* x)
{
  return 1 - pow(x[0]-x[6],2) - pow(x[1]-x[7],2);
}

dual2nd g_6(dual2nd* x)
{
  return 1 - pow(x[2]-x[6],2) - pow(x[3]-x[7],2);
}

dual2nd g_7(dual2nd* x)
{
  return 1 - pow(x[2]-x[4],2) - pow(x[3]-x[5],2);
}

dual2nd g_8(dual2nd* x)
{
  return 1 - pow(x[6],2) - pow(x[7]-x[8],2);
}

dual2nd g_9(dual2nd* x)
{
  return x[0]*x[3]-x[1]*x[2];
}

dual2nd g_10(dual2nd* x)
{
  return x[2]*x[8];
}

dual2nd g_11(dual2nd* x)
{
  return -x[4]*x[8];
}

dual2nd g_12(dual2nd* x)
{
  return x[4]*x[7]-x[5]*x[6];
}

dual2nd g_13(dual2nd* x)
{
  return x[8];
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
  n = 9;

  // one equality constraint and one inequality constraint
  m = 14;

  // in this example the jacobian is dense and contains 8 nonzeros
  nnz_jac_g = 40;

  // the hessian is also dense and has 16 total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = 25;

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
  assert(n == 9);
  assert(m == 14);


  
    x_l[0] = 0;
    x_u[0] = 2;
  
    x_l[1] = 0;
    x_u[1] = 2;
  
    x_l[2] = 0;
    x_u[2] = 2;
  
    x_l[3] = 0;
    x_u[3] = 2;
  
    x_l[4] = 0;
    x_u[4] = 2;
  
    x_l[5] = 0;
    x_u[5] = 2;
  
    x_l[6] = 0;
    x_u[6] = 2;
  
    x_l[7] = 0;
    x_u[7] = 2;
  
    x_l[8] = 0;
    x_u[8] = 2;
  

  // the variables have upper bounds of 5
  
    g_l[0] = 0;
    g_u[0] = 2e+19;
  
    g_l[1] = 0;
    g_u[1] = 2e+19;
  
    g_l[2] = 0;
    g_u[2] = 2e+19;
  
    g_l[3] = 0;
    g_u[3] = 2e+19;
  
    g_l[4] = 0;
    g_u[4] = 2e+19;
  
    g_l[5] = 0;
    g_u[5] = 2e+19;
  
    g_l[6] = 0;
    g_u[6] = 2e+19;
  
    g_l[7] = 0;
    g_u[7] = 2e+19;
  
    g_l[8] = 0;
    g_u[8] = 2e+19;
  
    g_l[9] = 0;
    g_u[9] = 2e+19;
  
    g_l[10] = 0;
    g_u[10] = 2e+19;
  
    g_l[11] = 0;
    g_u[11] = 2e+19;
  
    g_l[12] = 0;
    g_u[12] = 2e+19;
  
    g_l[13] = 0;
    g_u[13] = 2e+19;
  

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

  
    x[0] = 1;
  
    x[1] = 1;
  
    x[2] = 1;
  
    x[3] = 1;
  
    x[4] = 1;
  
    x[5] = 1;
  
    x[6] = 1;
  
    x[7] = 1;
  
    x[8] = 1;
  

  return true;
}

// returns the value of the objective function
bool HS071_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  assert(n == 9);

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
  assert(n == 9);

  // Gradiente restituito come vettore colonna

  dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }

  auto d_f = gradient(f, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));

  for (int i = 0; i < n; i++) {
    grad_f[i] = Number(d_f[i]);
  }


  return true;
}

// return the value of the constraints: g(x)
bool HS071_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  assert(n == 9);
  assert(m == 14);


  dual2nd* p = (dual2nd*) calloc(n, sizeof(dual2nd));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }

  
  g[0] = Number(g_0(p));
  
  g[1] = Number(g_1(p));
  
  g[2] = Number(g_2(p));
  
  g[3] = Number(g_3(p));
  
  g[4] = Number(g_4(p));
  
  g[5] = Number(g_5(p));
  
  g[6] = Number(g_6(p));
  
  g[7] = Number(g_7(p));
  
  g[8] = Number(g_8(p));
  
  g[9] = Number(g_9(p));
  
  g[10] = Number(g_10(p));
  
  g[11] = Number(g_11(p));
  
  g[12] = Number(g_12(p));
  
  g[13] = Number(g_13(p));
  

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

    
    auto d_g_0 = gradient(g_0, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_1 = gradient(g_1, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_2 = gradient(g_2, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_3 = gradient(g_3, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_4 = gradient(g_4, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_5 = gradient(g_5, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_6 = gradient(g_6, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_7 = gradient(g_7, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_8 = gradient(g_8, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_9 = gradient(g_9, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_10 = gradient(g_10, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_11 = gradient(g_11, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_12 = gradient(g_12, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_13 = gradient(g_13, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    

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
    
    for(int i = 0; i < n; i++) {
      if (d_g_2[i] != 0) {
        iRow[index] = 2;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_3[i] != 0) {
        iRow[index] = 3;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_4[i] != 0) {
        iRow[index] = 4;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_5[i] != 0) {
        iRow[index] = 5;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_6[i] != 0) {
        iRow[index] = 6;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_7[i] != 0) {
        iRow[index] = 7;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_8[i] != 0) {
        iRow[index] = 8;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_9[i] != 0) {
        iRow[index] = 9;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_10[i] != 0) {
        iRow[index] = 10;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_11[i] != 0) {
        iRow[index] = 11;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_12[i] != 0) {
        iRow[index] = 12;
        jCol[index] = i;
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if (d_g_13[i] != 0) {
        iRow[index] = 13;
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

    
    auto d_g_0 = gradient(g_0, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_1 = gradient(g_1, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_2 = gradient(g_2, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_3 = gradient(g_3, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_4 = gradient(g_4, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_5 = gradient(g_5, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_6 = gradient(g_6, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_7 = gradient(g_7, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_8 = gradient(g_8, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_9 = gradient(g_9, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_10 = gradient(g_10, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_11 = gradient(g_11, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_12 = gradient(g_12, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    
    auto d_g_13 = gradient(g_13, wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]), at(p));
    

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
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_2[i]) != 0) {
        values[index] = Number(d_g_2[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_3[i]) != 0) {
        values[index] = Number(d_g_3[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_4[i]) != 0) {
        values[index] = Number(d_g_4[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_5[i]) != 0) {
        values[index] = Number(d_g_5[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_6[i]) != 0) {
        values[index] = Number(d_g_6[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_7[i]) != 0) {
        values[index] = Number(d_g_7[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_8[i]) != 0) {
        values[index] = Number(d_g_8[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_9[i]) != 0) {
        values[index] = Number(d_g_9[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_10[i]) != 0) {
        values[index] = Number(d_g_10[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_11[i]) != 0) {
        values[index] = Number(d_g_11[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_12[i]) != 0) {
        values[index] = Number(d_g_12[i]);
        index++;
      }
    }
    
    for(int i = 0; i < n; i++) {
      if(Number(d_g_13[i]) != 0) {
        values[index] = Number(d_g_13[i]);
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
    
    values1[3] = Number(derivatives(f, wrt(p[2], p[0]), at(p))[2]);
    values1[4] = Number(derivatives(f, wrt(p[2], p[1]), at(p))[2]);
    values1[5] = Number(derivatives(f, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] = Number(derivatives(f, wrt(p[3], p[0]), at(p))[2]);
    values1[7] = Number(derivatives(f, wrt(p[3], p[1]), at(p))[2]);
    values1[8] = Number(derivatives(f, wrt(p[3], p[2]), at(p))[2]);
    values1[9] = Number(derivatives(f, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] = Number(derivatives(f, wrt(p[4], p[0]), at(p))[2]);
    values1[11] = Number(derivatives(f, wrt(p[4], p[1]), at(p))[2]);
    values1[12] = Number(derivatives(f, wrt(p[4], p[2]), at(p))[2]);
    values1[13] = Number(derivatives(f, wrt(p[4], p[3]), at(p))[2]);
    values1[14] = Number(derivatives(f, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] = Number(derivatives(f, wrt(p[5], p[0]), at(p))[2]);
    values1[16] = Number(derivatives(f, wrt(p[5], p[1]), at(p))[2]);
    values1[17] = Number(derivatives(f, wrt(p[5], p[2]), at(p))[2]);
    values1[18] = Number(derivatives(f, wrt(p[5], p[3]), at(p))[2]);
    values1[19] = Number(derivatives(f, wrt(p[5], p[4]), at(p))[2]);
    values1[20] = Number(derivatives(f, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] = Number(derivatives(f, wrt(p[6], p[0]), at(p))[2]);
    values1[22] = Number(derivatives(f, wrt(p[6], p[1]), at(p))[2]);
    values1[23] = Number(derivatives(f, wrt(p[6], p[2]), at(p))[2]);
    values1[24] = Number(derivatives(f, wrt(p[6], p[3]), at(p))[2]);
    values1[25] = Number(derivatives(f, wrt(p[6], p[4]), at(p))[2]);
    values1[26] = Number(derivatives(f, wrt(p[6], p[5]), at(p))[2]);
    values1[27] = Number(derivatives(f, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] = Number(derivatives(f, wrt(p[7], p[0]), at(p))[2]);
    values1[29] = Number(derivatives(f, wrt(p[7], p[1]), at(p))[2]);
    values1[30] = Number(derivatives(f, wrt(p[7], p[2]), at(p))[2]);
    values1[31] = Number(derivatives(f, wrt(p[7], p[3]), at(p))[2]);
    values1[32] = Number(derivatives(f, wrt(p[7], p[4]), at(p))[2]);
    values1[33] = Number(derivatives(f, wrt(p[7], p[5]), at(p))[2]);
    values1[34] = Number(derivatives(f, wrt(p[7], p[6]), at(p))[2]);
    values1[35] = Number(derivatives(f, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] = Number(derivatives(f, wrt(p[8], p[0]), at(p))[2]);
    values1[37] = Number(derivatives(f, wrt(p[8], p[1]), at(p))[2]);
    values1[38] = Number(derivatives(f, wrt(p[8], p[2]), at(p))[2]);
    values1[39] = Number(derivatives(f, wrt(p[8], p[3]), at(p))[2]);
    values1[40] = Number(derivatives(f, wrt(p[8], p[4]), at(p))[2]);
    values1[41] = Number(derivatives(f, wrt(p[8], p[5]), at(p))[2]);
    values1[42] = Number(derivatives(f, wrt(p[8], p[6]), at(p))[2]);
    values1[43] = Number(derivatives(f, wrt(p[8], p[7]), at(p))[2]);
    values1[44] = Number(derivatives(f, wrt(p[8], p[8]), at(p))[2]);


    
    
    
    
    values1[0] += Number(derivatives(g_0, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_0, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_0, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_0, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_0, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_0, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_0, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_0, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_0, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_0, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_0, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_0, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_0, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_0, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_0, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_0, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_0, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_0, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_0, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_0, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_0, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_0, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_0, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_0, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_0, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_0, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_0, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_0, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_0, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_0, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_0, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_0, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_0, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_0, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_0, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_0, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_0, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_0, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_0, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_0, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_0, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_0, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_0, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_0, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_0, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_1, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_1, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_1, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_1, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_1, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_1, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_1, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_1, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_1, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_1, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_1, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_1, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_1, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_1, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_1, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_1, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_1, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_1, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_1, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_1, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_1, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_1, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_1, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_1, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_1, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_1, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_1, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_1, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_1, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_1, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_1, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_1, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_1, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_1, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_1, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_1, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_1, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_1, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_1, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_1, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_1, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_1, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_1, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_1, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_1, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_2, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_2, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_2, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_2, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_2, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_2, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_2, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_2, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_2, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_2, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_2, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_2, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_2, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_2, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_2, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_2, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_2, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_2, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_2, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_2, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_2, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_2, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_2, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_2, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_2, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_2, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_2, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_2, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_2, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_2, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_2, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_2, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_2, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_2, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_2, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_2, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_2, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_2, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_2, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_2, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_2, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_2, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_2, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_2, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_2, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_3, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_3, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_3, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_3, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_3, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_3, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_3, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_3, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_3, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_3, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_3, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_3, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_3, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_3, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_3, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_3, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_3, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_3, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_3, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_3, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_3, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_3, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_3, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_3, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_3, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_3, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_3, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_3, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_3, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_3, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_3, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_3, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_3, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_3, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_3, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_3, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_3, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_3, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_3, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_3, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_3, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_3, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_3, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_3, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_3, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_4, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_4, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_4, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_4, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_4, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_4, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_4, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_4, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_4, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_4, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_4, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_4, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_4, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_4, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_4, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_4, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_4, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_4, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_4, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_4, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_4, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_4, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_4, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_4, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_4, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_4, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_4, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_4, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_4, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_4, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_4, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_4, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_4, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_4, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_4, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_4, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_4, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_4, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_4, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_4, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_4, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_4, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_4, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_4, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_4, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_5, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_5, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_5, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_5, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_5, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_5, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_5, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_5, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_5, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_5, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_5, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_5, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_5, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_5, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_5, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_5, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_5, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_5, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_5, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_5, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_5, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_5, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_5, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_5, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_5, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_5, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_5, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_5, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_5, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_5, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_5, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_5, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_5, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_5, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_5, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_5, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_5, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_5, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_5, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_5, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_5, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_5, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_5, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_5, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_5, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_6, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_6, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_6, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_6, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_6, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_6, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_6, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_6, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_6, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_6, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_6, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_6, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_6, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_6, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_6, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_6, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_6, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_6, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_6, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_6, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_6, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_6, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_6, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_6, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_6, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_6, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_6, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_6, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_6, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_6, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_6, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_6, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_6, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_6, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_6, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_6, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_6, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_6, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_6, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_6, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_6, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_6, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_6, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_6, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_6, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_7, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_7, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_7, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_7, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_7, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_7, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_7, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_7, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_7, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_7, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_7, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_7, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_7, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_7, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_7, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_7, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_7, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_7, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_7, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_7, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_7, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_7, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_7, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_7, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_7, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_7, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_7, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_7, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_7, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_7, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_7, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_7, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_7, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_7, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_7, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_7, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_7, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_7, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_7, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_7, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_7, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_7, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_7, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_7, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_7, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_8, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_8, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_8, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_8, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_8, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_8, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_8, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_8, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_8, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_8, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_8, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_8, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_8, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_8, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_8, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_8, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_8, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_8, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_8, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_8, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_8, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_8, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_8, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_8, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_8, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_8, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_8, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_8, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_8, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_8, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_8, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_8, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_8, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_8, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_8, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_8, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_8, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_8, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_8, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_8, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_8, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_8, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_8, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_8, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_8, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_9, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_9, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_9, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_9, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_9, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_9, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_9, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_9, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_9, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_9, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_9, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_9, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_9, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_9, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_9, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_9, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_9, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_9, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_9, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_9, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_9, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_9, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_9, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_9, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_9, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_9, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_9, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_9, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_9, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_9, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_9, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_9, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_9, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_9, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_9, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_9, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_9, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_9, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_9, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_9, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_9, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_9, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_9, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_9, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_9, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_10, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_10, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_10, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_10, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_10, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_10, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_10, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_10, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_10, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_10, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_10, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_10, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_10, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_10, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_10, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_10, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_10, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_10, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_10, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_10, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_10, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_10, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_10, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_10, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_10, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_10, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_10, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_10, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_10, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_10, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_10, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_10, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_10, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_10, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_10, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_10, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_10, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_10, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_10, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_10, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_10, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_10, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_10, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_10, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_10, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_11, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_11, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_11, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_11, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_11, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_11, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_11, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_11, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_11, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_11, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_11, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_11, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_11, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_11, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_11, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_11, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_11, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_11, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_11, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_11, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_11, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_11, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_11, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_11, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_11, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_11, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_11, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_11, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_11, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_11, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_11, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_11, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_11, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_11, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_11, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_11, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_11, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_11, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_11, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_11, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_11, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_11, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_11, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_11, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_11, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_12, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_12, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_12, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_12, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_12, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_12, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_12, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_12, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_12, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_12, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_12, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_12, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_12, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_12, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_12, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_12, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_12, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_12, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_12, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_12, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_12, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_12, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_12, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_12, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_12, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_12, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_12, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_12, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_12, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_12, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_12, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_12, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_12, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_12, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_12, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_12, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_12, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_12, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_12, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_12, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_12, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_12, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_12, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_12, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_12, wrt(p[8], p[8]), at(p))[2]);

    
    
    
    
    values1[0] += Number(derivatives(g_13, wrt(p[0], p[0]), at(p))[2]);
    
    values1[1] += Number(derivatives(g_13, wrt(p[1], p[0]), at(p))[2]);
    values1[2] += Number(derivatives(g_13, wrt(p[1], p[1]), at(p))[2]);
    
    values1[3] += Number(derivatives(g_13, wrt(p[2], p[0]), at(p))[2]);
    values1[4] += Number(derivatives(g_13, wrt(p[2], p[1]), at(p))[2]);
    values1[5] += Number(derivatives(g_13, wrt(p[2], p[2]), at(p))[2]);
    
    values1[6] += Number(derivatives(g_13, wrt(p[3], p[0]), at(p))[2]);
    values1[7] += Number(derivatives(g_13, wrt(p[3], p[1]), at(p))[2]);
    values1[8] += Number(derivatives(g_13, wrt(p[3], p[2]), at(p))[2]);
    values1[9] += Number(derivatives(g_13, wrt(p[3], p[3]), at(p))[2]);
    
    values1[10] += Number(derivatives(g_13, wrt(p[4], p[0]), at(p))[2]);
    values1[11] += Number(derivatives(g_13, wrt(p[4], p[1]), at(p))[2]);
    values1[12] += Number(derivatives(g_13, wrt(p[4], p[2]), at(p))[2]);
    values1[13] += Number(derivatives(g_13, wrt(p[4], p[3]), at(p))[2]);
    values1[14] += Number(derivatives(g_13, wrt(p[4], p[4]), at(p))[2]);
    
    values1[15] += Number(derivatives(g_13, wrt(p[5], p[0]), at(p))[2]);
    values1[16] += Number(derivatives(g_13, wrt(p[5], p[1]), at(p))[2]);
    values1[17] += Number(derivatives(g_13, wrt(p[5], p[2]), at(p))[2]);
    values1[18] += Number(derivatives(g_13, wrt(p[5], p[3]), at(p))[2]);
    values1[19] += Number(derivatives(g_13, wrt(p[5], p[4]), at(p))[2]);
    values1[20] += Number(derivatives(g_13, wrt(p[5], p[5]), at(p))[2]);
    
    values1[21] += Number(derivatives(g_13, wrt(p[6], p[0]), at(p))[2]);
    values1[22] += Number(derivatives(g_13, wrt(p[6], p[1]), at(p))[2]);
    values1[23] += Number(derivatives(g_13, wrt(p[6], p[2]), at(p))[2]);
    values1[24] += Number(derivatives(g_13, wrt(p[6], p[3]), at(p))[2]);
    values1[25] += Number(derivatives(g_13, wrt(p[6], p[4]), at(p))[2]);
    values1[26] += Number(derivatives(g_13, wrt(p[6], p[5]), at(p))[2]);
    values1[27] += Number(derivatives(g_13, wrt(p[6], p[6]), at(p))[2]);
    
    values1[28] += Number(derivatives(g_13, wrt(p[7], p[0]), at(p))[2]);
    values1[29] += Number(derivatives(g_13, wrt(p[7], p[1]), at(p))[2]);
    values1[30] += Number(derivatives(g_13, wrt(p[7], p[2]), at(p))[2]);
    values1[31] += Number(derivatives(g_13, wrt(p[7], p[3]), at(p))[2]);
    values1[32] += Number(derivatives(g_13, wrt(p[7], p[4]), at(p))[2]);
    values1[33] += Number(derivatives(g_13, wrt(p[7], p[5]), at(p))[2]);
    values1[34] += Number(derivatives(g_13, wrt(p[7], p[6]), at(p))[2]);
    values1[35] += Number(derivatives(g_13, wrt(p[7], p[7]), at(p))[2]);
    
    values1[36] += Number(derivatives(g_13, wrt(p[8], p[0]), at(p))[2]);
    values1[37] += Number(derivatives(g_13, wrt(p[8], p[1]), at(p))[2]);
    values1[38] += Number(derivatives(g_13, wrt(p[8], p[2]), at(p))[2]);
    values1[39] += Number(derivatives(g_13, wrt(p[8], p[3]), at(p))[2]);
    values1[40] += Number(derivatives(g_13, wrt(p[8], p[4]), at(p))[2]);
    values1[41] += Number(derivatives(g_13, wrt(p[8], p[5]), at(p))[2]);
    values1[42] += Number(derivatives(g_13, wrt(p[8], p[6]), at(p))[2]);
    values1[43] += Number(derivatives(g_13, wrt(p[8], p[7]), at(p))[2]);
    values1[44] += Number(derivatives(g_13, wrt(p[8], p[8]), at(p))[2]);

    


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
    
    values1[3] = Number(derivatives(f, wrt(p[2], p[0]), at(p))[2]) * obj_factor;
    values1[4] = Number(derivatives(f, wrt(p[2], p[1]), at(p))[2]) * obj_factor;
    values1[5] = Number(derivatives(f, wrt(p[2], p[2]), at(p))[2]) * obj_factor;
    
    values1[6] = Number(derivatives(f, wrt(p[3], p[0]), at(p))[2]) * obj_factor;
    values1[7] = Number(derivatives(f, wrt(p[3], p[1]), at(p))[2]) * obj_factor;
    values1[8] = Number(derivatives(f, wrt(p[3], p[2]), at(p))[2]) * obj_factor;
    values1[9] = Number(derivatives(f, wrt(p[3], p[3]), at(p))[2]) * obj_factor;
    
    values1[10] = Number(derivatives(f, wrt(p[4], p[0]), at(p))[2]) * obj_factor;
    values1[11] = Number(derivatives(f, wrt(p[4], p[1]), at(p))[2]) * obj_factor;
    values1[12] = Number(derivatives(f, wrt(p[4], p[2]), at(p))[2]) * obj_factor;
    values1[13] = Number(derivatives(f, wrt(p[4], p[3]), at(p))[2]) * obj_factor;
    values1[14] = Number(derivatives(f, wrt(p[4], p[4]), at(p))[2]) * obj_factor;
    
    values1[15] = Number(derivatives(f, wrt(p[5], p[0]), at(p))[2]) * obj_factor;
    values1[16] = Number(derivatives(f, wrt(p[5], p[1]), at(p))[2]) * obj_factor;
    values1[17] = Number(derivatives(f, wrt(p[5], p[2]), at(p))[2]) * obj_factor;
    values1[18] = Number(derivatives(f, wrt(p[5], p[3]), at(p))[2]) * obj_factor;
    values1[19] = Number(derivatives(f, wrt(p[5], p[4]), at(p))[2]) * obj_factor;
    values1[20] = Number(derivatives(f, wrt(p[5], p[5]), at(p))[2]) * obj_factor;
    
    values1[21] = Number(derivatives(f, wrt(p[6], p[0]), at(p))[2]) * obj_factor;
    values1[22] = Number(derivatives(f, wrt(p[6], p[1]), at(p))[2]) * obj_factor;
    values1[23] = Number(derivatives(f, wrt(p[6], p[2]), at(p))[2]) * obj_factor;
    values1[24] = Number(derivatives(f, wrt(p[6], p[3]), at(p))[2]) * obj_factor;
    values1[25] = Number(derivatives(f, wrt(p[6], p[4]), at(p))[2]) * obj_factor;
    values1[26] = Number(derivatives(f, wrt(p[6], p[5]), at(p))[2]) * obj_factor;
    values1[27] = Number(derivatives(f, wrt(p[6], p[6]), at(p))[2]) * obj_factor;
    
    values1[28] = Number(derivatives(f, wrt(p[7], p[0]), at(p))[2]) * obj_factor;
    values1[29] = Number(derivatives(f, wrt(p[7], p[1]), at(p))[2]) * obj_factor;
    values1[30] = Number(derivatives(f, wrt(p[7], p[2]), at(p))[2]) * obj_factor;
    values1[31] = Number(derivatives(f, wrt(p[7], p[3]), at(p))[2]) * obj_factor;
    values1[32] = Number(derivatives(f, wrt(p[7], p[4]), at(p))[2]) * obj_factor;
    values1[33] = Number(derivatives(f, wrt(p[7], p[5]), at(p))[2]) * obj_factor;
    values1[34] = Number(derivatives(f, wrt(p[7], p[6]), at(p))[2]) * obj_factor;
    values1[35] = Number(derivatives(f, wrt(p[7], p[7]), at(p))[2]) * obj_factor;
    
    values1[36] = Number(derivatives(f, wrt(p[8], p[0]), at(p))[2]) * obj_factor;
    values1[37] = Number(derivatives(f, wrt(p[8], p[1]), at(p))[2]) * obj_factor;
    values1[38] = Number(derivatives(f, wrt(p[8], p[2]), at(p))[2]) * obj_factor;
    values1[39] = Number(derivatives(f, wrt(p[8], p[3]), at(p))[2]) * obj_factor;
    values1[40] = Number(derivatives(f, wrt(p[8], p[4]), at(p))[2]) * obj_factor;
    values1[41] = Number(derivatives(f, wrt(p[8], p[5]), at(p))[2]) * obj_factor;
    values1[42] = Number(derivatives(f, wrt(p[8], p[6]), at(p))[2]) * obj_factor;
    values1[43] = Number(derivatives(f, wrt(p[8], p[7]), at(p))[2]) * obj_factor;
    values1[44] = Number(derivatives(f, wrt(p[8], p[8]), at(p))[2]) * obj_factor;


    
    
    
    
    values1[0] += Number(derivatives(g_0, wrt(p[0], p[0]), at(p))[2]) * lambda[0];
    
    values1[1] += Number(derivatives(g_0, wrt(p[1], p[0]), at(p))[2]) * lambda[0];
    values1[2] += Number(derivatives(g_0, wrt(p[1], p[1]), at(p))[2]) * lambda[0];
    
    values1[3] += Number(derivatives(g_0, wrt(p[2], p[0]), at(p))[2]) * lambda[0];
    values1[4] += Number(derivatives(g_0, wrt(p[2], p[1]), at(p))[2]) * lambda[0];
    values1[5] += Number(derivatives(g_0, wrt(p[2], p[2]), at(p))[2]) * lambda[0];
    
    values1[6] += Number(derivatives(g_0, wrt(p[3], p[0]), at(p))[2]) * lambda[0];
    values1[7] += Number(derivatives(g_0, wrt(p[3], p[1]), at(p))[2]) * lambda[0];
    values1[8] += Number(derivatives(g_0, wrt(p[3], p[2]), at(p))[2]) * lambda[0];
    values1[9] += Number(derivatives(g_0, wrt(p[3], p[3]), at(p))[2]) * lambda[0];
    
    values1[10] += Number(derivatives(g_0, wrt(p[4], p[0]), at(p))[2]) * lambda[0];
    values1[11] += Number(derivatives(g_0, wrt(p[4], p[1]), at(p))[2]) * lambda[0];
    values1[12] += Number(derivatives(g_0, wrt(p[4], p[2]), at(p))[2]) * lambda[0];
    values1[13] += Number(derivatives(g_0, wrt(p[4], p[3]), at(p))[2]) * lambda[0];
    values1[14] += Number(derivatives(g_0, wrt(p[4], p[4]), at(p))[2]) * lambda[0];
    
    values1[15] += Number(derivatives(g_0, wrt(p[5], p[0]), at(p))[2]) * lambda[0];
    values1[16] += Number(derivatives(g_0, wrt(p[5], p[1]), at(p))[2]) * lambda[0];
    values1[17] += Number(derivatives(g_0, wrt(p[5], p[2]), at(p))[2]) * lambda[0];
    values1[18] += Number(derivatives(g_0, wrt(p[5], p[3]), at(p))[2]) * lambda[0];
    values1[19] += Number(derivatives(g_0, wrt(p[5], p[4]), at(p))[2]) * lambda[0];
    values1[20] += Number(derivatives(g_0, wrt(p[5], p[5]), at(p))[2]) * lambda[0];
    
    values1[21] += Number(derivatives(g_0, wrt(p[6], p[0]), at(p))[2]) * lambda[0];
    values1[22] += Number(derivatives(g_0, wrt(p[6], p[1]), at(p))[2]) * lambda[0];
    values1[23] += Number(derivatives(g_0, wrt(p[6], p[2]), at(p))[2]) * lambda[0];
    values1[24] += Number(derivatives(g_0, wrt(p[6], p[3]), at(p))[2]) * lambda[0];
    values1[25] += Number(derivatives(g_0, wrt(p[6], p[4]), at(p))[2]) * lambda[0];
    values1[26] += Number(derivatives(g_0, wrt(p[6], p[5]), at(p))[2]) * lambda[0];
    values1[27] += Number(derivatives(g_0, wrt(p[6], p[6]), at(p))[2]) * lambda[0];
    
    values1[28] += Number(derivatives(g_0, wrt(p[7], p[0]), at(p))[2]) * lambda[0];
    values1[29] += Number(derivatives(g_0, wrt(p[7], p[1]), at(p))[2]) * lambda[0];
    values1[30] += Number(derivatives(g_0, wrt(p[7], p[2]), at(p))[2]) * lambda[0];
    values1[31] += Number(derivatives(g_0, wrt(p[7], p[3]), at(p))[2]) * lambda[0];
    values1[32] += Number(derivatives(g_0, wrt(p[7], p[4]), at(p))[2]) * lambda[0];
    values1[33] += Number(derivatives(g_0, wrt(p[7], p[5]), at(p))[2]) * lambda[0];
    values1[34] += Number(derivatives(g_0, wrt(p[7], p[6]), at(p))[2]) * lambda[0];
    values1[35] += Number(derivatives(g_0, wrt(p[7], p[7]), at(p))[2]) * lambda[0];
    
    values1[36] += Number(derivatives(g_0, wrt(p[8], p[0]), at(p))[2]) * lambda[0];
    values1[37] += Number(derivatives(g_0, wrt(p[8], p[1]), at(p))[2]) * lambda[0];
    values1[38] += Number(derivatives(g_0, wrt(p[8], p[2]), at(p))[2]) * lambda[0];
    values1[39] += Number(derivatives(g_0, wrt(p[8], p[3]), at(p))[2]) * lambda[0];
    values1[40] += Number(derivatives(g_0, wrt(p[8], p[4]), at(p))[2]) * lambda[0];
    values1[41] += Number(derivatives(g_0, wrt(p[8], p[5]), at(p))[2]) * lambda[0];
    values1[42] += Number(derivatives(g_0, wrt(p[8], p[6]), at(p))[2]) * lambda[0];
    values1[43] += Number(derivatives(g_0, wrt(p[8], p[7]), at(p))[2]) * lambda[0];
    values1[44] += Number(derivatives(g_0, wrt(p[8], p[8]), at(p))[2]) * lambda[0];

    
    
    
    
    values1[0] += Number(derivatives(g_1, wrt(p[0], p[0]), at(p))[2]) * lambda[1];
    
    values1[1] += Number(derivatives(g_1, wrt(p[1], p[0]), at(p))[2]) * lambda[1];
    values1[2] += Number(derivatives(g_1, wrt(p[1], p[1]), at(p))[2]) * lambda[1];
    
    values1[3] += Number(derivatives(g_1, wrt(p[2], p[0]), at(p))[2]) * lambda[1];
    values1[4] += Number(derivatives(g_1, wrt(p[2], p[1]), at(p))[2]) * lambda[1];
    values1[5] += Number(derivatives(g_1, wrt(p[2], p[2]), at(p))[2]) * lambda[1];
    
    values1[6] += Number(derivatives(g_1, wrt(p[3], p[0]), at(p))[2]) * lambda[1];
    values1[7] += Number(derivatives(g_1, wrt(p[3], p[1]), at(p))[2]) * lambda[1];
    values1[8] += Number(derivatives(g_1, wrt(p[3], p[2]), at(p))[2]) * lambda[1];
    values1[9] += Number(derivatives(g_1, wrt(p[3], p[3]), at(p))[2]) * lambda[1];
    
    values1[10] += Number(derivatives(g_1, wrt(p[4], p[0]), at(p))[2]) * lambda[1];
    values1[11] += Number(derivatives(g_1, wrt(p[4], p[1]), at(p))[2]) * lambda[1];
    values1[12] += Number(derivatives(g_1, wrt(p[4], p[2]), at(p))[2]) * lambda[1];
    values1[13] += Number(derivatives(g_1, wrt(p[4], p[3]), at(p))[2]) * lambda[1];
    values1[14] += Number(derivatives(g_1, wrt(p[4], p[4]), at(p))[2]) * lambda[1];
    
    values1[15] += Number(derivatives(g_1, wrt(p[5], p[0]), at(p))[2]) * lambda[1];
    values1[16] += Number(derivatives(g_1, wrt(p[5], p[1]), at(p))[2]) * lambda[1];
    values1[17] += Number(derivatives(g_1, wrt(p[5], p[2]), at(p))[2]) * lambda[1];
    values1[18] += Number(derivatives(g_1, wrt(p[5], p[3]), at(p))[2]) * lambda[1];
    values1[19] += Number(derivatives(g_1, wrt(p[5], p[4]), at(p))[2]) * lambda[1];
    values1[20] += Number(derivatives(g_1, wrt(p[5], p[5]), at(p))[2]) * lambda[1];
    
    values1[21] += Number(derivatives(g_1, wrt(p[6], p[0]), at(p))[2]) * lambda[1];
    values1[22] += Number(derivatives(g_1, wrt(p[6], p[1]), at(p))[2]) * lambda[1];
    values1[23] += Number(derivatives(g_1, wrt(p[6], p[2]), at(p))[2]) * lambda[1];
    values1[24] += Number(derivatives(g_1, wrt(p[6], p[3]), at(p))[2]) * lambda[1];
    values1[25] += Number(derivatives(g_1, wrt(p[6], p[4]), at(p))[2]) * lambda[1];
    values1[26] += Number(derivatives(g_1, wrt(p[6], p[5]), at(p))[2]) * lambda[1];
    values1[27] += Number(derivatives(g_1, wrt(p[6], p[6]), at(p))[2]) * lambda[1];
    
    values1[28] += Number(derivatives(g_1, wrt(p[7], p[0]), at(p))[2]) * lambda[1];
    values1[29] += Number(derivatives(g_1, wrt(p[7], p[1]), at(p))[2]) * lambda[1];
    values1[30] += Number(derivatives(g_1, wrt(p[7], p[2]), at(p))[2]) * lambda[1];
    values1[31] += Number(derivatives(g_1, wrt(p[7], p[3]), at(p))[2]) * lambda[1];
    values1[32] += Number(derivatives(g_1, wrt(p[7], p[4]), at(p))[2]) * lambda[1];
    values1[33] += Number(derivatives(g_1, wrt(p[7], p[5]), at(p))[2]) * lambda[1];
    values1[34] += Number(derivatives(g_1, wrt(p[7], p[6]), at(p))[2]) * lambda[1];
    values1[35] += Number(derivatives(g_1, wrt(p[7], p[7]), at(p))[2]) * lambda[1];
    
    values1[36] += Number(derivatives(g_1, wrt(p[8], p[0]), at(p))[2]) * lambda[1];
    values1[37] += Number(derivatives(g_1, wrt(p[8], p[1]), at(p))[2]) * lambda[1];
    values1[38] += Number(derivatives(g_1, wrt(p[8], p[2]), at(p))[2]) * lambda[1];
    values1[39] += Number(derivatives(g_1, wrt(p[8], p[3]), at(p))[2]) * lambda[1];
    values1[40] += Number(derivatives(g_1, wrt(p[8], p[4]), at(p))[2]) * lambda[1];
    values1[41] += Number(derivatives(g_1, wrt(p[8], p[5]), at(p))[2]) * lambda[1];
    values1[42] += Number(derivatives(g_1, wrt(p[8], p[6]), at(p))[2]) * lambda[1];
    values1[43] += Number(derivatives(g_1, wrt(p[8], p[7]), at(p))[2]) * lambda[1];
    values1[44] += Number(derivatives(g_1, wrt(p[8], p[8]), at(p))[2]) * lambda[1];

    
    
    
    
    values1[0] += Number(derivatives(g_2, wrt(p[0], p[0]), at(p))[2]) * lambda[2];
    
    values1[1] += Number(derivatives(g_2, wrt(p[1], p[0]), at(p))[2]) * lambda[2];
    values1[2] += Number(derivatives(g_2, wrt(p[1], p[1]), at(p))[2]) * lambda[2];
    
    values1[3] += Number(derivatives(g_2, wrt(p[2], p[0]), at(p))[2]) * lambda[2];
    values1[4] += Number(derivatives(g_2, wrt(p[2], p[1]), at(p))[2]) * lambda[2];
    values1[5] += Number(derivatives(g_2, wrt(p[2], p[2]), at(p))[2]) * lambda[2];
    
    values1[6] += Number(derivatives(g_2, wrt(p[3], p[0]), at(p))[2]) * lambda[2];
    values1[7] += Number(derivatives(g_2, wrt(p[3], p[1]), at(p))[2]) * lambda[2];
    values1[8] += Number(derivatives(g_2, wrt(p[3], p[2]), at(p))[2]) * lambda[2];
    values1[9] += Number(derivatives(g_2, wrt(p[3], p[3]), at(p))[2]) * lambda[2];
    
    values1[10] += Number(derivatives(g_2, wrt(p[4], p[0]), at(p))[2]) * lambda[2];
    values1[11] += Number(derivatives(g_2, wrt(p[4], p[1]), at(p))[2]) * lambda[2];
    values1[12] += Number(derivatives(g_2, wrt(p[4], p[2]), at(p))[2]) * lambda[2];
    values1[13] += Number(derivatives(g_2, wrt(p[4], p[3]), at(p))[2]) * lambda[2];
    values1[14] += Number(derivatives(g_2, wrt(p[4], p[4]), at(p))[2]) * lambda[2];
    
    values1[15] += Number(derivatives(g_2, wrt(p[5], p[0]), at(p))[2]) * lambda[2];
    values1[16] += Number(derivatives(g_2, wrt(p[5], p[1]), at(p))[2]) * lambda[2];
    values1[17] += Number(derivatives(g_2, wrt(p[5], p[2]), at(p))[2]) * lambda[2];
    values1[18] += Number(derivatives(g_2, wrt(p[5], p[3]), at(p))[2]) * lambda[2];
    values1[19] += Number(derivatives(g_2, wrt(p[5], p[4]), at(p))[2]) * lambda[2];
    values1[20] += Number(derivatives(g_2, wrt(p[5], p[5]), at(p))[2]) * lambda[2];
    
    values1[21] += Number(derivatives(g_2, wrt(p[6], p[0]), at(p))[2]) * lambda[2];
    values1[22] += Number(derivatives(g_2, wrt(p[6], p[1]), at(p))[2]) * lambda[2];
    values1[23] += Number(derivatives(g_2, wrt(p[6], p[2]), at(p))[2]) * lambda[2];
    values1[24] += Number(derivatives(g_2, wrt(p[6], p[3]), at(p))[2]) * lambda[2];
    values1[25] += Number(derivatives(g_2, wrt(p[6], p[4]), at(p))[2]) * lambda[2];
    values1[26] += Number(derivatives(g_2, wrt(p[6], p[5]), at(p))[2]) * lambda[2];
    values1[27] += Number(derivatives(g_2, wrt(p[6], p[6]), at(p))[2]) * lambda[2];
    
    values1[28] += Number(derivatives(g_2, wrt(p[7], p[0]), at(p))[2]) * lambda[2];
    values1[29] += Number(derivatives(g_2, wrt(p[7], p[1]), at(p))[2]) * lambda[2];
    values1[30] += Number(derivatives(g_2, wrt(p[7], p[2]), at(p))[2]) * lambda[2];
    values1[31] += Number(derivatives(g_2, wrt(p[7], p[3]), at(p))[2]) * lambda[2];
    values1[32] += Number(derivatives(g_2, wrt(p[7], p[4]), at(p))[2]) * lambda[2];
    values1[33] += Number(derivatives(g_2, wrt(p[7], p[5]), at(p))[2]) * lambda[2];
    values1[34] += Number(derivatives(g_2, wrt(p[7], p[6]), at(p))[2]) * lambda[2];
    values1[35] += Number(derivatives(g_2, wrt(p[7], p[7]), at(p))[2]) * lambda[2];
    
    values1[36] += Number(derivatives(g_2, wrt(p[8], p[0]), at(p))[2]) * lambda[2];
    values1[37] += Number(derivatives(g_2, wrt(p[8], p[1]), at(p))[2]) * lambda[2];
    values1[38] += Number(derivatives(g_2, wrt(p[8], p[2]), at(p))[2]) * lambda[2];
    values1[39] += Number(derivatives(g_2, wrt(p[8], p[3]), at(p))[2]) * lambda[2];
    values1[40] += Number(derivatives(g_2, wrt(p[8], p[4]), at(p))[2]) * lambda[2];
    values1[41] += Number(derivatives(g_2, wrt(p[8], p[5]), at(p))[2]) * lambda[2];
    values1[42] += Number(derivatives(g_2, wrt(p[8], p[6]), at(p))[2]) * lambda[2];
    values1[43] += Number(derivatives(g_2, wrt(p[8], p[7]), at(p))[2]) * lambda[2];
    values1[44] += Number(derivatives(g_2, wrt(p[8], p[8]), at(p))[2]) * lambda[2];

    
    
    
    
    values1[0] += Number(derivatives(g_3, wrt(p[0], p[0]), at(p))[2]) * lambda[3];
    
    values1[1] += Number(derivatives(g_3, wrt(p[1], p[0]), at(p))[2]) * lambda[3];
    values1[2] += Number(derivatives(g_3, wrt(p[1], p[1]), at(p))[2]) * lambda[3];
    
    values1[3] += Number(derivatives(g_3, wrt(p[2], p[0]), at(p))[2]) * lambda[3];
    values1[4] += Number(derivatives(g_3, wrt(p[2], p[1]), at(p))[2]) * lambda[3];
    values1[5] += Number(derivatives(g_3, wrt(p[2], p[2]), at(p))[2]) * lambda[3];
    
    values1[6] += Number(derivatives(g_3, wrt(p[3], p[0]), at(p))[2]) * lambda[3];
    values1[7] += Number(derivatives(g_3, wrt(p[3], p[1]), at(p))[2]) * lambda[3];
    values1[8] += Number(derivatives(g_3, wrt(p[3], p[2]), at(p))[2]) * lambda[3];
    values1[9] += Number(derivatives(g_3, wrt(p[3], p[3]), at(p))[2]) * lambda[3];
    
    values1[10] += Number(derivatives(g_3, wrt(p[4], p[0]), at(p))[2]) * lambda[3];
    values1[11] += Number(derivatives(g_3, wrt(p[4], p[1]), at(p))[2]) * lambda[3];
    values1[12] += Number(derivatives(g_3, wrt(p[4], p[2]), at(p))[2]) * lambda[3];
    values1[13] += Number(derivatives(g_3, wrt(p[4], p[3]), at(p))[2]) * lambda[3];
    values1[14] += Number(derivatives(g_3, wrt(p[4], p[4]), at(p))[2]) * lambda[3];
    
    values1[15] += Number(derivatives(g_3, wrt(p[5], p[0]), at(p))[2]) * lambda[3];
    values1[16] += Number(derivatives(g_3, wrt(p[5], p[1]), at(p))[2]) * lambda[3];
    values1[17] += Number(derivatives(g_3, wrt(p[5], p[2]), at(p))[2]) * lambda[3];
    values1[18] += Number(derivatives(g_3, wrt(p[5], p[3]), at(p))[2]) * lambda[3];
    values1[19] += Number(derivatives(g_3, wrt(p[5], p[4]), at(p))[2]) * lambda[3];
    values1[20] += Number(derivatives(g_3, wrt(p[5], p[5]), at(p))[2]) * lambda[3];
    
    values1[21] += Number(derivatives(g_3, wrt(p[6], p[0]), at(p))[2]) * lambda[3];
    values1[22] += Number(derivatives(g_3, wrt(p[6], p[1]), at(p))[2]) * lambda[3];
    values1[23] += Number(derivatives(g_3, wrt(p[6], p[2]), at(p))[2]) * lambda[3];
    values1[24] += Number(derivatives(g_3, wrt(p[6], p[3]), at(p))[2]) * lambda[3];
    values1[25] += Number(derivatives(g_3, wrt(p[6], p[4]), at(p))[2]) * lambda[3];
    values1[26] += Number(derivatives(g_3, wrt(p[6], p[5]), at(p))[2]) * lambda[3];
    values1[27] += Number(derivatives(g_3, wrt(p[6], p[6]), at(p))[2]) * lambda[3];
    
    values1[28] += Number(derivatives(g_3, wrt(p[7], p[0]), at(p))[2]) * lambda[3];
    values1[29] += Number(derivatives(g_3, wrt(p[7], p[1]), at(p))[2]) * lambda[3];
    values1[30] += Number(derivatives(g_3, wrt(p[7], p[2]), at(p))[2]) * lambda[3];
    values1[31] += Number(derivatives(g_3, wrt(p[7], p[3]), at(p))[2]) * lambda[3];
    values1[32] += Number(derivatives(g_3, wrt(p[7], p[4]), at(p))[2]) * lambda[3];
    values1[33] += Number(derivatives(g_3, wrt(p[7], p[5]), at(p))[2]) * lambda[3];
    values1[34] += Number(derivatives(g_3, wrt(p[7], p[6]), at(p))[2]) * lambda[3];
    values1[35] += Number(derivatives(g_3, wrt(p[7], p[7]), at(p))[2]) * lambda[3];
    
    values1[36] += Number(derivatives(g_3, wrt(p[8], p[0]), at(p))[2]) * lambda[3];
    values1[37] += Number(derivatives(g_3, wrt(p[8], p[1]), at(p))[2]) * lambda[3];
    values1[38] += Number(derivatives(g_3, wrt(p[8], p[2]), at(p))[2]) * lambda[3];
    values1[39] += Number(derivatives(g_3, wrt(p[8], p[3]), at(p))[2]) * lambda[3];
    values1[40] += Number(derivatives(g_3, wrt(p[8], p[4]), at(p))[2]) * lambda[3];
    values1[41] += Number(derivatives(g_3, wrt(p[8], p[5]), at(p))[2]) * lambda[3];
    values1[42] += Number(derivatives(g_3, wrt(p[8], p[6]), at(p))[2]) * lambda[3];
    values1[43] += Number(derivatives(g_3, wrt(p[8], p[7]), at(p))[2]) * lambda[3];
    values1[44] += Number(derivatives(g_3, wrt(p[8], p[8]), at(p))[2]) * lambda[3];

    
    
    
    
    values1[0] += Number(derivatives(g_4, wrt(p[0], p[0]), at(p))[2]) * lambda[4];
    
    values1[1] += Number(derivatives(g_4, wrt(p[1], p[0]), at(p))[2]) * lambda[4];
    values1[2] += Number(derivatives(g_4, wrt(p[1], p[1]), at(p))[2]) * lambda[4];
    
    values1[3] += Number(derivatives(g_4, wrt(p[2], p[0]), at(p))[2]) * lambda[4];
    values1[4] += Number(derivatives(g_4, wrt(p[2], p[1]), at(p))[2]) * lambda[4];
    values1[5] += Number(derivatives(g_4, wrt(p[2], p[2]), at(p))[2]) * lambda[4];
    
    values1[6] += Number(derivatives(g_4, wrt(p[3], p[0]), at(p))[2]) * lambda[4];
    values1[7] += Number(derivatives(g_4, wrt(p[3], p[1]), at(p))[2]) * lambda[4];
    values1[8] += Number(derivatives(g_4, wrt(p[3], p[2]), at(p))[2]) * lambda[4];
    values1[9] += Number(derivatives(g_4, wrt(p[3], p[3]), at(p))[2]) * lambda[4];
    
    values1[10] += Number(derivatives(g_4, wrt(p[4], p[0]), at(p))[2]) * lambda[4];
    values1[11] += Number(derivatives(g_4, wrt(p[4], p[1]), at(p))[2]) * lambda[4];
    values1[12] += Number(derivatives(g_4, wrt(p[4], p[2]), at(p))[2]) * lambda[4];
    values1[13] += Number(derivatives(g_4, wrt(p[4], p[3]), at(p))[2]) * lambda[4];
    values1[14] += Number(derivatives(g_4, wrt(p[4], p[4]), at(p))[2]) * lambda[4];
    
    values1[15] += Number(derivatives(g_4, wrt(p[5], p[0]), at(p))[2]) * lambda[4];
    values1[16] += Number(derivatives(g_4, wrt(p[5], p[1]), at(p))[2]) * lambda[4];
    values1[17] += Number(derivatives(g_4, wrt(p[5], p[2]), at(p))[2]) * lambda[4];
    values1[18] += Number(derivatives(g_4, wrt(p[5], p[3]), at(p))[2]) * lambda[4];
    values1[19] += Number(derivatives(g_4, wrt(p[5], p[4]), at(p))[2]) * lambda[4];
    values1[20] += Number(derivatives(g_4, wrt(p[5], p[5]), at(p))[2]) * lambda[4];
    
    values1[21] += Number(derivatives(g_4, wrt(p[6], p[0]), at(p))[2]) * lambda[4];
    values1[22] += Number(derivatives(g_4, wrt(p[6], p[1]), at(p))[2]) * lambda[4];
    values1[23] += Number(derivatives(g_4, wrt(p[6], p[2]), at(p))[2]) * lambda[4];
    values1[24] += Number(derivatives(g_4, wrt(p[6], p[3]), at(p))[2]) * lambda[4];
    values1[25] += Number(derivatives(g_4, wrt(p[6], p[4]), at(p))[2]) * lambda[4];
    values1[26] += Number(derivatives(g_4, wrt(p[6], p[5]), at(p))[2]) * lambda[4];
    values1[27] += Number(derivatives(g_4, wrt(p[6], p[6]), at(p))[2]) * lambda[4];
    
    values1[28] += Number(derivatives(g_4, wrt(p[7], p[0]), at(p))[2]) * lambda[4];
    values1[29] += Number(derivatives(g_4, wrt(p[7], p[1]), at(p))[2]) * lambda[4];
    values1[30] += Number(derivatives(g_4, wrt(p[7], p[2]), at(p))[2]) * lambda[4];
    values1[31] += Number(derivatives(g_4, wrt(p[7], p[3]), at(p))[2]) * lambda[4];
    values1[32] += Number(derivatives(g_4, wrt(p[7], p[4]), at(p))[2]) * lambda[4];
    values1[33] += Number(derivatives(g_4, wrt(p[7], p[5]), at(p))[2]) * lambda[4];
    values1[34] += Number(derivatives(g_4, wrt(p[7], p[6]), at(p))[2]) * lambda[4];
    values1[35] += Number(derivatives(g_4, wrt(p[7], p[7]), at(p))[2]) * lambda[4];
    
    values1[36] += Number(derivatives(g_4, wrt(p[8], p[0]), at(p))[2]) * lambda[4];
    values1[37] += Number(derivatives(g_4, wrt(p[8], p[1]), at(p))[2]) * lambda[4];
    values1[38] += Number(derivatives(g_4, wrt(p[8], p[2]), at(p))[2]) * lambda[4];
    values1[39] += Number(derivatives(g_4, wrt(p[8], p[3]), at(p))[2]) * lambda[4];
    values1[40] += Number(derivatives(g_4, wrt(p[8], p[4]), at(p))[2]) * lambda[4];
    values1[41] += Number(derivatives(g_4, wrt(p[8], p[5]), at(p))[2]) * lambda[4];
    values1[42] += Number(derivatives(g_4, wrt(p[8], p[6]), at(p))[2]) * lambda[4];
    values1[43] += Number(derivatives(g_4, wrt(p[8], p[7]), at(p))[2]) * lambda[4];
    values1[44] += Number(derivatives(g_4, wrt(p[8], p[8]), at(p))[2]) * lambda[4];

    
    
    
    
    values1[0] += Number(derivatives(g_5, wrt(p[0], p[0]), at(p))[2]) * lambda[5];
    
    values1[1] += Number(derivatives(g_5, wrt(p[1], p[0]), at(p))[2]) * lambda[5];
    values1[2] += Number(derivatives(g_5, wrt(p[1], p[1]), at(p))[2]) * lambda[5];
    
    values1[3] += Number(derivatives(g_5, wrt(p[2], p[0]), at(p))[2]) * lambda[5];
    values1[4] += Number(derivatives(g_5, wrt(p[2], p[1]), at(p))[2]) * lambda[5];
    values1[5] += Number(derivatives(g_5, wrt(p[2], p[2]), at(p))[2]) * lambda[5];
    
    values1[6] += Number(derivatives(g_5, wrt(p[3], p[0]), at(p))[2]) * lambda[5];
    values1[7] += Number(derivatives(g_5, wrt(p[3], p[1]), at(p))[2]) * lambda[5];
    values1[8] += Number(derivatives(g_5, wrt(p[3], p[2]), at(p))[2]) * lambda[5];
    values1[9] += Number(derivatives(g_5, wrt(p[3], p[3]), at(p))[2]) * lambda[5];
    
    values1[10] += Number(derivatives(g_5, wrt(p[4], p[0]), at(p))[2]) * lambda[5];
    values1[11] += Number(derivatives(g_5, wrt(p[4], p[1]), at(p))[2]) * lambda[5];
    values1[12] += Number(derivatives(g_5, wrt(p[4], p[2]), at(p))[2]) * lambda[5];
    values1[13] += Number(derivatives(g_5, wrt(p[4], p[3]), at(p))[2]) * lambda[5];
    values1[14] += Number(derivatives(g_5, wrt(p[4], p[4]), at(p))[2]) * lambda[5];
    
    values1[15] += Number(derivatives(g_5, wrt(p[5], p[0]), at(p))[2]) * lambda[5];
    values1[16] += Number(derivatives(g_5, wrt(p[5], p[1]), at(p))[2]) * lambda[5];
    values1[17] += Number(derivatives(g_5, wrt(p[5], p[2]), at(p))[2]) * lambda[5];
    values1[18] += Number(derivatives(g_5, wrt(p[5], p[3]), at(p))[2]) * lambda[5];
    values1[19] += Number(derivatives(g_5, wrt(p[5], p[4]), at(p))[2]) * lambda[5];
    values1[20] += Number(derivatives(g_5, wrt(p[5], p[5]), at(p))[2]) * lambda[5];
    
    values1[21] += Number(derivatives(g_5, wrt(p[6], p[0]), at(p))[2]) * lambda[5];
    values1[22] += Number(derivatives(g_5, wrt(p[6], p[1]), at(p))[2]) * lambda[5];
    values1[23] += Number(derivatives(g_5, wrt(p[6], p[2]), at(p))[2]) * lambda[5];
    values1[24] += Number(derivatives(g_5, wrt(p[6], p[3]), at(p))[2]) * lambda[5];
    values1[25] += Number(derivatives(g_5, wrt(p[6], p[4]), at(p))[2]) * lambda[5];
    values1[26] += Number(derivatives(g_5, wrt(p[6], p[5]), at(p))[2]) * lambda[5];
    values1[27] += Number(derivatives(g_5, wrt(p[6], p[6]), at(p))[2]) * lambda[5];
    
    values1[28] += Number(derivatives(g_5, wrt(p[7], p[0]), at(p))[2]) * lambda[5];
    values1[29] += Number(derivatives(g_5, wrt(p[7], p[1]), at(p))[2]) * lambda[5];
    values1[30] += Number(derivatives(g_5, wrt(p[7], p[2]), at(p))[2]) * lambda[5];
    values1[31] += Number(derivatives(g_5, wrt(p[7], p[3]), at(p))[2]) * lambda[5];
    values1[32] += Number(derivatives(g_5, wrt(p[7], p[4]), at(p))[2]) * lambda[5];
    values1[33] += Number(derivatives(g_5, wrt(p[7], p[5]), at(p))[2]) * lambda[5];
    values1[34] += Number(derivatives(g_5, wrt(p[7], p[6]), at(p))[2]) * lambda[5];
    values1[35] += Number(derivatives(g_5, wrt(p[7], p[7]), at(p))[2]) * lambda[5];
    
    values1[36] += Number(derivatives(g_5, wrt(p[8], p[0]), at(p))[2]) * lambda[5];
    values1[37] += Number(derivatives(g_5, wrt(p[8], p[1]), at(p))[2]) * lambda[5];
    values1[38] += Number(derivatives(g_5, wrt(p[8], p[2]), at(p))[2]) * lambda[5];
    values1[39] += Number(derivatives(g_5, wrt(p[8], p[3]), at(p))[2]) * lambda[5];
    values1[40] += Number(derivatives(g_5, wrt(p[8], p[4]), at(p))[2]) * lambda[5];
    values1[41] += Number(derivatives(g_5, wrt(p[8], p[5]), at(p))[2]) * lambda[5];
    values1[42] += Number(derivatives(g_5, wrt(p[8], p[6]), at(p))[2]) * lambda[5];
    values1[43] += Number(derivatives(g_5, wrt(p[8], p[7]), at(p))[2]) * lambda[5];
    values1[44] += Number(derivatives(g_5, wrt(p[8], p[8]), at(p))[2]) * lambda[5];

    
    
    
    
    values1[0] += Number(derivatives(g_6, wrt(p[0], p[0]), at(p))[2]) * lambda[6];
    
    values1[1] += Number(derivatives(g_6, wrt(p[1], p[0]), at(p))[2]) * lambda[6];
    values1[2] += Number(derivatives(g_6, wrt(p[1], p[1]), at(p))[2]) * lambda[6];
    
    values1[3] += Number(derivatives(g_6, wrt(p[2], p[0]), at(p))[2]) * lambda[6];
    values1[4] += Number(derivatives(g_6, wrt(p[2], p[1]), at(p))[2]) * lambda[6];
    values1[5] += Number(derivatives(g_6, wrt(p[2], p[2]), at(p))[2]) * lambda[6];
    
    values1[6] += Number(derivatives(g_6, wrt(p[3], p[0]), at(p))[2]) * lambda[6];
    values1[7] += Number(derivatives(g_6, wrt(p[3], p[1]), at(p))[2]) * lambda[6];
    values1[8] += Number(derivatives(g_6, wrt(p[3], p[2]), at(p))[2]) * lambda[6];
    values1[9] += Number(derivatives(g_6, wrt(p[3], p[3]), at(p))[2]) * lambda[6];
    
    values1[10] += Number(derivatives(g_6, wrt(p[4], p[0]), at(p))[2]) * lambda[6];
    values1[11] += Number(derivatives(g_6, wrt(p[4], p[1]), at(p))[2]) * lambda[6];
    values1[12] += Number(derivatives(g_6, wrt(p[4], p[2]), at(p))[2]) * lambda[6];
    values1[13] += Number(derivatives(g_6, wrt(p[4], p[3]), at(p))[2]) * lambda[6];
    values1[14] += Number(derivatives(g_6, wrt(p[4], p[4]), at(p))[2]) * lambda[6];
    
    values1[15] += Number(derivatives(g_6, wrt(p[5], p[0]), at(p))[2]) * lambda[6];
    values1[16] += Number(derivatives(g_6, wrt(p[5], p[1]), at(p))[2]) * lambda[6];
    values1[17] += Number(derivatives(g_6, wrt(p[5], p[2]), at(p))[2]) * lambda[6];
    values1[18] += Number(derivatives(g_6, wrt(p[5], p[3]), at(p))[2]) * lambda[6];
    values1[19] += Number(derivatives(g_6, wrt(p[5], p[4]), at(p))[2]) * lambda[6];
    values1[20] += Number(derivatives(g_6, wrt(p[5], p[5]), at(p))[2]) * lambda[6];
    
    values1[21] += Number(derivatives(g_6, wrt(p[6], p[0]), at(p))[2]) * lambda[6];
    values1[22] += Number(derivatives(g_6, wrt(p[6], p[1]), at(p))[2]) * lambda[6];
    values1[23] += Number(derivatives(g_6, wrt(p[6], p[2]), at(p))[2]) * lambda[6];
    values1[24] += Number(derivatives(g_6, wrt(p[6], p[3]), at(p))[2]) * lambda[6];
    values1[25] += Number(derivatives(g_6, wrt(p[6], p[4]), at(p))[2]) * lambda[6];
    values1[26] += Number(derivatives(g_6, wrt(p[6], p[5]), at(p))[2]) * lambda[6];
    values1[27] += Number(derivatives(g_6, wrt(p[6], p[6]), at(p))[2]) * lambda[6];
    
    values1[28] += Number(derivatives(g_6, wrt(p[7], p[0]), at(p))[2]) * lambda[6];
    values1[29] += Number(derivatives(g_6, wrt(p[7], p[1]), at(p))[2]) * lambda[6];
    values1[30] += Number(derivatives(g_6, wrt(p[7], p[2]), at(p))[2]) * lambda[6];
    values1[31] += Number(derivatives(g_6, wrt(p[7], p[3]), at(p))[2]) * lambda[6];
    values1[32] += Number(derivatives(g_6, wrt(p[7], p[4]), at(p))[2]) * lambda[6];
    values1[33] += Number(derivatives(g_6, wrt(p[7], p[5]), at(p))[2]) * lambda[6];
    values1[34] += Number(derivatives(g_6, wrt(p[7], p[6]), at(p))[2]) * lambda[6];
    values1[35] += Number(derivatives(g_6, wrt(p[7], p[7]), at(p))[2]) * lambda[6];
    
    values1[36] += Number(derivatives(g_6, wrt(p[8], p[0]), at(p))[2]) * lambda[6];
    values1[37] += Number(derivatives(g_6, wrt(p[8], p[1]), at(p))[2]) * lambda[6];
    values1[38] += Number(derivatives(g_6, wrt(p[8], p[2]), at(p))[2]) * lambda[6];
    values1[39] += Number(derivatives(g_6, wrt(p[8], p[3]), at(p))[2]) * lambda[6];
    values1[40] += Number(derivatives(g_6, wrt(p[8], p[4]), at(p))[2]) * lambda[6];
    values1[41] += Number(derivatives(g_6, wrt(p[8], p[5]), at(p))[2]) * lambda[6];
    values1[42] += Number(derivatives(g_6, wrt(p[8], p[6]), at(p))[2]) * lambda[6];
    values1[43] += Number(derivatives(g_6, wrt(p[8], p[7]), at(p))[2]) * lambda[6];
    values1[44] += Number(derivatives(g_6, wrt(p[8], p[8]), at(p))[2]) * lambda[6];

    
    
    
    
    values1[0] += Number(derivatives(g_7, wrt(p[0], p[0]), at(p))[2]) * lambda[7];
    
    values1[1] += Number(derivatives(g_7, wrt(p[1], p[0]), at(p))[2]) * lambda[7];
    values1[2] += Number(derivatives(g_7, wrt(p[1], p[1]), at(p))[2]) * lambda[7];
    
    values1[3] += Number(derivatives(g_7, wrt(p[2], p[0]), at(p))[2]) * lambda[7];
    values1[4] += Number(derivatives(g_7, wrt(p[2], p[1]), at(p))[2]) * lambda[7];
    values1[5] += Number(derivatives(g_7, wrt(p[2], p[2]), at(p))[2]) * lambda[7];
    
    values1[6] += Number(derivatives(g_7, wrt(p[3], p[0]), at(p))[2]) * lambda[7];
    values1[7] += Number(derivatives(g_7, wrt(p[3], p[1]), at(p))[2]) * lambda[7];
    values1[8] += Number(derivatives(g_7, wrt(p[3], p[2]), at(p))[2]) * lambda[7];
    values1[9] += Number(derivatives(g_7, wrt(p[3], p[3]), at(p))[2]) * lambda[7];
    
    values1[10] += Number(derivatives(g_7, wrt(p[4], p[0]), at(p))[2]) * lambda[7];
    values1[11] += Number(derivatives(g_7, wrt(p[4], p[1]), at(p))[2]) * lambda[7];
    values1[12] += Number(derivatives(g_7, wrt(p[4], p[2]), at(p))[2]) * lambda[7];
    values1[13] += Number(derivatives(g_7, wrt(p[4], p[3]), at(p))[2]) * lambda[7];
    values1[14] += Number(derivatives(g_7, wrt(p[4], p[4]), at(p))[2]) * lambda[7];
    
    values1[15] += Number(derivatives(g_7, wrt(p[5], p[0]), at(p))[2]) * lambda[7];
    values1[16] += Number(derivatives(g_7, wrt(p[5], p[1]), at(p))[2]) * lambda[7];
    values1[17] += Number(derivatives(g_7, wrt(p[5], p[2]), at(p))[2]) * lambda[7];
    values1[18] += Number(derivatives(g_7, wrt(p[5], p[3]), at(p))[2]) * lambda[7];
    values1[19] += Number(derivatives(g_7, wrt(p[5], p[4]), at(p))[2]) * lambda[7];
    values1[20] += Number(derivatives(g_7, wrt(p[5], p[5]), at(p))[2]) * lambda[7];
    
    values1[21] += Number(derivatives(g_7, wrt(p[6], p[0]), at(p))[2]) * lambda[7];
    values1[22] += Number(derivatives(g_7, wrt(p[6], p[1]), at(p))[2]) * lambda[7];
    values1[23] += Number(derivatives(g_7, wrt(p[6], p[2]), at(p))[2]) * lambda[7];
    values1[24] += Number(derivatives(g_7, wrt(p[6], p[3]), at(p))[2]) * lambda[7];
    values1[25] += Number(derivatives(g_7, wrt(p[6], p[4]), at(p))[2]) * lambda[7];
    values1[26] += Number(derivatives(g_7, wrt(p[6], p[5]), at(p))[2]) * lambda[7];
    values1[27] += Number(derivatives(g_7, wrt(p[6], p[6]), at(p))[2]) * lambda[7];
    
    values1[28] += Number(derivatives(g_7, wrt(p[7], p[0]), at(p))[2]) * lambda[7];
    values1[29] += Number(derivatives(g_7, wrt(p[7], p[1]), at(p))[2]) * lambda[7];
    values1[30] += Number(derivatives(g_7, wrt(p[7], p[2]), at(p))[2]) * lambda[7];
    values1[31] += Number(derivatives(g_7, wrt(p[7], p[3]), at(p))[2]) * lambda[7];
    values1[32] += Number(derivatives(g_7, wrt(p[7], p[4]), at(p))[2]) * lambda[7];
    values1[33] += Number(derivatives(g_7, wrt(p[7], p[5]), at(p))[2]) * lambda[7];
    values1[34] += Number(derivatives(g_7, wrt(p[7], p[6]), at(p))[2]) * lambda[7];
    values1[35] += Number(derivatives(g_7, wrt(p[7], p[7]), at(p))[2]) * lambda[7];
    
    values1[36] += Number(derivatives(g_7, wrt(p[8], p[0]), at(p))[2]) * lambda[7];
    values1[37] += Number(derivatives(g_7, wrt(p[8], p[1]), at(p))[2]) * lambda[7];
    values1[38] += Number(derivatives(g_7, wrt(p[8], p[2]), at(p))[2]) * lambda[7];
    values1[39] += Number(derivatives(g_7, wrt(p[8], p[3]), at(p))[2]) * lambda[7];
    values1[40] += Number(derivatives(g_7, wrt(p[8], p[4]), at(p))[2]) * lambda[7];
    values1[41] += Number(derivatives(g_7, wrt(p[8], p[5]), at(p))[2]) * lambda[7];
    values1[42] += Number(derivatives(g_7, wrt(p[8], p[6]), at(p))[2]) * lambda[7];
    values1[43] += Number(derivatives(g_7, wrt(p[8], p[7]), at(p))[2]) * lambda[7];
    values1[44] += Number(derivatives(g_7, wrt(p[8], p[8]), at(p))[2]) * lambda[7];

    
    
    
    
    values1[0] += Number(derivatives(g_8, wrt(p[0], p[0]), at(p))[2]) * lambda[8];
    
    values1[1] += Number(derivatives(g_8, wrt(p[1], p[0]), at(p))[2]) * lambda[8];
    values1[2] += Number(derivatives(g_8, wrt(p[1], p[1]), at(p))[2]) * lambda[8];
    
    values1[3] += Number(derivatives(g_8, wrt(p[2], p[0]), at(p))[2]) * lambda[8];
    values1[4] += Number(derivatives(g_8, wrt(p[2], p[1]), at(p))[2]) * lambda[8];
    values1[5] += Number(derivatives(g_8, wrt(p[2], p[2]), at(p))[2]) * lambda[8];
    
    values1[6] += Number(derivatives(g_8, wrt(p[3], p[0]), at(p))[2]) * lambda[8];
    values1[7] += Number(derivatives(g_8, wrt(p[3], p[1]), at(p))[2]) * lambda[8];
    values1[8] += Number(derivatives(g_8, wrt(p[3], p[2]), at(p))[2]) * lambda[8];
    values1[9] += Number(derivatives(g_8, wrt(p[3], p[3]), at(p))[2]) * lambda[8];
    
    values1[10] += Number(derivatives(g_8, wrt(p[4], p[0]), at(p))[2]) * lambda[8];
    values1[11] += Number(derivatives(g_8, wrt(p[4], p[1]), at(p))[2]) * lambda[8];
    values1[12] += Number(derivatives(g_8, wrt(p[4], p[2]), at(p))[2]) * lambda[8];
    values1[13] += Number(derivatives(g_8, wrt(p[4], p[3]), at(p))[2]) * lambda[8];
    values1[14] += Number(derivatives(g_8, wrt(p[4], p[4]), at(p))[2]) * lambda[8];
    
    values1[15] += Number(derivatives(g_8, wrt(p[5], p[0]), at(p))[2]) * lambda[8];
    values1[16] += Number(derivatives(g_8, wrt(p[5], p[1]), at(p))[2]) * lambda[8];
    values1[17] += Number(derivatives(g_8, wrt(p[5], p[2]), at(p))[2]) * lambda[8];
    values1[18] += Number(derivatives(g_8, wrt(p[5], p[3]), at(p))[2]) * lambda[8];
    values1[19] += Number(derivatives(g_8, wrt(p[5], p[4]), at(p))[2]) * lambda[8];
    values1[20] += Number(derivatives(g_8, wrt(p[5], p[5]), at(p))[2]) * lambda[8];
    
    values1[21] += Number(derivatives(g_8, wrt(p[6], p[0]), at(p))[2]) * lambda[8];
    values1[22] += Number(derivatives(g_8, wrt(p[6], p[1]), at(p))[2]) * lambda[8];
    values1[23] += Number(derivatives(g_8, wrt(p[6], p[2]), at(p))[2]) * lambda[8];
    values1[24] += Number(derivatives(g_8, wrt(p[6], p[3]), at(p))[2]) * lambda[8];
    values1[25] += Number(derivatives(g_8, wrt(p[6], p[4]), at(p))[2]) * lambda[8];
    values1[26] += Number(derivatives(g_8, wrt(p[6], p[5]), at(p))[2]) * lambda[8];
    values1[27] += Number(derivatives(g_8, wrt(p[6], p[6]), at(p))[2]) * lambda[8];
    
    values1[28] += Number(derivatives(g_8, wrt(p[7], p[0]), at(p))[2]) * lambda[8];
    values1[29] += Number(derivatives(g_8, wrt(p[7], p[1]), at(p))[2]) * lambda[8];
    values1[30] += Number(derivatives(g_8, wrt(p[7], p[2]), at(p))[2]) * lambda[8];
    values1[31] += Number(derivatives(g_8, wrt(p[7], p[3]), at(p))[2]) * lambda[8];
    values1[32] += Number(derivatives(g_8, wrt(p[7], p[4]), at(p))[2]) * lambda[8];
    values1[33] += Number(derivatives(g_8, wrt(p[7], p[5]), at(p))[2]) * lambda[8];
    values1[34] += Number(derivatives(g_8, wrt(p[7], p[6]), at(p))[2]) * lambda[8];
    values1[35] += Number(derivatives(g_8, wrt(p[7], p[7]), at(p))[2]) * lambda[8];
    
    values1[36] += Number(derivatives(g_8, wrt(p[8], p[0]), at(p))[2]) * lambda[8];
    values1[37] += Number(derivatives(g_8, wrt(p[8], p[1]), at(p))[2]) * lambda[8];
    values1[38] += Number(derivatives(g_8, wrt(p[8], p[2]), at(p))[2]) * lambda[8];
    values1[39] += Number(derivatives(g_8, wrt(p[8], p[3]), at(p))[2]) * lambda[8];
    values1[40] += Number(derivatives(g_8, wrt(p[8], p[4]), at(p))[2]) * lambda[8];
    values1[41] += Number(derivatives(g_8, wrt(p[8], p[5]), at(p))[2]) * lambda[8];
    values1[42] += Number(derivatives(g_8, wrt(p[8], p[6]), at(p))[2]) * lambda[8];
    values1[43] += Number(derivatives(g_8, wrt(p[8], p[7]), at(p))[2]) * lambda[8];
    values1[44] += Number(derivatives(g_8, wrt(p[8], p[8]), at(p))[2]) * lambda[8];

    
    
    
    
    values1[0] += Number(derivatives(g_9, wrt(p[0], p[0]), at(p))[2]) * lambda[9];
    
    values1[1] += Number(derivatives(g_9, wrt(p[1], p[0]), at(p))[2]) * lambda[9];
    values1[2] += Number(derivatives(g_9, wrt(p[1], p[1]), at(p))[2]) * lambda[9];
    
    values1[3] += Number(derivatives(g_9, wrt(p[2], p[0]), at(p))[2]) * lambda[9];
    values1[4] += Number(derivatives(g_9, wrt(p[2], p[1]), at(p))[2]) * lambda[9];
    values1[5] += Number(derivatives(g_9, wrt(p[2], p[2]), at(p))[2]) * lambda[9];
    
    values1[6] += Number(derivatives(g_9, wrt(p[3], p[0]), at(p))[2]) * lambda[9];
    values1[7] += Number(derivatives(g_9, wrt(p[3], p[1]), at(p))[2]) * lambda[9];
    values1[8] += Number(derivatives(g_9, wrt(p[3], p[2]), at(p))[2]) * lambda[9];
    values1[9] += Number(derivatives(g_9, wrt(p[3], p[3]), at(p))[2]) * lambda[9];
    
    values1[10] += Number(derivatives(g_9, wrt(p[4], p[0]), at(p))[2]) * lambda[9];
    values1[11] += Number(derivatives(g_9, wrt(p[4], p[1]), at(p))[2]) * lambda[9];
    values1[12] += Number(derivatives(g_9, wrt(p[4], p[2]), at(p))[2]) * lambda[9];
    values1[13] += Number(derivatives(g_9, wrt(p[4], p[3]), at(p))[2]) * lambda[9];
    values1[14] += Number(derivatives(g_9, wrt(p[4], p[4]), at(p))[2]) * lambda[9];
    
    values1[15] += Number(derivatives(g_9, wrt(p[5], p[0]), at(p))[2]) * lambda[9];
    values1[16] += Number(derivatives(g_9, wrt(p[5], p[1]), at(p))[2]) * lambda[9];
    values1[17] += Number(derivatives(g_9, wrt(p[5], p[2]), at(p))[2]) * lambda[9];
    values1[18] += Number(derivatives(g_9, wrt(p[5], p[3]), at(p))[2]) * lambda[9];
    values1[19] += Number(derivatives(g_9, wrt(p[5], p[4]), at(p))[2]) * lambda[9];
    values1[20] += Number(derivatives(g_9, wrt(p[5], p[5]), at(p))[2]) * lambda[9];
    
    values1[21] += Number(derivatives(g_9, wrt(p[6], p[0]), at(p))[2]) * lambda[9];
    values1[22] += Number(derivatives(g_9, wrt(p[6], p[1]), at(p))[2]) * lambda[9];
    values1[23] += Number(derivatives(g_9, wrt(p[6], p[2]), at(p))[2]) * lambda[9];
    values1[24] += Number(derivatives(g_9, wrt(p[6], p[3]), at(p))[2]) * lambda[9];
    values1[25] += Number(derivatives(g_9, wrt(p[6], p[4]), at(p))[2]) * lambda[9];
    values1[26] += Number(derivatives(g_9, wrt(p[6], p[5]), at(p))[2]) * lambda[9];
    values1[27] += Number(derivatives(g_9, wrt(p[6], p[6]), at(p))[2]) * lambda[9];
    
    values1[28] += Number(derivatives(g_9, wrt(p[7], p[0]), at(p))[2]) * lambda[9];
    values1[29] += Number(derivatives(g_9, wrt(p[7], p[1]), at(p))[2]) * lambda[9];
    values1[30] += Number(derivatives(g_9, wrt(p[7], p[2]), at(p))[2]) * lambda[9];
    values1[31] += Number(derivatives(g_9, wrt(p[7], p[3]), at(p))[2]) * lambda[9];
    values1[32] += Number(derivatives(g_9, wrt(p[7], p[4]), at(p))[2]) * lambda[9];
    values1[33] += Number(derivatives(g_9, wrt(p[7], p[5]), at(p))[2]) * lambda[9];
    values1[34] += Number(derivatives(g_9, wrt(p[7], p[6]), at(p))[2]) * lambda[9];
    values1[35] += Number(derivatives(g_9, wrt(p[7], p[7]), at(p))[2]) * lambda[9];
    
    values1[36] += Number(derivatives(g_9, wrt(p[8], p[0]), at(p))[2]) * lambda[9];
    values1[37] += Number(derivatives(g_9, wrt(p[8], p[1]), at(p))[2]) * lambda[9];
    values1[38] += Number(derivatives(g_9, wrt(p[8], p[2]), at(p))[2]) * lambda[9];
    values1[39] += Number(derivatives(g_9, wrt(p[8], p[3]), at(p))[2]) * lambda[9];
    values1[40] += Number(derivatives(g_9, wrt(p[8], p[4]), at(p))[2]) * lambda[9];
    values1[41] += Number(derivatives(g_9, wrt(p[8], p[5]), at(p))[2]) * lambda[9];
    values1[42] += Number(derivatives(g_9, wrt(p[8], p[6]), at(p))[2]) * lambda[9];
    values1[43] += Number(derivatives(g_9, wrt(p[8], p[7]), at(p))[2]) * lambda[9];
    values1[44] += Number(derivatives(g_9, wrt(p[8], p[8]), at(p))[2]) * lambda[9];

    
    
    
    
    values1[0] += Number(derivatives(g_10, wrt(p[0], p[0]), at(p))[2]) * lambda[10];
    
    values1[1] += Number(derivatives(g_10, wrt(p[1], p[0]), at(p))[2]) * lambda[10];
    values1[2] += Number(derivatives(g_10, wrt(p[1], p[1]), at(p))[2]) * lambda[10];
    
    values1[3] += Number(derivatives(g_10, wrt(p[2], p[0]), at(p))[2]) * lambda[10];
    values1[4] += Number(derivatives(g_10, wrt(p[2], p[1]), at(p))[2]) * lambda[10];
    values1[5] += Number(derivatives(g_10, wrt(p[2], p[2]), at(p))[2]) * lambda[10];
    
    values1[6] += Number(derivatives(g_10, wrt(p[3], p[0]), at(p))[2]) * lambda[10];
    values1[7] += Number(derivatives(g_10, wrt(p[3], p[1]), at(p))[2]) * lambda[10];
    values1[8] += Number(derivatives(g_10, wrt(p[3], p[2]), at(p))[2]) * lambda[10];
    values1[9] += Number(derivatives(g_10, wrt(p[3], p[3]), at(p))[2]) * lambda[10];
    
    values1[10] += Number(derivatives(g_10, wrt(p[4], p[0]), at(p))[2]) * lambda[10];
    values1[11] += Number(derivatives(g_10, wrt(p[4], p[1]), at(p))[2]) * lambda[10];
    values1[12] += Number(derivatives(g_10, wrt(p[4], p[2]), at(p))[2]) * lambda[10];
    values1[13] += Number(derivatives(g_10, wrt(p[4], p[3]), at(p))[2]) * lambda[10];
    values1[14] += Number(derivatives(g_10, wrt(p[4], p[4]), at(p))[2]) * lambda[10];
    
    values1[15] += Number(derivatives(g_10, wrt(p[5], p[0]), at(p))[2]) * lambda[10];
    values1[16] += Number(derivatives(g_10, wrt(p[5], p[1]), at(p))[2]) * lambda[10];
    values1[17] += Number(derivatives(g_10, wrt(p[5], p[2]), at(p))[2]) * lambda[10];
    values1[18] += Number(derivatives(g_10, wrt(p[5], p[3]), at(p))[2]) * lambda[10];
    values1[19] += Number(derivatives(g_10, wrt(p[5], p[4]), at(p))[2]) * lambda[10];
    values1[20] += Number(derivatives(g_10, wrt(p[5], p[5]), at(p))[2]) * lambda[10];
    
    values1[21] += Number(derivatives(g_10, wrt(p[6], p[0]), at(p))[2]) * lambda[10];
    values1[22] += Number(derivatives(g_10, wrt(p[6], p[1]), at(p))[2]) * lambda[10];
    values1[23] += Number(derivatives(g_10, wrt(p[6], p[2]), at(p))[2]) * lambda[10];
    values1[24] += Number(derivatives(g_10, wrt(p[6], p[3]), at(p))[2]) * lambda[10];
    values1[25] += Number(derivatives(g_10, wrt(p[6], p[4]), at(p))[2]) * lambda[10];
    values1[26] += Number(derivatives(g_10, wrt(p[6], p[5]), at(p))[2]) * lambda[10];
    values1[27] += Number(derivatives(g_10, wrt(p[6], p[6]), at(p))[2]) * lambda[10];
    
    values1[28] += Number(derivatives(g_10, wrt(p[7], p[0]), at(p))[2]) * lambda[10];
    values1[29] += Number(derivatives(g_10, wrt(p[7], p[1]), at(p))[2]) * lambda[10];
    values1[30] += Number(derivatives(g_10, wrt(p[7], p[2]), at(p))[2]) * lambda[10];
    values1[31] += Number(derivatives(g_10, wrt(p[7], p[3]), at(p))[2]) * lambda[10];
    values1[32] += Number(derivatives(g_10, wrt(p[7], p[4]), at(p))[2]) * lambda[10];
    values1[33] += Number(derivatives(g_10, wrt(p[7], p[5]), at(p))[2]) * lambda[10];
    values1[34] += Number(derivatives(g_10, wrt(p[7], p[6]), at(p))[2]) * lambda[10];
    values1[35] += Number(derivatives(g_10, wrt(p[7], p[7]), at(p))[2]) * lambda[10];
    
    values1[36] += Number(derivatives(g_10, wrt(p[8], p[0]), at(p))[2]) * lambda[10];
    values1[37] += Number(derivatives(g_10, wrt(p[8], p[1]), at(p))[2]) * lambda[10];
    values1[38] += Number(derivatives(g_10, wrt(p[8], p[2]), at(p))[2]) * lambda[10];
    values1[39] += Number(derivatives(g_10, wrt(p[8], p[3]), at(p))[2]) * lambda[10];
    values1[40] += Number(derivatives(g_10, wrt(p[8], p[4]), at(p))[2]) * lambda[10];
    values1[41] += Number(derivatives(g_10, wrt(p[8], p[5]), at(p))[2]) * lambda[10];
    values1[42] += Number(derivatives(g_10, wrt(p[8], p[6]), at(p))[2]) * lambda[10];
    values1[43] += Number(derivatives(g_10, wrt(p[8], p[7]), at(p))[2]) * lambda[10];
    values1[44] += Number(derivatives(g_10, wrt(p[8], p[8]), at(p))[2]) * lambda[10];

    
    
    
    
    values1[0] += Number(derivatives(g_11, wrt(p[0], p[0]), at(p))[2]) * lambda[11];
    
    values1[1] += Number(derivatives(g_11, wrt(p[1], p[0]), at(p))[2]) * lambda[11];
    values1[2] += Number(derivatives(g_11, wrt(p[1], p[1]), at(p))[2]) * lambda[11];
    
    values1[3] += Number(derivatives(g_11, wrt(p[2], p[0]), at(p))[2]) * lambda[11];
    values1[4] += Number(derivatives(g_11, wrt(p[2], p[1]), at(p))[2]) * lambda[11];
    values1[5] += Number(derivatives(g_11, wrt(p[2], p[2]), at(p))[2]) * lambda[11];
    
    values1[6] += Number(derivatives(g_11, wrt(p[3], p[0]), at(p))[2]) * lambda[11];
    values1[7] += Number(derivatives(g_11, wrt(p[3], p[1]), at(p))[2]) * lambda[11];
    values1[8] += Number(derivatives(g_11, wrt(p[3], p[2]), at(p))[2]) * lambda[11];
    values1[9] += Number(derivatives(g_11, wrt(p[3], p[3]), at(p))[2]) * lambda[11];
    
    values1[10] += Number(derivatives(g_11, wrt(p[4], p[0]), at(p))[2]) * lambda[11];
    values1[11] += Number(derivatives(g_11, wrt(p[4], p[1]), at(p))[2]) * lambda[11];
    values1[12] += Number(derivatives(g_11, wrt(p[4], p[2]), at(p))[2]) * lambda[11];
    values1[13] += Number(derivatives(g_11, wrt(p[4], p[3]), at(p))[2]) * lambda[11];
    values1[14] += Number(derivatives(g_11, wrt(p[4], p[4]), at(p))[2]) * lambda[11];
    
    values1[15] += Number(derivatives(g_11, wrt(p[5], p[0]), at(p))[2]) * lambda[11];
    values1[16] += Number(derivatives(g_11, wrt(p[5], p[1]), at(p))[2]) * lambda[11];
    values1[17] += Number(derivatives(g_11, wrt(p[5], p[2]), at(p))[2]) * lambda[11];
    values1[18] += Number(derivatives(g_11, wrt(p[5], p[3]), at(p))[2]) * lambda[11];
    values1[19] += Number(derivatives(g_11, wrt(p[5], p[4]), at(p))[2]) * lambda[11];
    values1[20] += Number(derivatives(g_11, wrt(p[5], p[5]), at(p))[2]) * lambda[11];
    
    values1[21] += Number(derivatives(g_11, wrt(p[6], p[0]), at(p))[2]) * lambda[11];
    values1[22] += Number(derivatives(g_11, wrt(p[6], p[1]), at(p))[2]) * lambda[11];
    values1[23] += Number(derivatives(g_11, wrt(p[6], p[2]), at(p))[2]) * lambda[11];
    values1[24] += Number(derivatives(g_11, wrt(p[6], p[3]), at(p))[2]) * lambda[11];
    values1[25] += Number(derivatives(g_11, wrt(p[6], p[4]), at(p))[2]) * lambda[11];
    values1[26] += Number(derivatives(g_11, wrt(p[6], p[5]), at(p))[2]) * lambda[11];
    values1[27] += Number(derivatives(g_11, wrt(p[6], p[6]), at(p))[2]) * lambda[11];
    
    values1[28] += Number(derivatives(g_11, wrt(p[7], p[0]), at(p))[2]) * lambda[11];
    values1[29] += Number(derivatives(g_11, wrt(p[7], p[1]), at(p))[2]) * lambda[11];
    values1[30] += Number(derivatives(g_11, wrt(p[7], p[2]), at(p))[2]) * lambda[11];
    values1[31] += Number(derivatives(g_11, wrt(p[7], p[3]), at(p))[2]) * lambda[11];
    values1[32] += Number(derivatives(g_11, wrt(p[7], p[4]), at(p))[2]) * lambda[11];
    values1[33] += Number(derivatives(g_11, wrt(p[7], p[5]), at(p))[2]) * lambda[11];
    values1[34] += Number(derivatives(g_11, wrt(p[7], p[6]), at(p))[2]) * lambda[11];
    values1[35] += Number(derivatives(g_11, wrt(p[7], p[7]), at(p))[2]) * lambda[11];
    
    values1[36] += Number(derivatives(g_11, wrt(p[8], p[0]), at(p))[2]) * lambda[11];
    values1[37] += Number(derivatives(g_11, wrt(p[8], p[1]), at(p))[2]) * lambda[11];
    values1[38] += Number(derivatives(g_11, wrt(p[8], p[2]), at(p))[2]) * lambda[11];
    values1[39] += Number(derivatives(g_11, wrt(p[8], p[3]), at(p))[2]) * lambda[11];
    values1[40] += Number(derivatives(g_11, wrt(p[8], p[4]), at(p))[2]) * lambda[11];
    values1[41] += Number(derivatives(g_11, wrt(p[8], p[5]), at(p))[2]) * lambda[11];
    values1[42] += Number(derivatives(g_11, wrt(p[8], p[6]), at(p))[2]) * lambda[11];
    values1[43] += Number(derivatives(g_11, wrt(p[8], p[7]), at(p))[2]) * lambda[11];
    values1[44] += Number(derivatives(g_11, wrt(p[8], p[8]), at(p))[2]) * lambda[11];

    
    
    
    
    values1[0] += Number(derivatives(g_12, wrt(p[0], p[0]), at(p))[2]) * lambda[12];
    
    values1[1] += Number(derivatives(g_12, wrt(p[1], p[0]), at(p))[2]) * lambda[12];
    values1[2] += Number(derivatives(g_12, wrt(p[1], p[1]), at(p))[2]) * lambda[12];
    
    values1[3] += Number(derivatives(g_12, wrt(p[2], p[0]), at(p))[2]) * lambda[12];
    values1[4] += Number(derivatives(g_12, wrt(p[2], p[1]), at(p))[2]) * lambda[12];
    values1[5] += Number(derivatives(g_12, wrt(p[2], p[2]), at(p))[2]) * lambda[12];
    
    values1[6] += Number(derivatives(g_12, wrt(p[3], p[0]), at(p))[2]) * lambda[12];
    values1[7] += Number(derivatives(g_12, wrt(p[3], p[1]), at(p))[2]) * lambda[12];
    values1[8] += Number(derivatives(g_12, wrt(p[3], p[2]), at(p))[2]) * lambda[12];
    values1[9] += Number(derivatives(g_12, wrt(p[3], p[3]), at(p))[2]) * lambda[12];
    
    values1[10] += Number(derivatives(g_12, wrt(p[4], p[0]), at(p))[2]) * lambda[12];
    values1[11] += Number(derivatives(g_12, wrt(p[4], p[1]), at(p))[2]) * lambda[12];
    values1[12] += Number(derivatives(g_12, wrt(p[4], p[2]), at(p))[2]) * lambda[12];
    values1[13] += Number(derivatives(g_12, wrt(p[4], p[3]), at(p))[2]) * lambda[12];
    values1[14] += Number(derivatives(g_12, wrt(p[4], p[4]), at(p))[2]) * lambda[12];
    
    values1[15] += Number(derivatives(g_12, wrt(p[5], p[0]), at(p))[2]) * lambda[12];
    values1[16] += Number(derivatives(g_12, wrt(p[5], p[1]), at(p))[2]) * lambda[12];
    values1[17] += Number(derivatives(g_12, wrt(p[5], p[2]), at(p))[2]) * lambda[12];
    values1[18] += Number(derivatives(g_12, wrt(p[5], p[3]), at(p))[2]) * lambda[12];
    values1[19] += Number(derivatives(g_12, wrt(p[5], p[4]), at(p))[2]) * lambda[12];
    values1[20] += Number(derivatives(g_12, wrt(p[5], p[5]), at(p))[2]) * lambda[12];
    
    values1[21] += Number(derivatives(g_12, wrt(p[6], p[0]), at(p))[2]) * lambda[12];
    values1[22] += Number(derivatives(g_12, wrt(p[6], p[1]), at(p))[2]) * lambda[12];
    values1[23] += Number(derivatives(g_12, wrt(p[6], p[2]), at(p))[2]) * lambda[12];
    values1[24] += Number(derivatives(g_12, wrt(p[6], p[3]), at(p))[2]) * lambda[12];
    values1[25] += Number(derivatives(g_12, wrt(p[6], p[4]), at(p))[2]) * lambda[12];
    values1[26] += Number(derivatives(g_12, wrt(p[6], p[5]), at(p))[2]) * lambda[12];
    values1[27] += Number(derivatives(g_12, wrt(p[6], p[6]), at(p))[2]) * lambda[12];
    
    values1[28] += Number(derivatives(g_12, wrt(p[7], p[0]), at(p))[2]) * lambda[12];
    values1[29] += Number(derivatives(g_12, wrt(p[7], p[1]), at(p))[2]) * lambda[12];
    values1[30] += Number(derivatives(g_12, wrt(p[7], p[2]), at(p))[2]) * lambda[12];
    values1[31] += Number(derivatives(g_12, wrt(p[7], p[3]), at(p))[2]) * lambda[12];
    values1[32] += Number(derivatives(g_12, wrt(p[7], p[4]), at(p))[2]) * lambda[12];
    values1[33] += Number(derivatives(g_12, wrt(p[7], p[5]), at(p))[2]) * lambda[12];
    values1[34] += Number(derivatives(g_12, wrt(p[7], p[6]), at(p))[2]) * lambda[12];
    values1[35] += Number(derivatives(g_12, wrt(p[7], p[7]), at(p))[2]) * lambda[12];
    
    values1[36] += Number(derivatives(g_12, wrt(p[8], p[0]), at(p))[2]) * lambda[12];
    values1[37] += Number(derivatives(g_12, wrt(p[8], p[1]), at(p))[2]) * lambda[12];
    values1[38] += Number(derivatives(g_12, wrt(p[8], p[2]), at(p))[2]) * lambda[12];
    values1[39] += Number(derivatives(g_12, wrt(p[8], p[3]), at(p))[2]) * lambda[12];
    values1[40] += Number(derivatives(g_12, wrt(p[8], p[4]), at(p))[2]) * lambda[12];
    values1[41] += Number(derivatives(g_12, wrt(p[8], p[5]), at(p))[2]) * lambda[12];
    values1[42] += Number(derivatives(g_12, wrt(p[8], p[6]), at(p))[2]) * lambda[12];
    values1[43] += Number(derivatives(g_12, wrt(p[8], p[7]), at(p))[2]) * lambda[12];
    values1[44] += Number(derivatives(g_12, wrt(p[8], p[8]), at(p))[2]) * lambda[12];

    
    
    
    
    values1[0] += Number(derivatives(g_13, wrt(p[0], p[0]), at(p))[2]) * lambda[13];
    
    values1[1] += Number(derivatives(g_13, wrt(p[1], p[0]), at(p))[2]) * lambda[13];
    values1[2] += Number(derivatives(g_13, wrt(p[1], p[1]), at(p))[2]) * lambda[13];
    
    values1[3] += Number(derivatives(g_13, wrt(p[2], p[0]), at(p))[2]) * lambda[13];
    values1[4] += Number(derivatives(g_13, wrt(p[2], p[1]), at(p))[2]) * lambda[13];
    values1[5] += Number(derivatives(g_13, wrt(p[2], p[2]), at(p))[2]) * lambda[13];
    
    values1[6] += Number(derivatives(g_13, wrt(p[3], p[0]), at(p))[2]) * lambda[13];
    values1[7] += Number(derivatives(g_13, wrt(p[3], p[1]), at(p))[2]) * lambda[13];
    values1[8] += Number(derivatives(g_13, wrt(p[3], p[2]), at(p))[2]) * lambda[13];
    values1[9] += Number(derivatives(g_13, wrt(p[3], p[3]), at(p))[2]) * lambda[13];
    
    values1[10] += Number(derivatives(g_13, wrt(p[4], p[0]), at(p))[2]) * lambda[13];
    values1[11] += Number(derivatives(g_13, wrt(p[4], p[1]), at(p))[2]) * lambda[13];
    values1[12] += Number(derivatives(g_13, wrt(p[4], p[2]), at(p))[2]) * lambda[13];
    values1[13] += Number(derivatives(g_13, wrt(p[4], p[3]), at(p))[2]) * lambda[13];
    values1[14] += Number(derivatives(g_13, wrt(p[4], p[4]), at(p))[2]) * lambda[13];
    
    values1[15] += Number(derivatives(g_13, wrt(p[5], p[0]), at(p))[2]) * lambda[13];
    values1[16] += Number(derivatives(g_13, wrt(p[5], p[1]), at(p))[2]) * lambda[13];
    values1[17] += Number(derivatives(g_13, wrt(p[5], p[2]), at(p))[2]) * lambda[13];
    values1[18] += Number(derivatives(g_13, wrt(p[5], p[3]), at(p))[2]) * lambda[13];
    values1[19] += Number(derivatives(g_13, wrt(p[5], p[4]), at(p))[2]) * lambda[13];
    values1[20] += Number(derivatives(g_13, wrt(p[5], p[5]), at(p))[2]) * lambda[13];
    
    values1[21] += Number(derivatives(g_13, wrt(p[6], p[0]), at(p))[2]) * lambda[13];
    values1[22] += Number(derivatives(g_13, wrt(p[6], p[1]), at(p))[2]) * lambda[13];
    values1[23] += Number(derivatives(g_13, wrt(p[6], p[2]), at(p))[2]) * lambda[13];
    values1[24] += Number(derivatives(g_13, wrt(p[6], p[3]), at(p))[2]) * lambda[13];
    values1[25] += Number(derivatives(g_13, wrt(p[6], p[4]), at(p))[2]) * lambda[13];
    values1[26] += Number(derivatives(g_13, wrt(p[6], p[5]), at(p))[2]) * lambda[13];
    values1[27] += Number(derivatives(g_13, wrt(p[6], p[6]), at(p))[2]) * lambda[13];
    
    values1[28] += Number(derivatives(g_13, wrt(p[7], p[0]), at(p))[2]) * lambda[13];
    values1[29] += Number(derivatives(g_13, wrt(p[7], p[1]), at(p))[2]) * lambda[13];
    values1[30] += Number(derivatives(g_13, wrt(p[7], p[2]), at(p))[2]) * lambda[13];
    values1[31] += Number(derivatives(g_13, wrt(p[7], p[3]), at(p))[2]) * lambda[13];
    values1[32] += Number(derivatives(g_13, wrt(p[7], p[4]), at(p))[2]) * lambda[13];
    values1[33] += Number(derivatives(g_13, wrt(p[7], p[5]), at(p))[2]) * lambda[13];
    values1[34] += Number(derivatives(g_13, wrt(p[7], p[6]), at(p))[2]) * lambda[13];
    values1[35] += Number(derivatives(g_13, wrt(p[7], p[7]), at(p))[2]) * lambda[13];
    
    values1[36] += Number(derivatives(g_13, wrt(p[8], p[0]), at(p))[2]) * lambda[13];
    values1[37] += Number(derivatives(g_13, wrt(p[8], p[1]), at(p))[2]) * lambda[13];
    values1[38] += Number(derivatives(g_13, wrt(p[8], p[2]), at(p))[2]) * lambda[13];
    values1[39] += Number(derivatives(g_13, wrt(p[8], p[3]), at(p))[2]) * lambda[13];
    values1[40] += Number(derivatives(g_13, wrt(p[8], p[4]), at(p))[2]) * lambda[13];
    values1[41] += Number(derivatives(g_13, wrt(p[8], p[5]), at(p))[2]) * lambda[13];
    values1[42] += Number(derivatives(g_13, wrt(p[8], p[6]), at(p))[2]) * lambda[13];
    values1[43] += Number(derivatives(g_13, wrt(p[8], p[7]), at(p))[2]) * lambda[13];
    values1[44] += Number(derivatives(g_13, wrt(p[8], p[8]), at(p))[2]) * lambda[13];

    




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