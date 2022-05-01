// In this file I will try to use the CasADi created file to evaluate functions 

#include "test_5_nlp.hpp"

#include <cassert>
#include <iostream>

#include <autodiff/reverse/var.hpp>
using namespace autodiff;


var f(var* x)
{
  return -0.5*(x[0]*x[3]-x[1]*x[2]+x[2]*x[8]-x[4]*x[8]+x[4]*x[7]-x[5]*x[6]);
}


var g_0(var* x)
{
  return 1 - pow(x[2],2) - pow(x[3],2);
}

var g_1(var* x)
{
  return 1 - pow(x[4],2) - pow(x[5],2);
}

var g_2(var* x)
{
  return 1 - pow(x[8],2);
}

var g_3(var* x)
{
  return 1 - pow(x[0],2) - pow(x[1]-x[8],2);
}

var g_4(var* x)
{
  return 1 - pow(x[0]-x[4],2) - pow(x[1]-x[5],2);
}

var g_5(var* x)
{
  return 1 - pow(x[0]-x[6],2) - pow(x[1]-x[7],2);
}

var g_6(var* x)
{
  return 1 - pow(x[2]-x[6],2) - pow(x[3]-x[7],2);
}

var g_7(var* x)
{
  return 1 - pow(x[2]-x[4],2) - pow(x[3]-x[5],2);
}

var g_8(var* x)
{
  return 1 - pow(x[6],2) - pow(x[7]-x[8],2);
}

var g_9(var* x)
{
  return x[0]*x[3]-x[1]*x[2];
}

var g_10(var* x)
{
  return x[2]*x[8];
}

var g_11(var* x)
{
  return -x[4]*x[8];
}

var g_12(var* x)
{
  return x[4]*x[7]-x[5]*x[6];
}

var g_13(var* x)
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

  var* p = (var*) calloc(n, sizeof(var));
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

  var* p = (var*) calloc(n, sizeof(var));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }

  auto d_f = derivativesx(f(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));

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


  var* p = (var*) calloc(n, sizeof(var));
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

    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = rand() % 10000 + 1;
    }

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_2 = derivativesx(g_2(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_3 = derivativesx(g_3(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_4 = derivativesx(g_4(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_5 = derivativesx(g_5(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_6 = derivativesx(g_6(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_7 = derivativesx(g_7(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_8 = derivativesx(g_8(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_9 = derivativesx(g_9(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_10 = derivativesx(g_10(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_11 = derivativesx(g_11(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_12 = derivativesx(g_12(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_13 = derivativesx(g_13(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    

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

    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_2 = derivativesx(g_2(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_3 = derivativesx(g_3(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_4 = derivativesx(g_4(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_5 = derivativesx(g_5(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_6 = derivativesx(g_6(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_7 = derivativesx(g_7(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_8 = derivativesx(g_8(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_9 = derivativesx(g_9(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_10 = derivativesx(g_10(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_11 = derivativesx(g_11(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_12 = derivativesx(g_12(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto d_g_13 = derivativesx(g_13(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    

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
   
    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = rand() % 10000 + 1;
    }

    double* values1 = (double*) calloc(n*n,sizeof(double));

    auto d_f = derivativesx(f(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));

    
    auto dd_f0 = derivativesx(d_f[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f1 = derivativesx(d_f[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f2 = derivativesx(d_f[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f3 = derivativesx(d_f[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f4 = derivativesx(d_f[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f5 = derivativesx(d_f[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f6 = derivativesx(d_f[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f7 = derivativesx(d_f[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f8 = derivativesx(d_f[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    

    
    
    
    values1[0] = Number(dd_f0[0]);
    
    values1[1] = Number(dd_f1[0]);
    values1[2] = Number(dd_f1[1]);
    
    values1[3] = Number(dd_f2[0]);
    values1[4] = Number(dd_f2[1]);
    values1[5] = Number(dd_f2[2]);
    
    values1[6] = Number(dd_f3[0]);
    values1[7] = Number(dd_f3[1]);
    values1[8] = Number(dd_f3[2]);
    values1[9] = Number(dd_f3[3]);
    
    values1[10] = Number(dd_f4[0]);
    values1[11] = Number(dd_f4[1]);
    values1[12] = Number(dd_f4[2]);
    values1[13] = Number(dd_f4[3]);
    values1[14] = Number(dd_f4[4]);
    
    values1[15] = Number(dd_f5[0]);
    values1[16] = Number(dd_f5[1]);
    values1[17] = Number(dd_f5[2]);
    values1[18] = Number(dd_f5[3]);
    values1[19] = Number(dd_f5[4]);
    values1[20] = Number(dd_f5[5]);
    
    values1[21] = Number(dd_f6[0]);
    values1[22] = Number(dd_f6[1]);
    values1[23] = Number(dd_f6[2]);
    values1[24] = Number(dd_f6[3]);
    values1[25] = Number(dd_f6[4]);
    values1[26] = Number(dd_f6[5]);
    values1[27] = Number(dd_f6[6]);
    
    values1[28] = Number(dd_f7[0]);
    values1[29] = Number(dd_f7[1]);
    values1[30] = Number(dd_f7[2]);
    values1[31] = Number(dd_f7[3]);
    values1[32] = Number(dd_f7[4]);
    values1[33] = Number(dd_f7[5]);
    values1[34] = Number(dd_f7[6]);
    values1[35] = Number(dd_f7[7]);
    
    values1[36] = Number(dd_f8[0]);
    values1[37] = Number(dd_f8[1]);
    values1[38] = Number(dd_f8[2]);
    values1[39] = Number(dd_f8[3]);
    values1[40] = Number(dd_f8[4]);
    values1[41] = Number(dd_f8[5]);
    values1[42] = Number(dd_f8[6]);
    values1[43] = Number(dd_f8[7]);
    values1[44] = Number(dd_f8[8]);

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_00 = derivativesx(d_g_0[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_01 = derivativesx(d_g_0[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_02 = derivativesx(d_g_0[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_03 = derivativesx(d_g_0[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_04 = derivativesx(d_g_0[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_05 = derivativesx(d_g_0[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_06 = derivativesx(d_g_0[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_07 = derivativesx(d_g_0[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_08 = derivativesx(d_g_0[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_00[0]);
    
    values1[1] += Number(dd_g_01[0]);
    values1[2] += Number(dd_g_01[1]);
    
    values1[3] += Number(dd_g_02[0]);
    values1[4] += Number(dd_g_02[1]);
    values1[5] += Number(dd_g_02[2]);
    
    values1[6] += Number(dd_g_03[0]);
    values1[7] += Number(dd_g_03[1]);
    values1[8] += Number(dd_g_03[2]);
    values1[9] += Number(dd_g_03[3]);
    
    values1[10] += Number(dd_g_04[0]);
    values1[11] += Number(dd_g_04[1]);
    values1[12] += Number(dd_g_04[2]);
    values1[13] += Number(dd_g_04[3]);
    values1[14] += Number(dd_g_04[4]);
    
    values1[15] += Number(dd_g_05[0]);
    values1[16] += Number(dd_g_05[1]);
    values1[17] += Number(dd_g_05[2]);
    values1[18] += Number(dd_g_05[3]);
    values1[19] += Number(dd_g_05[4]);
    values1[20] += Number(dd_g_05[5]);
    
    values1[21] += Number(dd_g_06[0]);
    values1[22] += Number(dd_g_06[1]);
    values1[23] += Number(dd_g_06[2]);
    values1[24] += Number(dd_g_06[3]);
    values1[25] += Number(dd_g_06[4]);
    values1[26] += Number(dd_g_06[5]);
    values1[27] += Number(dd_g_06[6]);
    
    values1[28] += Number(dd_g_07[0]);
    values1[29] += Number(dd_g_07[1]);
    values1[30] += Number(dd_g_07[2]);
    values1[31] += Number(dd_g_07[3]);
    values1[32] += Number(dd_g_07[4]);
    values1[33] += Number(dd_g_07[5]);
    values1[34] += Number(dd_g_07[6]);
    values1[35] += Number(dd_g_07[7]);
    
    values1[36] += Number(dd_g_08[0]);
    values1[37] += Number(dd_g_08[1]);
    values1[38] += Number(dd_g_08[2]);
    values1[39] += Number(dd_g_08[3]);
    values1[40] += Number(dd_g_08[4]);
    values1[41] += Number(dd_g_08[5]);
    values1[42] += Number(dd_g_08[6]);
    values1[43] += Number(dd_g_08[7]);
    values1[44] += Number(dd_g_08[8]);


    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_10 = derivativesx(d_g_1[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_11 = derivativesx(d_g_1[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_12 = derivativesx(d_g_1[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_13 = derivativesx(d_g_1[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_14 = derivativesx(d_g_1[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_15 = derivativesx(d_g_1[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_16 = derivativesx(d_g_1[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_17 = derivativesx(d_g_1[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_18 = derivativesx(d_g_1[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_10[0]);
    
    values1[1] += Number(dd_g_11[0]);
    values1[2] += Number(dd_g_11[1]);
    
    values1[3] += Number(dd_g_12[0]);
    values1[4] += Number(dd_g_12[1]);
    values1[5] += Number(dd_g_12[2]);
    
    values1[6] += Number(dd_g_13[0]);
    values1[7] += Number(dd_g_13[1]);
    values1[8] += Number(dd_g_13[2]);
    values1[9] += Number(dd_g_13[3]);
    
    values1[10] += Number(dd_g_14[0]);
    values1[11] += Number(dd_g_14[1]);
    values1[12] += Number(dd_g_14[2]);
    values1[13] += Number(dd_g_14[3]);
    values1[14] += Number(dd_g_14[4]);
    
    values1[15] += Number(dd_g_15[0]);
    values1[16] += Number(dd_g_15[1]);
    values1[17] += Number(dd_g_15[2]);
    values1[18] += Number(dd_g_15[3]);
    values1[19] += Number(dd_g_15[4]);
    values1[20] += Number(dd_g_15[5]);
    
    values1[21] += Number(dd_g_16[0]);
    values1[22] += Number(dd_g_16[1]);
    values1[23] += Number(dd_g_16[2]);
    values1[24] += Number(dd_g_16[3]);
    values1[25] += Number(dd_g_16[4]);
    values1[26] += Number(dd_g_16[5]);
    values1[27] += Number(dd_g_16[6]);
    
    values1[28] += Number(dd_g_17[0]);
    values1[29] += Number(dd_g_17[1]);
    values1[30] += Number(dd_g_17[2]);
    values1[31] += Number(dd_g_17[3]);
    values1[32] += Number(dd_g_17[4]);
    values1[33] += Number(dd_g_17[5]);
    values1[34] += Number(dd_g_17[6]);
    values1[35] += Number(dd_g_17[7]);
    
    values1[36] += Number(dd_g_18[0]);
    values1[37] += Number(dd_g_18[1]);
    values1[38] += Number(dd_g_18[2]);
    values1[39] += Number(dd_g_18[3]);
    values1[40] += Number(dd_g_18[4]);
    values1[41] += Number(dd_g_18[5]);
    values1[42] += Number(dd_g_18[6]);
    values1[43] += Number(dd_g_18[7]);
    values1[44] += Number(dd_g_18[8]);


    
    auto d_g_2 = derivativesx(g_2(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_20 = derivativesx(d_g_2[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_21 = derivativesx(d_g_2[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_22 = derivativesx(d_g_2[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_23 = derivativesx(d_g_2[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_24 = derivativesx(d_g_2[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_25 = derivativesx(d_g_2[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_26 = derivativesx(d_g_2[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_27 = derivativesx(d_g_2[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_28 = derivativesx(d_g_2[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_20[0]);
    
    values1[1] += Number(dd_g_21[0]);
    values1[2] += Number(dd_g_21[1]);
    
    values1[3] += Number(dd_g_22[0]);
    values1[4] += Number(dd_g_22[1]);
    values1[5] += Number(dd_g_22[2]);
    
    values1[6] += Number(dd_g_23[0]);
    values1[7] += Number(dd_g_23[1]);
    values1[8] += Number(dd_g_23[2]);
    values1[9] += Number(dd_g_23[3]);
    
    values1[10] += Number(dd_g_24[0]);
    values1[11] += Number(dd_g_24[1]);
    values1[12] += Number(dd_g_24[2]);
    values1[13] += Number(dd_g_24[3]);
    values1[14] += Number(dd_g_24[4]);
    
    values1[15] += Number(dd_g_25[0]);
    values1[16] += Number(dd_g_25[1]);
    values1[17] += Number(dd_g_25[2]);
    values1[18] += Number(dd_g_25[3]);
    values1[19] += Number(dd_g_25[4]);
    values1[20] += Number(dd_g_25[5]);
    
    values1[21] += Number(dd_g_26[0]);
    values1[22] += Number(dd_g_26[1]);
    values1[23] += Number(dd_g_26[2]);
    values1[24] += Number(dd_g_26[3]);
    values1[25] += Number(dd_g_26[4]);
    values1[26] += Number(dd_g_26[5]);
    values1[27] += Number(dd_g_26[6]);
    
    values1[28] += Number(dd_g_27[0]);
    values1[29] += Number(dd_g_27[1]);
    values1[30] += Number(dd_g_27[2]);
    values1[31] += Number(dd_g_27[3]);
    values1[32] += Number(dd_g_27[4]);
    values1[33] += Number(dd_g_27[5]);
    values1[34] += Number(dd_g_27[6]);
    values1[35] += Number(dd_g_27[7]);
    
    values1[36] += Number(dd_g_28[0]);
    values1[37] += Number(dd_g_28[1]);
    values1[38] += Number(dd_g_28[2]);
    values1[39] += Number(dd_g_28[3]);
    values1[40] += Number(dd_g_28[4]);
    values1[41] += Number(dd_g_28[5]);
    values1[42] += Number(dd_g_28[6]);
    values1[43] += Number(dd_g_28[7]);
    values1[44] += Number(dd_g_28[8]);


    
    auto d_g_3 = derivativesx(g_3(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_30 = derivativesx(d_g_3[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_31 = derivativesx(d_g_3[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_32 = derivativesx(d_g_3[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_33 = derivativesx(d_g_3[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_34 = derivativesx(d_g_3[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_35 = derivativesx(d_g_3[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_36 = derivativesx(d_g_3[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_37 = derivativesx(d_g_3[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_38 = derivativesx(d_g_3[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_30[0]);
    
    values1[1] += Number(dd_g_31[0]);
    values1[2] += Number(dd_g_31[1]);
    
    values1[3] += Number(dd_g_32[0]);
    values1[4] += Number(dd_g_32[1]);
    values1[5] += Number(dd_g_32[2]);
    
    values1[6] += Number(dd_g_33[0]);
    values1[7] += Number(dd_g_33[1]);
    values1[8] += Number(dd_g_33[2]);
    values1[9] += Number(dd_g_33[3]);
    
    values1[10] += Number(dd_g_34[0]);
    values1[11] += Number(dd_g_34[1]);
    values1[12] += Number(dd_g_34[2]);
    values1[13] += Number(dd_g_34[3]);
    values1[14] += Number(dd_g_34[4]);
    
    values1[15] += Number(dd_g_35[0]);
    values1[16] += Number(dd_g_35[1]);
    values1[17] += Number(dd_g_35[2]);
    values1[18] += Number(dd_g_35[3]);
    values1[19] += Number(dd_g_35[4]);
    values1[20] += Number(dd_g_35[5]);
    
    values1[21] += Number(dd_g_36[0]);
    values1[22] += Number(dd_g_36[1]);
    values1[23] += Number(dd_g_36[2]);
    values1[24] += Number(dd_g_36[3]);
    values1[25] += Number(dd_g_36[4]);
    values1[26] += Number(dd_g_36[5]);
    values1[27] += Number(dd_g_36[6]);
    
    values1[28] += Number(dd_g_37[0]);
    values1[29] += Number(dd_g_37[1]);
    values1[30] += Number(dd_g_37[2]);
    values1[31] += Number(dd_g_37[3]);
    values1[32] += Number(dd_g_37[4]);
    values1[33] += Number(dd_g_37[5]);
    values1[34] += Number(dd_g_37[6]);
    values1[35] += Number(dd_g_37[7]);
    
    values1[36] += Number(dd_g_38[0]);
    values1[37] += Number(dd_g_38[1]);
    values1[38] += Number(dd_g_38[2]);
    values1[39] += Number(dd_g_38[3]);
    values1[40] += Number(dd_g_38[4]);
    values1[41] += Number(dd_g_38[5]);
    values1[42] += Number(dd_g_38[6]);
    values1[43] += Number(dd_g_38[7]);
    values1[44] += Number(dd_g_38[8]);


    
    auto d_g_4 = derivativesx(g_4(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_40 = derivativesx(d_g_4[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_41 = derivativesx(d_g_4[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_42 = derivativesx(d_g_4[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_43 = derivativesx(d_g_4[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_44 = derivativesx(d_g_4[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_45 = derivativesx(d_g_4[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_46 = derivativesx(d_g_4[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_47 = derivativesx(d_g_4[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_48 = derivativesx(d_g_4[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_40[0]);
    
    values1[1] += Number(dd_g_41[0]);
    values1[2] += Number(dd_g_41[1]);
    
    values1[3] += Number(dd_g_42[0]);
    values1[4] += Number(dd_g_42[1]);
    values1[5] += Number(dd_g_42[2]);
    
    values1[6] += Number(dd_g_43[0]);
    values1[7] += Number(dd_g_43[1]);
    values1[8] += Number(dd_g_43[2]);
    values1[9] += Number(dd_g_43[3]);
    
    values1[10] += Number(dd_g_44[0]);
    values1[11] += Number(dd_g_44[1]);
    values1[12] += Number(dd_g_44[2]);
    values1[13] += Number(dd_g_44[3]);
    values1[14] += Number(dd_g_44[4]);
    
    values1[15] += Number(dd_g_45[0]);
    values1[16] += Number(dd_g_45[1]);
    values1[17] += Number(dd_g_45[2]);
    values1[18] += Number(dd_g_45[3]);
    values1[19] += Number(dd_g_45[4]);
    values1[20] += Number(dd_g_45[5]);
    
    values1[21] += Number(dd_g_46[0]);
    values1[22] += Number(dd_g_46[1]);
    values1[23] += Number(dd_g_46[2]);
    values1[24] += Number(dd_g_46[3]);
    values1[25] += Number(dd_g_46[4]);
    values1[26] += Number(dd_g_46[5]);
    values1[27] += Number(dd_g_46[6]);
    
    values1[28] += Number(dd_g_47[0]);
    values1[29] += Number(dd_g_47[1]);
    values1[30] += Number(dd_g_47[2]);
    values1[31] += Number(dd_g_47[3]);
    values1[32] += Number(dd_g_47[4]);
    values1[33] += Number(dd_g_47[5]);
    values1[34] += Number(dd_g_47[6]);
    values1[35] += Number(dd_g_47[7]);
    
    values1[36] += Number(dd_g_48[0]);
    values1[37] += Number(dd_g_48[1]);
    values1[38] += Number(dd_g_48[2]);
    values1[39] += Number(dd_g_48[3]);
    values1[40] += Number(dd_g_48[4]);
    values1[41] += Number(dd_g_48[5]);
    values1[42] += Number(dd_g_48[6]);
    values1[43] += Number(dd_g_48[7]);
    values1[44] += Number(dd_g_48[8]);


    
    auto d_g_5 = derivativesx(g_5(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_50 = derivativesx(d_g_5[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_51 = derivativesx(d_g_5[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_52 = derivativesx(d_g_5[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_53 = derivativesx(d_g_5[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_54 = derivativesx(d_g_5[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_55 = derivativesx(d_g_5[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_56 = derivativesx(d_g_5[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_57 = derivativesx(d_g_5[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_58 = derivativesx(d_g_5[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_50[0]);
    
    values1[1] += Number(dd_g_51[0]);
    values1[2] += Number(dd_g_51[1]);
    
    values1[3] += Number(dd_g_52[0]);
    values1[4] += Number(dd_g_52[1]);
    values1[5] += Number(dd_g_52[2]);
    
    values1[6] += Number(dd_g_53[0]);
    values1[7] += Number(dd_g_53[1]);
    values1[8] += Number(dd_g_53[2]);
    values1[9] += Number(dd_g_53[3]);
    
    values1[10] += Number(dd_g_54[0]);
    values1[11] += Number(dd_g_54[1]);
    values1[12] += Number(dd_g_54[2]);
    values1[13] += Number(dd_g_54[3]);
    values1[14] += Number(dd_g_54[4]);
    
    values1[15] += Number(dd_g_55[0]);
    values1[16] += Number(dd_g_55[1]);
    values1[17] += Number(dd_g_55[2]);
    values1[18] += Number(dd_g_55[3]);
    values1[19] += Number(dd_g_55[4]);
    values1[20] += Number(dd_g_55[5]);
    
    values1[21] += Number(dd_g_56[0]);
    values1[22] += Number(dd_g_56[1]);
    values1[23] += Number(dd_g_56[2]);
    values1[24] += Number(dd_g_56[3]);
    values1[25] += Number(dd_g_56[4]);
    values1[26] += Number(dd_g_56[5]);
    values1[27] += Number(dd_g_56[6]);
    
    values1[28] += Number(dd_g_57[0]);
    values1[29] += Number(dd_g_57[1]);
    values1[30] += Number(dd_g_57[2]);
    values1[31] += Number(dd_g_57[3]);
    values1[32] += Number(dd_g_57[4]);
    values1[33] += Number(dd_g_57[5]);
    values1[34] += Number(dd_g_57[6]);
    values1[35] += Number(dd_g_57[7]);
    
    values1[36] += Number(dd_g_58[0]);
    values1[37] += Number(dd_g_58[1]);
    values1[38] += Number(dd_g_58[2]);
    values1[39] += Number(dd_g_58[3]);
    values1[40] += Number(dd_g_58[4]);
    values1[41] += Number(dd_g_58[5]);
    values1[42] += Number(dd_g_58[6]);
    values1[43] += Number(dd_g_58[7]);
    values1[44] += Number(dd_g_58[8]);


    
    auto d_g_6 = derivativesx(g_6(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_60 = derivativesx(d_g_6[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_61 = derivativesx(d_g_6[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_62 = derivativesx(d_g_6[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_63 = derivativesx(d_g_6[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_64 = derivativesx(d_g_6[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_65 = derivativesx(d_g_6[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_66 = derivativesx(d_g_6[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_67 = derivativesx(d_g_6[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_68 = derivativesx(d_g_6[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_60[0]);
    
    values1[1] += Number(dd_g_61[0]);
    values1[2] += Number(dd_g_61[1]);
    
    values1[3] += Number(dd_g_62[0]);
    values1[4] += Number(dd_g_62[1]);
    values1[5] += Number(dd_g_62[2]);
    
    values1[6] += Number(dd_g_63[0]);
    values1[7] += Number(dd_g_63[1]);
    values1[8] += Number(dd_g_63[2]);
    values1[9] += Number(dd_g_63[3]);
    
    values1[10] += Number(dd_g_64[0]);
    values1[11] += Number(dd_g_64[1]);
    values1[12] += Number(dd_g_64[2]);
    values1[13] += Number(dd_g_64[3]);
    values1[14] += Number(dd_g_64[4]);
    
    values1[15] += Number(dd_g_65[0]);
    values1[16] += Number(dd_g_65[1]);
    values1[17] += Number(dd_g_65[2]);
    values1[18] += Number(dd_g_65[3]);
    values1[19] += Number(dd_g_65[4]);
    values1[20] += Number(dd_g_65[5]);
    
    values1[21] += Number(dd_g_66[0]);
    values1[22] += Number(dd_g_66[1]);
    values1[23] += Number(dd_g_66[2]);
    values1[24] += Number(dd_g_66[3]);
    values1[25] += Number(dd_g_66[4]);
    values1[26] += Number(dd_g_66[5]);
    values1[27] += Number(dd_g_66[6]);
    
    values1[28] += Number(dd_g_67[0]);
    values1[29] += Number(dd_g_67[1]);
    values1[30] += Number(dd_g_67[2]);
    values1[31] += Number(dd_g_67[3]);
    values1[32] += Number(dd_g_67[4]);
    values1[33] += Number(dd_g_67[5]);
    values1[34] += Number(dd_g_67[6]);
    values1[35] += Number(dd_g_67[7]);
    
    values1[36] += Number(dd_g_68[0]);
    values1[37] += Number(dd_g_68[1]);
    values1[38] += Number(dd_g_68[2]);
    values1[39] += Number(dd_g_68[3]);
    values1[40] += Number(dd_g_68[4]);
    values1[41] += Number(dd_g_68[5]);
    values1[42] += Number(dd_g_68[6]);
    values1[43] += Number(dd_g_68[7]);
    values1[44] += Number(dd_g_68[8]);


    
    auto d_g_7 = derivativesx(g_7(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_70 = derivativesx(d_g_7[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_71 = derivativesx(d_g_7[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_72 = derivativesx(d_g_7[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_73 = derivativesx(d_g_7[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_74 = derivativesx(d_g_7[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_75 = derivativesx(d_g_7[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_76 = derivativesx(d_g_7[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_77 = derivativesx(d_g_7[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_78 = derivativesx(d_g_7[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_70[0]);
    
    values1[1] += Number(dd_g_71[0]);
    values1[2] += Number(dd_g_71[1]);
    
    values1[3] += Number(dd_g_72[0]);
    values1[4] += Number(dd_g_72[1]);
    values1[5] += Number(dd_g_72[2]);
    
    values1[6] += Number(dd_g_73[0]);
    values1[7] += Number(dd_g_73[1]);
    values1[8] += Number(dd_g_73[2]);
    values1[9] += Number(dd_g_73[3]);
    
    values1[10] += Number(dd_g_74[0]);
    values1[11] += Number(dd_g_74[1]);
    values1[12] += Number(dd_g_74[2]);
    values1[13] += Number(dd_g_74[3]);
    values1[14] += Number(dd_g_74[4]);
    
    values1[15] += Number(dd_g_75[0]);
    values1[16] += Number(dd_g_75[1]);
    values1[17] += Number(dd_g_75[2]);
    values1[18] += Number(dd_g_75[3]);
    values1[19] += Number(dd_g_75[4]);
    values1[20] += Number(dd_g_75[5]);
    
    values1[21] += Number(dd_g_76[0]);
    values1[22] += Number(dd_g_76[1]);
    values1[23] += Number(dd_g_76[2]);
    values1[24] += Number(dd_g_76[3]);
    values1[25] += Number(dd_g_76[4]);
    values1[26] += Number(dd_g_76[5]);
    values1[27] += Number(dd_g_76[6]);
    
    values1[28] += Number(dd_g_77[0]);
    values1[29] += Number(dd_g_77[1]);
    values1[30] += Number(dd_g_77[2]);
    values1[31] += Number(dd_g_77[3]);
    values1[32] += Number(dd_g_77[4]);
    values1[33] += Number(dd_g_77[5]);
    values1[34] += Number(dd_g_77[6]);
    values1[35] += Number(dd_g_77[7]);
    
    values1[36] += Number(dd_g_78[0]);
    values1[37] += Number(dd_g_78[1]);
    values1[38] += Number(dd_g_78[2]);
    values1[39] += Number(dd_g_78[3]);
    values1[40] += Number(dd_g_78[4]);
    values1[41] += Number(dd_g_78[5]);
    values1[42] += Number(dd_g_78[6]);
    values1[43] += Number(dd_g_78[7]);
    values1[44] += Number(dd_g_78[8]);


    
    auto d_g_8 = derivativesx(g_8(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_80 = derivativesx(d_g_8[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_81 = derivativesx(d_g_8[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_82 = derivativesx(d_g_8[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_83 = derivativesx(d_g_8[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_84 = derivativesx(d_g_8[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_85 = derivativesx(d_g_8[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_86 = derivativesx(d_g_8[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_87 = derivativesx(d_g_8[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_88 = derivativesx(d_g_8[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_80[0]);
    
    values1[1] += Number(dd_g_81[0]);
    values1[2] += Number(dd_g_81[1]);
    
    values1[3] += Number(dd_g_82[0]);
    values1[4] += Number(dd_g_82[1]);
    values1[5] += Number(dd_g_82[2]);
    
    values1[6] += Number(dd_g_83[0]);
    values1[7] += Number(dd_g_83[1]);
    values1[8] += Number(dd_g_83[2]);
    values1[9] += Number(dd_g_83[3]);
    
    values1[10] += Number(dd_g_84[0]);
    values1[11] += Number(dd_g_84[1]);
    values1[12] += Number(dd_g_84[2]);
    values1[13] += Number(dd_g_84[3]);
    values1[14] += Number(dd_g_84[4]);
    
    values1[15] += Number(dd_g_85[0]);
    values1[16] += Number(dd_g_85[1]);
    values1[17] += Number(dd_g_85[2]);
    values1[18] += Number(dd_g_85[3]);
    values1[19] += Number(dd_g_85[4]);
    values1[20] += Number(dd_g_85[5]);
    
    values1[21] += Number(dd_g_86[0]);
    values1[22] += Number(dd_g_86[1]);
    values1[23] += Number(dd_g_86[2]);
    values1[24] += Number(dd_g_86[3]);
    values1[25] += Number(dd_g_86[4]);
    values1[26] += Number(dd_g_86[5]);
    values1[27] += Number(dd_g_86[6]);
    
    values1[28] += Number(dd_g_87[0]);
    values1[29] += Number(dd_g_87[1]);
    values1[30] += Number(dd_g_87[2]);
    values1[31] += Number(dd_g_87[3]);
    values1[32] += Number(dd_g_87[4]);
    values1[33] += Number(dd_g_87[5]);
    values1[34] += Number(dd_g_87[6]);
    values1[35] += Number(dd_g_87[7]);
    
    values1[36] += Number(dd_g_88[0]);
    values1[37] += Number(dd_g_88[1]);
    values1[38] += Number(dd_g_88[2]);
    values1[39] += Number(dd_g_88[3]);
    values1[40] += Number(dd_g_88[4]);
    values1[41] += Number(dd_g_88[5]);
    values1[42] += Number(dd_g_88[6]);
    values1[43] += Number(dd_g_88[7]);
    values1[44] += Number(dd_g_88[8]);


    
    auto d_g_9 = derivativesx(g_9(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_90 = derivativesx(d_g_9[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_91 = derivativesx(d_g_9[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_92 = derivativesx(d_g_9[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_93 = derivativesx(d_g_9[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_94 = derivativesx(d_g_9[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_95 = derivativesx(d_g_9[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_96 = derivativesx(d_g_9[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_97 = derivativesx(d_g_9[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_98 = derivativesx(d_g_9[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_90[0]);
    
    values1[1] += Number(dd_g_91[0]);
    values1[2] += Number(dd_g_91[1]);
    
    values1[3] += Number(dd_g_92[0]);
    values1[4] += Number(dd_g_92[1]);
    values1[5] += Number(dd_g_92[2]);
    
    values1[6] += Number(dd_g_93[0]);
    values1[7] += Number(dd_g_93[1]);
    values1[8] += Number(dd_g_93[2]);
    values1[9] += Number(dd_g_93[3]);
    
    values1[10] += Number(dd_g_94[0]);
    values1[11] += Number(dd_g_94[1]);
    values1[12] += Number(dd_g_94[2]);
    values1[13] += Number(dd_g_94[3]);
    values1[14] += Number(dd_g_94[4]);
    
    values1[15] += Number(dd_g_95[0]);
    values1[16] += Number(dd_g_95[1]);
    values1[17] += Number(dd_g_95[2]);
    values1[18] += Number(dd_g_95[3]);
    values1[19] += Number(dd_g_95[4]);
    values1[20] += Number(dd_g_95[5]);
    
    values1[21] += Number(dd_g_96[0]);
    values1[22] += Number(dd_g_96[1]);
    values1[23] += Number(dd_g_96[2]);
    values1[24] += Number(dd_g_96[3]);
    values1[25] += Number(dd_g_96[4]);
    values1[26] += Number(dd_g_96[5]);
    values1[27] += Number(dd_g_96[6]);
    
    values1[28] += Number(dd_g_97[0]);
    values1[29] += Number(dd_g_97[1]);
    values1[30] += Number(dd_g_97[2]);
    values1[31] += Number(dd_g_97[3]);
    values1[32] += Number(dd_g_97[4]);
    values1[33] += Number(dd_g_97[5]);
    values1[34] += Number(dd_g_97[6]);
    values1[35] += Number(dd_g_97[7]);
    
    values1[36] += Number(dd_g_98[0]);
    values1[37] += Number(dd_g_98[1]);
    values1[38] += Number(dd_g_98[2]);
    values1[39] += Number(dd_g_98[3]);
    values1[40] += Number(dd_g_98[4]);
    values1[41] += Number(dd_g_98[5]);
    values1[42] += Number(dd_g_98[6]);
    values1[43] += Number(dd_g_98[7]);
    values1[44] += Number(dd_g_98[8]);


    
    auto d_g_10 = derivativesx(g_10(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_100 = derivativesx(d_g_10[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_101 = derivativesx(d_g_10[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_102 = derivativesx(d_g_10[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_103 = derivativesx(d_g_10[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_104 = derivativesx(d_g_10[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_105 = derivativesx(d_g_10[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_106 = derivativesx(d_g_10[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_107 = derivativesx(d_g_10[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_108 = derivativesx(d_g_10[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_100[0]);
    
    values1[1] += Number(dd_g_101[0]);
    values1[2] += Number(dd_g_101[1]);
    
    values1[3] += Number(dd_g_102[0]);
    values1[4] += Number(dd_g_102[1]);
    values1[5] += Number(dd_g_102[2]);
    
    values1[6] += Number(dd_g_103[0]);
    values1[7] += Number(dd_g_103[1]);
    values1[8] += Number(dd_g_103[2]);
    values1[9] += Number(dd_g_103[3]);
    
    values1[10] += Number(dd_g_104[0]);
    values1[11] += Number(dd_g_104[1]);
    values1[12] += Number(dd_g_104[2]);
    values1[13] += Number(dd_g_104[3]);
    values1[14] += Number(dd_g_104[4]);
    
    values1[15] += Number(dd_g_105[0]);
    values1[16] += Number(dd_g_105[1]);
    values1[17] += Number(dd_g_105[2]);
    values1[18] += Number(dd_g_105[3]);
    values1[19] += Number(dd_g_105[4]);
    values1[20] += Number(dd_g_105[5]);
    
    values1[21] += Number(dd_g_106[0]);
    values1[22] += Number(dd_g_106[1]);
    values1[23] += Number(dd_g_106[2]);
    values1[24] += Number(dd_g_106[3]);
    values1[25] += Number(dd_g_106[4]);
    values1[26] += Number(dd_g_106[5]);
    values1[27] += Number(dd_g_106[6]);
    
    values1[28] += Number(dd_g_107[0]);
    values1[29] += Number(dd_g_107[1]);
    values1[30] += Number(dd_g_107[2]);
    values1[31] += Number(dd_g_107[3]);
    values1[32] += Number(dd_g_107[4]);
    values1[33] += Number(dd_g_107[5]);
    values1[34] += Number(dd_g_107[6]);
    values1[35] += Number(dd_g_107[7]);
    
    values1[36] += Number(dd_g_108[0]);
    values1[37] += Number(dd_g_108[1]);
    values1[38] += Number(dd_g_108[2]);
    values1[39] += Number(dd_g_108[3]);
    values1[40] += Number(dd_g_108[4]);
    values1[41] += Number(dd_g_108[5]);
    values1[42] += Number(dd_g_108[6]);
    values1[43] += Number(dd_g_108[7]);
    values1[44] += Number(dd_g_108[8]);


    
    auto d_g_11 = derivativesx(g_11(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_110 = derivativesx(d_g_11[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_111 = derivativesx(d_g_11[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_112 = derivativesx(d_g_11[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_113 = derivativesx(d_g_11[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_114 = derivativesx(d_g_11[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_115 = derivativesx(d_g_11[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_116 = derivativesx(d_g_11[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_117 = derivativesx(d_g_11[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_118 = derivativesx(d_g_11[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_110[0]);
    
    values1[1] += Number(dd_g_111[0]);
    values1[2] += Number(dd_g_111[1]);
    
    values1[3] += Number(dd_g_112[0]);
    values1[4] += Number(dd_g_112[1]);
    values1[5] += Number(dd_g_112[2]);
    
    values1[6] += Number(dd_g_113[0]);
    values1[7] += Number(dd_g_113[1]);
    values1[8] += Number(dd_g_113[2]);
    values1[9] += Number(dd_g_113[3]);
    
    values1[10] += Number(dd_g_114[0]);
    values1[11] += Number(dd_g_114[1]);
    values1[12] += Number(dd_g_114[2]);
    values1[13] += Number(dd_g_114[3]);
    values1[14] += Number(dd_g_114[4]);
    
    values1[15] += Number(dd_g_115[0]);
    values1[16] += Number(dd_g_115[1]);
    values1[17] += Number(dd_g_115[2]);
    values1[18] += Number(dd_g_115[3]);
    values1[19] += Number(dd_g_115[4]);
    values1[20] += Number(dd_g_115[5]);
    
    values1[21] += Number(dd_g_116[0]);
    values1[22] += Number(dd_g_116[1]);
    values1[23] += Number(dd_g_116[2]);
    values1[24] += Number(dd_g_116[3]);
    values1[25] += Number(dd_g_116[4]);
    values1[26] += Number(dd_g_116[5]);
    values1[27] += Number(dd_g_116[6]);
    
    values1[28] += Number(dd_g_117[0]);
    values1[29] += Number(dd_g_117[1]);
    values1[30] += Number(dd_g_117[2]);
    values1[31] += Number(dd_g_117[3]);
    values1[32] += Number(dd_g_117[4]);
    values1[33] += Number(dd_g_117[5]);
    values1[34] += Number(dd_g_117[6]);
    values1[35] += Number(dd_g_117[7]);
    
    values1[36] += Number(dd_g_118[0]);
    values1[37] += Number(dd_g_118[1]);
    values1[38] += Number(dd_g_118[2]);
    values1[39] += Number(dd_g_118[3]);
    values1[40] += Number(dd_g_118[4]);
    values1[41] += Number(dd_g_118[5]);
    values1[42] += Number(dd_g_118[6]);
    values1[43] += Number(dd_g_118[7]);
    values1[44] += Number(dd_g_118[8]);


    
    auto d_g_12 = derivativesx(g_12(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_120 = derivativesx(d_g_12[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_121 = derivativesx(d_g_12[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_122 = derivativesx(d_g_12[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_123 = derivativesx(d_g_12[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_124 = derivativesx(d_g_12[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_125 = derivativesx(d_g_12[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_126 = derivativesx(d_g_12[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_127 = derivativesx(d_g_12[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_128 = derivativesx(d_g_12[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_120[0]);
    
    values1[1] += Number(dd_g_121[0]);
    values1[2] += Number(dd_g_121[1]);
    
    values1[3] += Number(dd_g_122[0]);
    values1[4] += Number(dd_g_122[1]);
    values1[5] += Number(dd_g_122[2]);
    
    values1[6] += Number(dd_g_123[0]);
    values1[7] += Number(dd_g_123[1]);
    values1[8] += Number(dd_g_123[2]);
    values1[9] += Number(dd_g_123[3]);
    
    values1[10] += Number(dd_g_124[0]);
    values1[11] += Number(dd_g_124[1]);
    values1[12] += Number(dd_g_124[2]);
    values1[13] += Number(dd_g_124[3]);
    values1[14] += Number(dd_g_124[4]);
    
    values1[15] += Number(dd_g_125[0]);
    values1[16] += Number(dd_g_125[1]);
    values1[17] += Number(dd_g_125[2]);
    values1[18] += Number(dd_g_125[3]);
    values1[19] += Number(dd_g_125[4]);
    values1[20] += Number(dd_g_125[5]);
    
    values1[21] += Number(dd_g_126[0]);
    values1[22] += Number(dd_g_126[1]);
    values1[23] += Number(dd_g_126[2]);
    values1[24] += Number(dd_g_126[3]);
    values1[25] += Number(dd_g_126[4]);
    values1[26] += Number(dd_g_126[5]);
    values1[27] += Number(dd_g_126[6]);
    
    values1[28] += Number(dd_g_127[0]);
    values1[29] += Number(dd_g_127[1]);
    values1[30] += Number(dd_g_127[2]);
    values1[31] += Number(dd_g_127[3]);
    values1[32] += Number(dd_g_127[4]);
    values1[33] += Number(dd_g_127[5]);
    values1[34] += Number(dd_g_127[6]);
    values1[35] += Number(dd_g_127[7]);
    
    values1[36] += Number(dd_g_128[0]);
    values1[37] += Number(dd_g_128[1]);
    values1[38] += Number(dd_g_128[2]);
    values1[39] += Number(dd_g_128[3]);
    values1[40] += Number(dd_g_128[4]);
    values1[41] += Number(dd_g_128[5]);
    values1[42] += Number(dd_g_128[6]);
    values1[43] += Number(dd_g_128[7]);
    values1[44] += Number(dd_g_128[8]);


    
    auto d_g_13 = derivativesx(g_13(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_130 = derivativesx(d_g_13[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_131 = derivativesx(d_g_13[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_132 = derivativesx(d_g_13[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_133 = derivativesx(d_g_13[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_134 = derivativesx(d_g_13[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_135 = derivativesx(d_g_13[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_136 = derivativesx(d_g_13[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_137 = derivativesx(d_g_13[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_138 = derivativesx(d_g_13[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_130[0]);
    
    values1[1] += Number(dd_g_131[0]);
    values1[2] += Number(dd_g_131[1]);
    
    values1[3] += Number(dd_g_132[0]);
    values1[4] += Number(dd_g_132[1]);
    values1[5] += Number(dd_g_132[2]);
    
    values1[6] += Number(dd_g_133[0]);
    values1[7] += Number(dd_g_133[1]);
    values1[8] += Number(dd_g_133[2]);
    values1[9] += Number(dd_g_133[3]);
    
    values1[10] += Number(dd_g_134[0]);
    values1[11] += Number(dd_g_134[1]);
    values1[12] += Number(dd_g_134[2]);
    values1[13] += Number(dd_g_134[3]);
    values1[14] += Number(dd_g_134[4]);
    
    values1[15] += Number(dd_g_135[0]);
    values1[16] += Number(dd_g_135[1]);
    values1[17] += Number(dd_g_135[2]);
    values1[18] += Number(dd_g_135[3]);
    values1[19] += Number(dd_g_135[4]);
    values1[20] += Number(dd_g_135[5]);
    
    values1[21] += Number(dd_g_136[0]);
    values1[22] += Number(dd_g_136[1]);
    values1[23] += Number(dd_g_136[2]);
    values1[24] += Number(dd_g_136[3]);
    values1[25] += Number(dd_g_136[4]);
    values1[26] += Number(dd_g_136[5]);
    values1[27] += Number(dd_g_136[6]);
    
    values1[28] += Number(dd_g_137[0]);
    values1[29] += Number(dd_g_137[1]);
    values1[30] += Number(dd_g_137[2]);
    values1[31] += Number(dd_g_137[3]);
    values1[32] += Number(dd_g_137[4]);
    values1[33] += Number(dd_g_137[5]);
    values1[34] += Number(dd_g_137[6]);
    values1[35] += Number(dd_g_137[7]);
    
    values1[36] += Number(dd_g_138[0]);
    values1[37] += Number(dd_g_138[1]);
    values1[38] += Number(dd_g_138[2]);
    values1[39] += Number(dd_g_138[3]);
    values1[40] += Number(dd_g_138[4]);
    values1[41] += Number(dd_g_138[5]);
    values1[42] += Number(dd_g_138[6]);
    values1[43] += Number(dd_g_138[7]);
    values1[44] += Number(dd_g_138[8]);


    


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


    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    double* values1 = (double*) calloc(n*n,sizeof(double));

    auto d_f = derivativesx(f(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));

    
    auto dd_f0 = derivativesx(d_f[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f1 = derivativesx(d_f[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f2 = derivativesx(d_f[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f3 = derivativesx(d_f[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f4 = derivativesx(d_f[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f5 = derivativesx(d_f[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f6 = derivativesx(d_f[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f7 = derivativesx(d_f[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_f8 = derivativesx(d_f[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    

    
    
    
    values1[0] = Number(dd_f0[0]) * obj_factor;
    
    values1[1] = Number(dd_f1[0]) * obj_factor;
    values1[2] = Number(dd_f1[1]) * obj_factor;
    
    values1[3] = Number(dd_f2[0]) * obj_factor;
    values1[4] = Number(dd_f2[1]) * obj_factor;
    values1[5] = Number(dd_f2[2]) * obj_factor;
    
    values1[6] = Number(dd_f3[0]) * obj_factor;
    values1[7] = Number(dd_f3[1]) * obj_factor;
    values1[8] = Number(dd_f3[2]) * obj_factor;
    values1[9] = Number(dd_f3[3]) * obj_factor;
    
    values1[10] = Number(dd_f4[0]) * obj_factor;
    values1[11] = Number(dd_f4[1]) * obj_factor;
    values1[12] = Number(dd_f4[2]) * obj_factor;
    values1[13] = Number(dd_f4[3]) * obj_factor;
    values1[14] = Number(dd_f4[4]) * obj_factor;
    
    values1[15] = Number(dd_f5[0]) * obj_factor;
    values1[16] = Number(dd_f5[1]) * obj_factor;
    values1[17] = Number(dd_f5[2]) * obj_factor;
    values1[18] = Number(dd_f5[3]) * obj_factor;
    values1[19] = Number(dd_f5[4]) * obj_factor;
    values1[20] = Number(dd_f5[5]) * obj_factor;
    
    values1[21] = Number(dd_f6[0]) * obj_factor;
    values1[22] = Number(dd_f6[1]) * obj_factor;
    values1[23] = Number(dd_f6[2]) * obj_factor;
    values1[24] = Number(dd_f6[3]) * obj_factor;
    values1[25] = Number(dd_f6[4]) * obj_factor;
    values1[26] = Number(dd_f6[5]) * obj_factor;
    values1[27] = Number(dd_f6[6]) * obj_factor;
    
    values1[28] = Number(dd_f7[0]) * obj_factor;
    values1[29] = Number(dd_f7[1]) * obj_factor;
    values1[30] = Number(dd_f7[2]) * obj_factor;
    values1[31] = Number(dd_f7[3]) * obj_factor;
    values1[32] = Number(dd_f7[4]) * obj_factor;
    values1[33] = Number(dd_f7[5]) * obj_factor;
    values1[34] = Number(dd_f7[6]) * obj_factor;
    values1[35] = Number(dd_f7[7]) * obj_factor;
    
    values1[36] = Number(dd_f8[0]) * obj_factor;
    values1[37] = Number(dd_f8[1]) * obj_factor;
    values1[38] = Number(dd_f8[2]) * obj_factor;
    values1[39] = Number(dd_f8[3]) * obj_factor;
    values1[40] = Number(dd_f8[4]) * obj_factor;
    values1[41] = Number(dd_f8[5]) * obj_factor;
    values1[42] = Number(dd_f8[6]) * obj_factor;
    values1[43] = Number(dd_f8[7]) * obj_factor;
    values1[44] = Number(dd_f8[8]) * obj_factor;

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_00 = derivativesx(d_g_0[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_01 = derivativesx(d_g_0[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_02 = derivativesx(d_g_0[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_03 = derivativesx(d_g_0[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_04 = derivativesx(d_g_0[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_05 = derivativesx(d_g_0[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_06 = derivativesx(d_g_0[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_07 = derivativesx(d_g_0[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_08 = derivativesx(d_g_0[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_00[0]) * lambda[0];
    
    values1[1] += Number(dd_g_01[0]) * lambda[0];
    values1[2] += Number(dd_g_01[1]) * lambda[0];
    
    values1[3] += Number(dd_g_02[0]) * lambda[0];
    values1[4] += Number(dd_g_02[1]) * lambda[0];
    values1[5] += Number(dd_g_02[2]) * lambda[0];
    
    values1[6] += Number(dd_g_03[0]) * lambda[0];
    values1[7] += Number(dd_g_03[1]) * lambda[0];
    values1[8] += Number(dd_g_03[2]) * lambda[0];
    values1[9] += Number(dd_g_03[3]) * lambda[0];
    
    values1[10] += Number(dd_g_04[0]) * lambda[0];
    values1[11] += Number(dd_g_04[1]) * lambda[0];
    values1[12] += Number(dd_g_04[2]) * lambda[0];
    values1[13] += Number(dd_g_04[3]) * lambda[0];
    values1[14] += Number(dd_g_04[4]) * lambda[0];
    
    values1[15] += Number(dd_g_05[0]) * lambda[0];
    values1[16] += Number(dd_g_05[1]) * lambda[0];
    values1[17] += Number(dd_g_05[2]) * lambda[0];
    values1[18] += Number(dd_g_05[3]) * lambda[0];
    values1[19] += Number(dd_g_05[4]) * lambda[0];
    values1[20] += Number(dd_g_05[5]) * lambda[0];
    
    values1[21] += Number(dd_g_06[0]) * lambda[0];
    values1[22] += Number(dd_g_06[1]) * lambda[0];
    values1[23] += Number(dd_g_06[2]) * lambda[0];
    values1[24] += Number(dd_g_06[3]) * lambda[0];
    values1[25] += Number(dd_g_06[4]) * lambda[0];
    values1[26] += Number(dd_g_06[5]) * lambda[0];
    values1[27] += Number(dd_g_06[6]) * lambda[0];
    
    values1[28] += Number(dd_g_07[0]) * lambda[0];
    values1[29] += Number(dd_g_07[1]) * lambda[0];
    values1[30] += Number(dd_g_07[2]) * lambda[0];
    values1[31] += Number(dd_g_07[3]) * lambda[0];
    values1[32] += Number(dd_g_07[4]) * lambda[0];
    values1[33] += Number(dd_g_07[5]) * lambda[0];
    values1[34] += Number(dd_g_07[6]) * lambda[0];
    values1[35] += Number(dd_g_07[7]) * lambda[0];
    
    values1[36] += Number(dd_g_08[0]) * lambda[0];
    values1[37] += Number(dd_g_08[1]) * lambda[0];
    values1[38] += Number(dd_g_08[2]) * lambda[0];
    values1[39] += Number(dd_g_08[3]) * lambda[0];
    values1[40] += Number(dd_g_08[4]) * lambda[0];
    values1[41] += Number(dd_g_08[5]) * lambda[0];
    values1[42] += Number(dd_g_08[6]) * lambda[0];
    values1[43] += Number(dd_g_08[7]) * lambda[0];
    values1[44] += Number(dd_g_08[8]) * lambda[0];


    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_10 = derivativesx(d_g_1[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_11 = derivativesx(d_g_1[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_12 = derivativesx(d_g_1[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_13 = derivativesx(d_g_1[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_14 = derivativesx(d_g_1[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_15 = derivativesx(d_g_1[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_16 = derivativesx(d_g_1[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_17 = derivativesx(d_g_1[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_18 = derivativesx(d_g_1[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_10[0]) * lambda[1];
    
    values1[1] += Number(dd_g_11[0]) * lambda[1];
    values1[2] += Number(dd_g_11[1]) * lambda[1];
    
    values1[3] += Number(dd_g_12[0]) * lambda[1];
    values1[4] += Number(dd_g_12[1]) * lambda[1];
    values1[5] += Number(dd_g_12[2]) * lambda[1];
    
    values1[6] += Number(dd_g_13[0]) * lambda[1];
    values1[7] += Number(dd_g_13[1]) * lambda[1];
    values1[8] += Number(dd_g_13[2]) * lambda[1];
    values1[9] += Number(dd_g_13[3]) * lambda[1];
    
    values1[10] += Number(dd_g_14[0]) * lambda[1];
    values1[11] += Number(dd_g_14[1]) * lambda[1];
    values1[12] += Number(dd_g_14[2]) * lambda[1];
    values1[13] += Number(dd_g_14[3]) * lambda[1];
    values1[14] += Number(dd_g_14[4]) * lambda[1];
    
    values1[15] += Number(dd_g_15[0]) * lambda[1];
    values1[16] += Number(dd_g_15[1]) * lambda[1];
    values1[17] += Number(dd_g_15[2]) * lambda[1];
    values1[18] += Number(dd_g_15[3]) * lambda[1];
    values1[19] += Number(dd_g_15[4]) * lambda[1];
    values1[20] += Number(dd_g_15[5]) * lambda[1];
    
    values1[21] += Number(dd_g_16[0]) * lambda[1];
    values1[22] += Number(dd_g_16[1]) * lambda[1];
    values1[23] += Number(dd_g_16[2]) * lambda[1];
    values1[24] += Number(dd_g_16[3]) * lambda[1];
    values1[25] += Number(dd_g_16[4]) * lambda[1];
    values1[26] += Number(dd_g_16[5]) * lambda[1];
    values1[27] += Number(dd_g_16[6]) * lambda[1];
    
    values1[28] += Number(dd_g_17[0]) * lambda[1];
    values1[29] += Number(dd_g_17[1]) * lambda[1];
    values1[30] += Number(dd_g_17[2]) * lambda[1];
    values1[31] += Number(dd_g_17[3]) * lambda[1];
    values1[32] += Number(dd_g_17[4]) * lambda[1];
    values1[33] += Number(dd_g_17[5]) * lambda[1];
    values1[34] += Number(dd_g_17[6]) * lambda[1];
    values1[35] += Number(dd_g_17[7]) * lambda[1];
    
    values1[36] += Number(dd_g_18[0]) * lambda[1];
    values1[37] += Number(dd_g_18[1]) * lambda[1];
    values1[38] += Number(dd_g_18[2]) * lambda[1];
    values1[39] += Number(dd_g_18[3]) * lambda[1];
    values1[40] += Number(dd_g_18[4]) * lambda[1];
    values1[41] += Number(dd_g_18[5]) * lambda[1];
    values1[42] += Number(dd_g_18[6]) * lambda[1];
    values1[43] += Number(dd_g_18[7]) * lambda[1];
    values1[44] += Number(dd_g_18[8]) * lambda[1];


    
    auto d_g_2 = derivativesx(g_2(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_20 = derivativesx(d_g_2[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_21 = derivativesx(d_g_2[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_22 = derivativesx(d_g_2[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_23 = derivativesx(d_g_2[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_24 = derivativesx(d_g_2[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_25 = derivativesx(d_g_2[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_26 = derivativesx(d_g_2[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_27 = derivativesx(d_g_2[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_28 = derivativesx(d_g_2[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_20[0]) * lambda[2];
    
    values1[1] += Number(dd_g_21[0]) * lambda[2];
    values1[2] += Number(dd_g_21[1]) * lambda[2];
    
    values1[3] += Number(dd_g_22[0]) * lambda[2];
    values1[4] += Number(dd_g_22[1]) * lambda[2];
    values1[5] += Number(dd_g_22[2]) * lambda[2];
    
    values1[6] += Number(dd_g_23[0]) * lambda[2];
    values1[7] += Number(dd_g_23[1]) * lambda[2];
    values1[8] += Number(dd_g_23[2]) * lambda[2];
    values1[9] += Number(dd_g_23[3]) * lambda[2];
    
    values1[10] += Number(dd_g_24[0]) * lambda[2];
    values1[11] += Number(dd_g_24[1]) * lambda[2];
    values1[12] += Number(dd_g_24[2]) * lambda[2];
    values1[13] += Number(dd_g_24[3]) * lambda[2];
    values1[14] += Number(dd_g_24[4]) * lambda[2];
    
    values1[15] += Number(dd_g_25[0]) * lambda[2];
    values1[16] += Number(dd_g_25[1]) * lambda[2];
    values1[17] += Number(dd_g_25[2]) * lambda[2];
    values1[18] += Number(dd_g_25[3]) * lambda[2];
    values1[19] += Number(dd_g_25[4]) * lambda[2];
    values1[20] += Number(dd_g_25[5]) * lambda[2];
    
    values1[21] += Number(dd_g_26[0]) * lambda[2];
    values1[22] += Number(dd_g_26[1]) * lambda[2];
    values1[23] += Number(dd_g_26[2]) * lambda[2];
    values1[24] += Number(dd_g_26[3]) * lambda[2];
    values1[25] += Number(dd_g_26[4]) * lambda[2];
    values1[26] += Number(dd_g_26[5]) * lambda[2];
    values1[27] += Number(dd_g_26[6]) * lambda[2];
    
    values1[28] += Number(dd_g_27[0]) * lambda[2];
    values1[29] += Number(dd_g_27[1]) * lambda[2];
    values1[30] += Number(dd_g_27[2]) * lambda[2];
    values1[31] += Number(dd_g_27[3]) * lambda[2];
    values1[32] += Number(dd_g_27[4]) * lambda[2];
    values1[33] += Number(dd_g_27[5]) * lambda[2];
    values1[34] += Number(dd_g_27[6]) * lambda[2];
    values1[35] += Number(dd_g_27[7]) * lambda[2];
    
    values1[36] += Number(dd_g_28[0]) * lambda[2];
    values1[37] += Number(dd_g_28[1]) * lambda[2];
    values1[38] += Number(dd_g_28[2]) * lambda[2];
    values1[39] += Number(dd_g_28[3]) * lambda[2];
    values1[40] += Number(dd_g_28[4]) * lambda[2];
    values1[41] += Number(dd_g_28[5]) * lambda[2];
    values1[42] += Number(dd_g_28[6]) * lambda[2];
    values1[43] += Number(dd_g_28[7]) * lambda[2];
    values1[44] += Number(dd_g_28[8]) * lambda[2];


    
    auto d_g_3 = derivativesx(g_3(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_30 = derivativesx(d_g_3[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_31 = derivativesx(d_g_3[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_32 = derivativesx(d_g_3[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_33 = derivativesx(d_g_3[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_34 = derivativesx(d_g_3[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_35 = derivativesx(d_g_3[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_36 = derivativesx(d_g_3[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_37 = derivativesx(d_g_3[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_38 = derivativesx(d_g_3[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_30[0]) * lambda[3];
    
    values1[1] += Number(dd_g_31[0]) * lambda[3];
    values1[2] += Number(dd_g_31[1]) * lambda[3];
    
    values1[3] += Number(dd_g_32[0]) * lambda[3];
    values1[4] += Number(dd_g_32[1]) * lambda[3];
    values1[5] += Number(dd_g_32[2]) * lambda[3];
    
    values1[6] += Number(dd_g_33[0]) * lambda[3];
    values1[7] += Number(dd_g_33[1]) * lambda[3];
    values1[8] += Number(dd_g_33[2]) * lambda[3];
    values1[9] += Number(dd_g_33[3]) * lambda[3];
    
    values1[10] += Number(dd_g_34[0]) * lambda[3];
    values1[11] += Number(dd_g_34[1]) * lambda[3];
    values1[12] += Number(dd_g_34[2]) * lambda[3];
    values1[13] += Number(dd_g_34[3]) * lambda[3];
    values1[14] += Number(dd_g_34[4]) * lambda[3];
    
    values1[15] += Number(dd_g_35[0]) * lambda[3];
    values1[16] += Number(dd_g_35[1]) * lambda[3];
    values1[17] += Number(dd_g_35[2]) * lambda[3];
    values1[18] += Number(dd_g_35[3]) * lambda[3];
    values1[19] += Number(dd_g_35[4]) * lambda[3];
    values1[20] += Number(dd_g_35[5]) * lambda[3];
    
    values1[21] += Number(dd_g_36[0]) * lambda[3];
    values1[22] += Number(dd_g_36[1]) * lambda[3];
    values1[23] += Number(dd_g_36[2]) * lambda[3];
    values1[24] += Number(dd_g_36[3]) * lambda[3];
    values1[25] += Number(dd_g_36[4]) * lambda[3];
    values1[26] += Number(dd_g_36[5]) * lambda[3];
    values1[27] += Number(dd_g_36[6]) * lambda[3];
    
    values1[28] += Number(dd_g_37[0]) * lambda[3];
    values1[29] += Number(dd_g_37[1]) * lambda[3];
    values1[30] += Number(dd_g_37[2]) * lambda[3];
    values1[31] += Number(dd_g_37[3]) * lambda[3];
    values1[32] += Number(dd_g_37[4]) * lambda[3];
    values1[33] += Number(dd_g_37[5]) * lambda[3];
    values1[34] += Number(dd_g_37[6]) * lambda[3];
    values1[35] += Number(dd_g_37[7]) * lambda[3];
    
    values1[36] += Number(dd_g_38[0]) * lambda[3];
    values1[37] += Number(dd_g_38[1]) * lambda[3];
    values1[38] += Number(dd_g_38[2]) * lambda[3];
    values1[39] += Number(dd_g_38[3]) * lambda[3];
    values1[40] += Number(dd_g_38[4]) * lambda[3];
    values1[41] += Number(dd_g_38[5]) * lambda[3];
    values1[42] += Number(dd_g_38[6]) * lambda[3];
    values1[43] += Number(dd_g_38[7]) * lambda[3];
    values1[44] += Number(dd_g_38[8]) * lambda[3];


    
    auto d_g_4 = derivativesx(g_4(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_40 = derivativesx(d_g_4[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_41 = derivativesx(d_g_4[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_42 = derivativesx(d_g_4[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_43 = derivativesx(d_g_4[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_44 = derivativesx(d_g_4[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_45 = derivativesx(d_g_4[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_46 = derivativesx(d_g_4[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_47 = derivativesx(d_g_4[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_48 = derivativesx(d_g_4[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_40[0]) * lambda[4];
    
    values1[1] += Number(dd_g_41[0]) * lambda[4];
    values1[2] += Number(dd_g_41[1]) * lambda[4];
    
    values1[3] += Number(dd_g_42[0]) * lambda[4];
    values1[4] += Number(dd_g_42[1]) * lambda[4];
    values1[5] += Number(dd_g_42[2]) * lambda[4];
    
    values1[6] += Number(dd_g_43[0]) * lambda[4];
    values1[7] += Number(dd_g_43[1]) * lambda[4];
    values1[8] += Number(dd_g_43[2]) * lambda[4];
    values1[9] += Number(dd_g_43[3]) * lambda[4];
    
    values1[10] += Number(dd_g_44[0]) * lambda[4];
    values1[11] += Number(dd_g_44[1]) * lambda[4];
    values1[12] += Number(dd_g_44[2]) * lambda[4];
    values1[13] += Number(dd_g_44[3]) * lambda[4];
    values1[14] += Number(dd_g_44[4]) * lambda[4];
    
    values1[15] += Number(dd_g_45[0]) * lambda[4];
    values1[16] += Number(dd_g_45[1]) * lambda[4];
    values1[17] += Number(dd_g_45[2]) * lambda[4];
    values1[18] += Number(dd_g_45[3]) * lambda[4];
    values1[19] += Number(dd_g_45[4]) * lambda[4];
    values1[20] += Number(dd_g_45[5]) * lambda[4];
    
    values1[21] += Number(dd_g_46[0]) * lambda[4];
    values1[22] += Number(dd_g_46[1]) * lambda[4];
    values1[23] += Number(dd_g_46[2]) * lambda[4];
    values1[24] += Number(dd_g_46[3]) * lambda[4];
    values1[25] += Number(dd_g_46[4]) * lambda[4];
    values1[26] += Number(dd_g_46[5]) * lambda[4];
    values1[27] += Number(dd_g_46[6]) * lambda[4];
    
    values1[28] += Number(dd_g_47[0]) * lambda[4];
    values1[29] += Number(dd_g_47[1]) * lambda[4];
    values1[30] += Number(dd_g_47[2]) * lambda[4];
    values1[31] += Number(dd_g_47[3]) * lambda[4];
    values1[32] += Number(dd_g_47[4]) * lambda[4];
    values1[33] += Number(dd_g_47[5]) * lambda[4];
    values1[34] += Number(dd_g_47[6]) * lambda[4];
    values1[35] += Number(dd_g_47[7]) * lambda[4];
    
    values1[36] += Number(dd_g_48[0]) * lambda[4];
    values1[37] += Number(dd_g_48[1]) * lambda[4];
    values1[38] += Number(dd_g_48[2]) * lambda[4];
    values1[39] += Number(dd_g_48[3]) * lambda[4];
    values1[40] += Number(dd_g_48[4]) * lambda[4];
    values1[41] += Number(dd_g_48[5]) * lambda[4];
    values1[42] += Number(dd_g_48[6]) * lambda[4];
    values1[43] += Number(dd_g_48[7]) * lambda[4];
    values1[44] += Number(dd_g_48[8]) * lambda[4];


    
    auto d_g_5 = derivativesx(g_5(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_50 = derivativesx(d_g_5[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_51 = derivativesx(d_g_5[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_52 = derivativesx(d_g_5[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_53 = derivativesx(d_g_5[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_54 = derivativesx(d_g_5[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_55 = derivativesx(d_g_5[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_56 = derivativesx(d_g_5[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_57 = derivativesx(d_g_5[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_58 = derivativesx(d_g_5[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_50[0]) * lambda[5];
    
    values1[1] += Number(dd_g_51[0]) * lambda[5];
    values1[2] += Number(dd_g_51[1]) * lambda[5];
    
    values1[3] += Number(dd_g_52[0]) * lambda[5];
    values1[4] += Number(dd_g_52[1]) * lambda[5];
    values1[5] += Number(dd_g_52[2]) * lambda[5];
    
    values1[6] += Number(dd_g_53[0]) * lambda[5];
    values1[7] += Number(dd_g_53[1]) * lambda[5];
    values1[8] += Number(dd_g_53[2]) * lambda[5];
    values1[9] += Number(dd_g_53[3]) * lambda[5];
    
    values1[10] += Number(dd_g_54[0]) * lambda[5];
    values1[11] += Number(dd_g_54[1]) * lambda[5];
    values1[12] += Number(dd_g_54[2]) * lambda[5];
    values1[13] += Number(dd_g_54[3]) * lambda[5];
    values1[14] += Number(dd_g_54[4]) * lambda[5];
    
    values1[15] += Number(dd_g_55[0]) * lambda[5];
    values1[16] += Number(dd_g_55[1]) * lambda[5];
    values1[17] += Number(dd_g_55[2]) * lambda[5];
    values1[18] += Number(dd_g_55[3]) * lambda[5];
    values1[19] += Number(dd_g_55[4]) * lambda[5];
    values1[20] += Number(dd_g_55[5]) * lambda[5];
    
    values1[21] += Number(dd_g_56[0]) * lambda[5];
    values1[22] += Number(dd_g_56[1]) * lambda[5];
    values1[23] += Number(dd_g_56[2]) * lambda[5];
    values1[24] += Number(dd_g_56[3]) * lambda[5];
    values1[25] += Number(dd_g_56[4]) * lambda[5];
    values1[26] += Number(dd_g_56[5]) * lambda[5];
    values1[27] += Number(dd_g_56[6]) * lambda[5];
    
    values1[28] += Number(dd_g_57[0]) * lambda[5];
    values1[29] += Number(dd_g_57[1]) * lambda[5];
    values1[30] += Number(dd_g_57[2]) * lambda[5];
    values1[31] += Number(dd_g_57[3]) * lambda[5];
    values1[32] += Number(dd_g_57[4]) * lambda[5];
    values1[33] += Number(dd_g_57[5]) * lambda[5];
    values1[34] += Number(dd_g_57[6]) * lambda[5];
    values1[35] += Number(dd_g_57[7]) * lambda[5];
    
    values1[36] += Number(dd_g_58[0]) * lambda[5];
    values1[37] += Number(dd_g_58[1]) * lambda[5];
    values1[38] += Number(dd_g_58[2]) * lambda[5];
    values1[39] += Number(dd_g_58[3]) * lambda[5];
    values1[40] += Number(dd_g_58[4]) * lambda[5];
    values1[41] += Number(dd_g_58[5]) * lambda[5];
    values1[42] += Number(dd_g_58[6]) * lambda[5];
    values1[43] += Number(dd_g_58[7]) * lambda[5];
    values1[44] += Number(dd_g_58[8]) * lambda[5];


    
    auto d_g_6 = derivativesx(g_6(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_60 = derivativesx(d_g_6[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_61 = derivativesx(d_g_6[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_62 = derivativesx(d_g_6[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_63 = derivativesx(d_g_6[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_64 = derivativesx(d_g_6[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_65 = derivativesx(d_g_6[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_66 = derivativesx(d_g_6[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_67 = derivativesx(d_g_6[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_68 = derivativesx(d_g_6[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_60[0]) * lambda[6];
    
    values1[1] += Number(dd_g_61[0]) * lambda[6];
    values1[2] += Number(dd_g_61[1]) * lambda[6];
    
    values1[3] += Number(dd_g_62[0]) * lambda[6];
    values1[4] += Number(dd_g_62[1]) * lambda[6];
    values1[5] += Number(dd_g_62[2]) * lambda[6];
    
    values1[6] += Number(dd_g_63[0]) * lambda[6];
    values1[7] += Number(dd_g_63[1]) * lambda[6];
    values1[8] += Number(dd_g_63[2]) * lambda[6];
    values1[9] += Number(dd_g_63[3]) * lambda[6];
    
    values1[10] += Number(dd_g_64[0]) * lambda[6];
    values1[11] += Number(dd_g_64[1]) * lambda[6];
    values1[12] += Number(dd_g_64[2]) * lambda[6];
    values1[13] += Number(dd_g_64[3]) * lambda[6];
    values1[14] += Number(dd_g_64[4]) * lambda[6];
    
    values1[15] += Number(dd_g_65[0]) * lambda[6];
    values1[16] += Number(dd_g_65[1]) * lambda[6];
    values1[17] += Number(dd_g_65[2]) * lambda[6];
    values1[18] += Number(dd_g_65[3]) * lambda[6];
    values1[19] += Number(dd_g_65[4]) * lambda[6];
    values1[20] += Number(dd_g_65[5]) * lambda[6];
    
    values1[21] += Number(dd_g_66[0]) * lambda[6];
    values1[22] += Number(dd_g_66[1]) * lambda[6];
    values1[23] += Number(dd_g_66[2]) * lambda[6];
    values1[24] += Number(dd_g_66[3]) * lambda[6];
    values1[25] += Number(dd_g_66[4]) * lambda[6];
    values1[26] += Number(dd_g_66[5]) * lambda[6];
    values1[27] += Number(dd_g_66[6]) * lambda[6];
    
    values1[28] += Number(dd_g_67[0]) * lambda[6];
    values1[29] += Number(dd_g_67[1]) * lambda[6];
    values1[30] += Number(dd_g_67[2]) * lambda[6];
    values1[31] += Number(dd_g_67[3]) * lambda[6];
    values1[32] += Number(dd_g_67[4]) * lambda[6];
    values1[33] += Number(dd_g_67[5]) * lambda[6];
    values1[34] += Number(dd_g_67[6]) * lambda[6];
    values1[35] += Number(dd_g_67[7]) * lambda[6];
    
    values1[36] += Number(dd_g_68[0]) * lambda[6];
    values1[37] += Number(dd_g_68[1]) * lambda[6];
    values1[38] += Number(dd_g_68[2]) * lambda[6];
    values1[39] += Number(dd_g_68[3]) * lambda[6];
    values1[40] += Number(dd_g_68[4]) * lambda[6];
    values1[41] += Number(dd_g_68[5]) * lambda[6];
    values1[42] += Number(dd_g_68[6]) * lambda[6];
    values1[43] += Number(dd_g_68[7]) * lambda[6];
    values1[44] += Number(dd_g_68[8]) * lambda[6];


    
    auto d_g_7 = derivativesx(g_7(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_70 = derivativesx(d_g_7[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_71 = derivativesx(d_g_7[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_72 = derivativesx(d_g_7[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_73 = derivativesx(d_g_7[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_74 = derivativesx(d_g_7[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_75 = derivativesx(d_g_7[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_76 = derivativesx(d_g_7[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_77 = derivativesx(d_g_7[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_78 = derivativesx(d_g_7[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_70[0]) * lambda[7];
    
    values1[1] += Number(dd_g_71[0]) * lambda[7];
    values1[2] += Number(dd_g_71[1]) * lambda[7];
    
    values1[3] += Number(dd_g_72[0]) * lambda[7];
    values1[4] += Number(dd_g_72[1]) * lambda[7];
    values1[5] += Number(dd_g_72[2]) * lambda[7];
    
    values1[6] += Number(dd_g_73[0]) * lambda[7];
    values1[7] += Number(dd_g_73[1]) * lambda[7];
    values1[8] += Number(dd_g_73[2]) * lambda[7];
    values1[9] += Number(dd_g_73[3]) * lambda[7];
    
    values1[10] += Number(dd_g_74[0]) * lambda[7];
    values1[11] += Number(dd_g_74[1]) * lambda[7];
    values1[12] += Number(dd_g_74[2]) * lambda[7];
    values1[13] += Number(dd_g_74[3]) * lambda[7];
    values1[14] += Number(dd_g_74[4]) * lambda[7];
    
    values1[15] += Number(dd_g_75[0]) * lambda[7];
    values1[16] += Number(dd_g_75[1]) * lambda[7];
    values1[17] += Number(dd_g_75[2]) * lambda[7];
    values1[18] += Number(dd_g_75[3]) * lambda[7];
    values1[19] += Number(dd_g_75[4]) * lambda[7];
    values1[20] += Number(dd_g_75[5]) * lambda[7];
    
    values1[21] += Number(dd_g_76[0]) * lambda[7];
    values1[22] += Number(dd_g_76[1]) * lambda[7];
    values1[23] += Number(dd_g_76[2]) * lambda[7];
    values1[24] += Number(dd_g_76[3]) * lambda[7];
    values1[25] += Number(dd_g_76[4]) * lambda[7];
    values1[26] += Number(dd_g_76[5]) * lambda[7];
    values1[27] += Number(dd_g_76[6]) * lambda[7];
    
    values1[28] += Number(dd_g_77[0]) * lambda[7];
    values1[29] += Number(dd_g_77[1]) * lambda[7];
    values1[30] += Number(dd_g_77[2]) * lambda[7];
    values1[31] += Number(dd_g_77[3]) * lambda[7];
    values1[32] += Number(dd_g_77[4]) * lambda[7];
    values1[33] += Number(dd_g_77[5]) * lambda[7];
    values1[34] += Number(dd_g_77[6]) * lambda[7];
    values1[35] += Number(dd_g_77[7]) * lambda[7];
    
    values1[36] += Number(dd_g_78[0]) * lambda[7];
    values1[37] += Number(dd_g_78[1]) * lambda[7];
    values1[38] += Number(dd_g_78[2]) * lambda[7];
    values1[39] += Number(dd_g_78[3]) * lambda[7];
    values1[40] += Number(dd_g_78[4]) * lambda[7];
    values1[41] += Number(dd_g_78[5]) * lambda[7];
    values1[42] += Number(dd_g_78[6]) * lambda[7];
    values1[43] += Number(dd_g_78[7]) * lambda[7];
    values1[44] += Number(dd_g_78[8]) * lambda[7];


    
    auto d_g_8 = derivativesx(g_8(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_80 = derivativesx(d_g_8[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_81 = derivativesx(d_g_8[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_82 = derivativesx(d_g_8[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_83 = derivativesx(d_g_8[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_84 = derivativesx(d_g_8[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_85 = derivativesx(d_g_8[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_86 = derivativesx(d_g_8[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_87 = derivativesx(d_g_8[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_88 = derivativesx(d_g_8[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_80[0]) * lambda[8];
    
    values1[1] += Number(dd_g_81[0]) * lambda[8];
    values1[2] += Number(dd_g_81[1]) * lambda[8];
    
    values1[3] += Number(dd_g_82[0]) * lambda[8];
    values1[4] += Number(dd_g_82[1]) * lambda[8];
    values1[5] += Number(dd_g_82[2]) * lambda[8];
    
    values1[6] += Number(dd_g_83[0]) * lambda[8];
    values1[7] += Number(dd_g_83[1]) * lambda[8];
    values1[8] += Number(dd_g_83[2]) * lambda[8];
    values1[9] += Number(dd_g_83[3]) * lambda[8];
    
    values1[10] += Number(dd_g_84[0]) * lambda[8];
    values1[11] += Number(dd_g_84[1]) * lambda[8];
    values1[12] += Number(dd_g_84[2]) * lambda[8];
    values1[13] += Number(dd_g_84[3]) * lambda[8];
    values1[14] += Number(dd_g_84[4]) * lambda[8];
    
    values1[15] += Number(dd_g_85[0]) * lambda[8];
    values1[16] += Number(dd_g_85[1]) * lambda[8];
    values1[17] += Number(dd_g_85[2]) * lambda[8];
    values1[18] += Number(dd_g_85[3]) * lambda[8];
    values1[19] += Number(dd_g_85[4]) * lambda[8];
    values1[20] += Number(dd_g_85[5]) * lambda[8];
    
    values1[21] += Number(dd_g_86[0]) * lambda[8];
    values1[22] += Number(dd_g_86[1]) * lambda[8];
    values1[23] += Number(dd_g_86[2]) * lambda[8];
    values1[24] += Number(dd_g_86[3]) * lambda[8];
    values1[25] += Number(dd_g_86[4]) * lambda[8];
    values1[26] += Number(dd_g_86[5]) * lambda[8];
    values1[27] += Number(dd_g_86[6]) * lambda[8];
    
    values1[28] += Number(dd_g_87[0]) * lambda[8];
    values1[29] += Number(dd_g_87[1]) * lambda[8];
    values1[30] += Number(dd_g_87[2]) * lambda[8];
    values1[31] += Number(dd_g_87[3]) * lambda[8];
    values1[32] += Number(dd_g_87[4]) * lambda[8];
    values1[33] += Number(dd_g_87[5]) * lambda[8];
    values1[34] += Number(dd_g_87[6]) * lambda[8];
    values1[35] += Number(dd_g_87[7]) * lambda[8];
    
    values1[36] += Number(dd_g_88[0]) * lambda[8];
    values1[37] += Number(dd_g_88[1]) * lambda[8];
    values1[38] += Number(dd_g_88[2]) * lambda[8];
    values1[39] += Number(dd_g_88[3]) * lambda[8];
    values1[40] += Number(dd_g_88[4]) * lambda[8];
    values1[41] += Number(dd_g_88[5]) * lambda[8];
    values1[42] += Number(dd_g_88[6]) * lambda[8];
    values1[43] += Number(dd_g_88[7]) * lambda[8];
    values1[44] += Number(dd_g_88[8]) * lambda[8];


    
    auto d_g_9 = derivativesx(g_9(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_90 = derivativesx(d_g_9[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_91 = derivativesx(d_g_9[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_92 = derivativesx(d_g_9[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_93 = derivativesx(d_g_9[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_94 = derivativesx(d_g_9[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_95 = derivativesx(d_g_9[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_96 = derivativesx(d_g_9[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_97 = derivativesx(d_g_9[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_98 = derivativesx(d_g_9[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_90[0]) * lambda[9];
    
    values1[1] += Number(dd_g_91[0]) * lambda[9];
    values1[2] += Number(dd_g_91[1]) * lambda[9];
    
    values1[3] += Number(dd_g_92[0]) * lambda[9];
    values1[4] += Number(dd_g_92[1]) * lambda[9];
    values1[5] += Number(dd_g_92[2]) * lambda[9];
    
    values1[6] += Number(dd_g_93[0]) * lambda[9];
    values1[7] += Number(dd_g_93[1]) * lambda[9];
    values1[8] += Number(dd_g_93[2]) * lambda[9];
    values1[9] += Number(dd_g_93[3]) * lambda[9];
    
    values1[10] += Number(dd_g_94[0]) * lambda[9];
    values1[11] += Number(dd_g_94[1]) * lambda[9];
    values1[12] += Number(dd_g_94[2]) * lambda[9];
    values1[13] += Number(dd_g_94[3]) * lambda[9];
    values1[14] += Number(dd_g_94[4]) * lambda[9];
    
    values1[15] += Number(dd_g_95[0]) * lambda[9];
    values1[16] += Number(dd_g_95[1]) * lambda[9];
    values1[17] += Number(dd_g_95[2]) * lambda[9];
    values1[18] += Number(dd_g_95[3]) * lambda[9];
    values1[19] += Number(dd_g_95[4]) * lambda[9];
    values1[20] += Number(dd_g_95[5]) * lambda[9];
    
    values1[21] += Number(dd_g_96[0]) * lambda[9];
    values1[22] += Number(dd_g_96[1]) * lambda[9];
    values1[23] += Number(dd_g_96[2]) * lambda[9];
    values1[24] += Number(dd_g_96[3]) * lambda[9];
    values1[25] += Number(dd_g_96[4]) * lambda[9];
    values1[26] += Number(dd_g_96[5]) * lambda[9];
    values1[27] += Number(dd_g_96[6]) * lambda[9];
    
    values1[28] += Number(dd_g_97[0]) * lambda[9];
    values1[29] += Number(dd_g_97[1]) * lambda[9];
    values1[30] += Number(dd_g_97[2]) * lambda[9];
    values1[31] += Number(dd_g_97[3]) * lambda[9];
    values1[32] += Number(dd_g_97[4]) * lambda[9];
    values1[33] += Number(dd_g_97[5]) * lambda[9];
    values1[34] += Number(dd_g_97[6]) * lambda[9];
    values1[35] += Number(dd_g_97[7]) * lambda[9];
    
    values1[36] += Number(dd_g_98[0]) * lambda[9];
    values1[37] += Number(dd_g_98[1]) * lambda[9];
    values1[38] += Number(dd_g_98[2]) * lambda[9];
    values1[39] += Number(dd_g_98[3]) * lambda[9];
    values1[40] += Number(dd_g_98[4]) * lambda[9];
    values1[41] += Number(dd_g_98[5]) * lambda[9];
    values1[42] += Number(dd_g_98[6]) * lambda[9];
    values1[43] += Number(dd_g_98[7]) * lambda[9];
    values1[44] += Number(dd_g_98[8]) * lambda[9];


    
    auto d_g_10 = derivativesx(g_10(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_100 = derivativesx(d_g_10[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_101 = derivativesx(d_g_10[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_102 = derivativesx(d_g_10[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_103 = derivativesx(d_g_10[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_104 = derivativesx(d_g_10[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_105 = derivativesx(d_g_10[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_106 = derivativesx(d_g_10[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_107 = derivativesx(d_g_10[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_108 = derivativesx(d_g_10[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_100[0]) * lambda[10];
    
    values1[1] += Number(dd_g_101[0]) * lambda[10];
    values1[2] += Number(dd_g_101[1]) * lambda[10];
    
    values1[3] += Number(dd_g_102[0]) * lambda[10];
    values1[4] += Number(dd_g_102[1]) * lambda[10];
    values1[5] += Number(dd_g_102[2]) * lambda[10];
    
    values1[6] += Number(dd_g_103[0]) * lambda[10];
    values1[7] += Number(dd_g_103[1]) * lambda[10];
    values1[8] += Number(dd_g_103[2]) * lambda[10];
    values1[9] += Number(dd_g_103[3]) * lambda[10];
    
    values1[10] += Number(dd_g_104[0]) * lambda[10];
    values1[11] += Number(dd_g_104[1]) * lambda[10];
    values1[12] += Number(dd_g_104[2]) * lambda[10];
    values1[13] += Number(dd_g_104[3]) * lambda[10];
    values1[14] += Number(dd_g_104[4]) * lambda[10];
    
    values1[15] += Number(dd_g_105[0]) * lambda[10];
    values1[16] += Number(dd_g_105[1]) * lambda[10];
    values1[17] += Number(dd_g_105[2]) * lambda[10];
    values1[18] += Number(dd_g_105[3]) * lambda[10];
    values1[19] += Number(dd_g_105[4]) * lambda[10];
    values1[20] += Number(dd_g_105[5]) * lambda[10];
    
    values1[21] += Number(dd_g_106[0]) * lambda[10];
    values1[22] += Number(dd_g_106[1]) * lambda[10];
    values1[23] += Number(dd_g_106[2]) * lambda[10];
    values1[24] += Number(dd_g_106[3]) * lambda[10];
    values1[25] += Number(dd_g_106[4]) * lambda[10];
    values1[26] += Number(dd_g_106[5]) * lambda[10];
    values1[27] += Number(dd_g_106[6]) * lambda[10];
    
    values1[28] += Number(dd_g_107[0]) * lambda[10];
    values1[29] += Number(dd_g_107[1]) * lambda[10];
    values1[30] += Number(dd_g_107[2]) * lambda[10];
    values1[31] += Number(dd_g_107[3]) * lambda[10];
    values1[32] += Number(dd_g_107[4]) * lambda[10];
    values1[33] += Number(dd_g_107[5]) * lambda[10];
    values1[34] += Number(dd_g_107[6]) * lambda[10];
    values1[35] += Number(dd_g_107[7]) * lambda[10];
    
    values1[36] += Number(dd_g_108[0]) * lambda[10];
    values1[37] += Number(dd_g_108[1]) * lambda[10];
    values1[38] += Number(dd_g_108[2]) * lambda[10];
    values1[39] += Number(dd_g_108[3]) * lambda[10];
    values1[40] += Number(dd_g_108[4]) * lambda[10];
    values1[41] += Number(dd_g_108[5]) * lambda[10];
    values1[42] += Number(dd_g_108[6]) * lambda[10];
    values1[43] += Number(dd_g_108[7]) * lambda[10];
    values1[44] += Number(dd_g_108[8]) * lambda[10];


    
    auto d_g_11 = derivativesx(g_11(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_110 = derivativesx(d_g_11[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_111 = derivativesx(d_g_11[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_112 = derivativesx(d_g_11[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_113 = derivativesx(d_g_11[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_114 = derivativesx(d_g_11[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_115 = derivativesx(d_g_11[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_116 = derivativesx(d_g_11[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_117 = derivativesx(d_g_11[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_118 = derivativesx(d_g_11[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_110[0]) * lambda[11];
    
    values1[1] += Number(dd_g_111[0]) * lambda[11];
    values1[2] += Number(dd_g_111[1]) * lambda[11];
    
    values1[3] += Number(dd_g_112[0]) * lambda[11];
    values1[4] += Number(dd_g_112[1]) * lambda[11];
    values1[5] += Number(dd_g_112[2]) * lambda[11];
    
    values1[6] += Number(dd_g_113[0]) * lambda[11];
    values1[7] += Number(dd_g_113[1]) * lambda[11];
    values1[8] += Number(dd_g_113[2]) * lambda[11];
    values1[9] += Number(dd_g_113[3]) * lambda[11];
    
    values1[10] += Number(dd_g_114[0]) * lambda[11];
    values1[11] += Number(dd_g_114[1]) * lambda[11];
    values1[12] += Number(dd_g_114[2]) * lambda[11];
    values1[13] += Number(dd_g_114[3]) * lambda[11];
    values1[14] += Number(dd_g_114[4]) * lambda[11];
    
    values1[15] += Number(dd_g_115[0]) * lambda[11];
    values1[16] += Number(dd_g_115[1]) * lambda[11];
    values1[17] += Number(dd_g_115[2]) * lambda[11];
    values1[18] += Number(dd_g_115[3]) * lambda[11];
    values1[19] += Number(dd_g_115[4]) * lambda[11];
    values1[20] += Number(dd_g_115[5]) * lambda[11];
    
    values1[21] += Number(dd_g_116[0]) * lambda[11];
    values1[22] += Number(dd_g_116[1]) * lambda[11];
    values1[23] += Number(dd_g_116[2]) * lambda[11];
    values1[24] += Number(dd_g_116[3]) * lambda[11];
    values1[25] += Number(dd_g_116[4]) * lambda[11];
    values1[26] += Number(dd_g_116[5]) * lambda[11];
    values1[27] += Number(dd_g_116[6]) * lambda[11];
    
    values1[28] += Number(dd_g_117[0]) * lambda[11];
    values1[29] += Number(dd_g_117[1]) * lambda[11];
    values1[30] += Number(dd_g_117[2]) * lambda[11];
    values1[31] += Number(dd_g_117[3]) * lambda[11];
    values1[32] += Number(dd_g_117[4]) * lambda[11];
    values1[33] += Number(dd_g_117[5]) * lambda[11];
    values1[34] += Number(dd_g_117[6]) * lambda[11];
    values1[35] += Number(dd_g_117[7]) * lambda[11];
    
    values1[36] += Number(dd_g_118[0]) * lambda[11];
    values1[37] += Number(dd_g_118[1]) * lambda[11];
    values1[38] += Number(dd_g_118[2]) * lambda[11];
    values1[39] += Number(dd_g_118[3]) * lambda[11];
    values1[40] += Number(dd_g_118[4]) * lambda[11];
    values1[41] += Number(dd_g_118[5]) * lambda[11];
    values1[42] += Number(dd_g_118[6]) * lambda[11];
    values1[43] += Number(dd_g_118[7]) * lambda[11];
    values1[44] += Number(dd_g_118[8]) * lambda[11];


    
    auto d_g_12 = derivativesx(g_12(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_120 = derivativesx(d_g_12[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_121 = derivativesx(d_g_12[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_122 = derivativesx(d_g_12[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_123 = derivativesx(d_g_12[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_124 = derivativesx(d_g_12[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_125 = derivativesx(d_g_12[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_126 = derivativesx(d_g_12[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_127 = derivativesx(d_g_12[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_128 = derivativesx(d_g_12[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_120[0]) * lambda[12];
    
    values1[1] += Number(dd_g_121[0]) * lambda[12];
    values1[2] += Number(dd_g_121[1]) * lambda[12];
    
    values1[3] += Number(dd_g_122[0]) * lambda[12];
    values1[4] += Number(dd_g_122[1]) * lambda[12];
    values1[5] += Number(dd_g_122[2]) * lambda[12];
    
    values1[6] += Number(dd_g_123[0]) * lambda[12];
    values1[7] += Number(dd_g_123[1]) * lambda[12];
    values1[8] += Number(dd_g_123[2]) * lambda[12];
    values1[9] += Number(dd_g_123[3]) * lambda[12];
    
    values1[10] += Number(dd_g_124[0]) * lambda[12];
    values1[11] += Number(dd_g_124[1]) * lambda[12];
    values1[12] += Number(dd_g_124[2]) * lambda[12];
    values1[13] += Number(dd_g_124[3]) * lambda[12];
    values1[14] += Number(dd_g_124[4]) * lambda[12];
    
    values1[15] += Number(dd_g_125[0]) * lambda[12];
    values1[16] += Number(dd_g_125[1]) * lambda[12];
    values1[17] += Number(dd_g_125[2]) * lambda[12];
    values1[18] += Number(dd_g_125[3]) * lambda[12];
    values1[19] += Number(dd_g_125[4]) * lambda[12];
    values1[20] += Number(dd_g_125[5]) * lambda[12];
    
    values1[21] += Number(dd_g_126[0]) * lambda[12];
    values1[22] += Number(dd_g_126[1]) * lambda[12];
    values1[23] += Number(dd_g_126[2]) * lambda[12];
    values1[24] += Number(dd_g_126[3]) * lambda[12];
    values1[25] += Number(dd_g_126[4]) * lambda[12];
    values1[26] += Number(dd_g_126[5]) * lambda[12];
    values1[27] += Number(dd_g_126[6]) * lambda[12];
    
    values1[28] += Number(dd_g_127[0]) * lambda[12];
    values1[29] += Number(dd_g_127[1]) * lambda[12];
    values1[30] += Number(dd_g_127[2]) * lambda[12];
    values1[31] += Number(dd_g_127[3]) * lambda[12];
    values1[32] += Number(dd_g_127[4]) * lambda[12];
    values1[33] += Number(dd_g_127[5]) * lambda[12];
    values1[34] += Number(dd_g_127[6]) * lambda[12];
    values1[35] += Number(dd_g_127[7]) * lambda[12];
    
    values1[36] += Number(dd_g_128[0]) * lambda[12];
    values1[37] += Number(dd_g_128[1]) * lambda[12];
    values1[38] += Number(dd_g_128[2]) * lambda[12];
    values1[39] += Number(dd_g_128[3]) * lambda[12];
    values1[40] += Number(dd_g_128[4]) * lambda[12];
    values1[41] += Number(dd_g_128[5]) * lambda[12];
    values1[42] += Number(dd_g_128[6]) * lambda[12];
    values1[43] += Number(dd_g_128[7]) * lambda[12];
    values1[44] += Number(dd_g_128[8]) * lambda[12];


    
    auto d_g_13 = derivativesx(g_13(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    auto dd_g_130 = derivativesx(d_g_13[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_131 = derivativesx(d_g_13[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_132 = derivativesx(d_g_13[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_133 = derivativesx(d_g_13[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_134 = derivativesx(d_g_13[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_135 = derivativesx(d_g_13[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_136 = derivativesx(d_g_13[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_137 = derivativesx(d_g_13[7], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    auto dd_g_138 = derivativesx(d_g_13[8], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]));
    
    
    values1[0] += Number(dd_g_130[0]) * lambda[13];
    
    values1[1] += Number(dd_g_131[0]) * lambda[13];
    values1[2] += Number(dd_g_131[1]) * lambda[13];
    
    values1[3] += Number(dd_g_132[0]) * lambda[13];
    values1[4] += Number(dd_g_132[1]) * lambda[13];
    values1[5] += Number(dd_g_132[2]) * lambda[13];
    
    values1[6] += Number(dd_g_133[0]) * lambda[13];
    values1[7] += Number(dd_g_133[1]) * lambda[13];
    values1[8] += Number(dd_g_133[2]) * lambda[13];
    values1[9] += Number(dd_g_133[3]) * lambda[13];
    
    values1[10] += Number(dd_g_134[0]) * lambda[13];
    values1[11] += Number(dd_g_134[1]) * lambda[13];
    values1[12] += Number(dd_g_134[2]) * lambda[13];
    values1[13] += Number(dd_g_134[3]) * lambda[13];
    values1[14] += Number(dd_g_134[4]) * lambda[13];
    
    values1[15] += Number(dd_g_135[0]) * lambda[13];
    values1[16] += Number(dd_g_135[1]) * lambda[13];
    values1[17] += Number(dd_g_135[2]) * lambda[13];
    values1[18] += Number(dd_g_135[3]) * lambda[13];
    values1[19] += Number(dd_g_135[4]) * lambda[13];
    values1[20] += Number(dd_g_135[5]) * lambda[13];
    
    values1[21] += Number(dd_g_136[0]) * lambda[13];
    values1[22] += Number(dd_g_136[1]) * lambda[13];
    values1[23] += Number(dd_g_136[2]) * lambda[13];
    values1[24] += Number(dd_g_136[3]) * lambda[13];
    values1[25] += Number(dd_g_136[4]) * lambda[13];
    values1[26] += Number(dd_g_136[5]) * lambda[13];
    values1[27] += Number(dd_g_136[6]) * lambda[13];
    
    values1[28] += Number(dd_g_137[0]) * lambda[13];
    values1[29] += Number(dd_g_137[1]) * lambda[13];
    values1[30] += Number(dd_g_137[2]) * lambda[13];
    values1[31] += Number(dd_g_137[3]) * lambda[13];
    values1[32] += Number(dd_g_137[4]) * lambda[13];
    values1[33] += Number(dd_g_137[5]) * lambda[13];
    values1[34] += Number(dd_g_137[6]) * lambda[13];
    values1[35] += Number(dd_g_137[7]) * lambda[13];
    
    values1[36] += Number(dd_g_138[0]) * lambda[13];
    values1[37] += Number(dd_g_138[1]) * lambda[13];
    values1[38] += Number(dd_g_138[2]) * lambda[13];
    values1[39] += Number(dd_g_138[3]) * lambda[13];
    values1[40] += Number(dd_g_138[4]) * lambda[13];
    values1[41] += Number(dd_g_138[5]) * lambda[13];
    values1[42] += Number(dd_g_138[6]) * lambda[13];
    values1[43] += Number(dd_g_138[7]) * lambda[13];
    values1[44] += Number(dd_g_138[8]) * lambda[13];


    


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