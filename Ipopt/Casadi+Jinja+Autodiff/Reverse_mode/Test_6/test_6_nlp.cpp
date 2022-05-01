// In this file I will try to use the CasADi created file to evaluate functions 

#include "test_6_nlp.hpp"

#include <cassert>
#include <iostream>

#include <autodiff/reverse/var.hpp>
using namespace autodiff;


var f(var* x)
{
  return pow(x[0]-10,2) + 5*pow(x[1]-12,2) + pow(x[2],4) + 3*pow(x[3]-11,2) + 10*pow(x[4],6) + 7*pow(x[5],2) + pow(x[6],4) - 4*x[5]*x[6] -10*x[5] - 8*x[6];
}


var g_0(var* x)
{
  return 2*pow(x[0],2) + 3*pow(x[1],4) +x[2] +4*pow(x[3],2) + 5*x[4];
}

var g_1(var* x)
{
  return -4*pow(x[0],2) -pow(x[1],2) + 3*x[0]*x[1] -2*pow(x[2],2) - 5*x[5] + 11*x[6];
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
  n = 7;

  // one equality constraint and one inequality constraint
  m = 2;

  // in this example the jacobian is dense and contains 8 nonzeros
  nnz_jac_g = 10;

  // the hessian is also dense and has 16 total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = 9;

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
  assert(n == 7);
  assert(m == 2);


  
    x_l[0] = -5;
    x_u[0] = 5;
  
    x_l[1] = -5;
    x_u[1] = 5;
  
    x_l[2] = -5;
    x_u[2] = 5;
  
    x_l[3] = -5;
    x_u[3] = 5;
  
    x_l[4] = -5;
    x_u[4] = 5;
  
    x_l[5] = -5;
    x_u[5] = 5;
  
    x_l[6] = -5;
    x_u[6] = 5;
  

  // the variables have upper bounds of 5
  
    g_l[0] = 127;
    g_u[0] = 127;
  
    g_l[1] = 0;
    g_u[1] = 0;
  

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
  
    x[1] = 2;
  
    x[2] = 0;
  
    x[3] = 4;
  
    x[4] = 0;
  
    x[5] = 1;
  
    x[6] = 1;
  

  return true;
}

// returns the value of the objective function
bool HS071_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  assert(n == 7);

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
  assert(n == 7);

  // Gradiente restituito come vettore colonna

  var* p = (var*) calloc(n, sizeof(var));
  for(int i = 0; i < n; i++) {
    p[i] = x[i];
  }

  auto d_f = derivativesx(f(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));

  for (int i = 0; i < n; i++) {
    grad_f[i] = Number(d_f[i]);
  }


  return true;
}

// return the value of the constraints: g(x)
bool HS071_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  assert(n == 7);
  assert(m == 2);


  var* p = (var*) calloc(n, sizeof(var));
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

    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = rand() % 10000 + 1;
    }

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    

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

    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    

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
   
    var* p = (var*) calloc(n, sizeof(var));
    for(int i = 0; i < n; i++) {
      p[i] = rand() % 10000 + 1;
    }

    double* values1 = (double*) calloc(n*n,sizeof(double));

    auto d_f = derivativesx(f(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));

    
    auto dd_f0 = derivativesx(d_f[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f1 = derivativesx(d_f[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f2 = derivativesx(d_f[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f3 = derivativesx(d_f[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f4 = derivativesx(d_f[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f5 = derivativesx(d_f[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f6 = derivativesx(d_f[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    

    
    
    
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

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_g_00 = derivativesx(d_g_0[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_01 = derivativesx(d_g_0[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_02 = derivativesx(d_g_0[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_03 = derivativesx(d_g_0[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_04 = derivativesx(d_g_0[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_05 = derivativesx(d_g_0[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_06 = derivativesx(d_g_0[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    
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


    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_g_10 = derivativesx(d_g_1[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_11 = derivativesx(d_g_1[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_12 = derivativesx(d_g_1[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_13 = derivativesx(d_g_1[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_14 = derivativesx(d_g_1[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_15 = derivativesx(d_g_1[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_16 = derivativesx(d_g_1[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    
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

    auto d_f = derivativesx(f(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));

    
    auto dd_f0 = derivativesx(d_f[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f1 = derivativesx(d_f[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f2 = derivativesx(d_f[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f3 = derivativesx(d_f[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f4 = derivativesx(d_f[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f5 = derivativesx(d_f[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_f6 = derivativesx(d_f[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    

    
    
    
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

    
    auto d_g_0 = derivativesx(g_0(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_g_00 = derivativesx(d_g_0[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_01 = derivativesx(d_g_0[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_02 = derivativesx(d_g_0[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_03 = derivativesx(d_g_0[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_04 = derivativesx(d_g_0[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_05 = derivativesx(d_g_0[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_06 = derivativesx(d_g_0[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    
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


    
    auto d_g_1 = derivativesx(g_1(p), wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    auto dd_g_10 = derivativesx(d_g_1[0], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_11 = derivativesx(d_g_1[1], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_12 = derivativesx(d_g_1[2], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_13 = derivativesx(d_g_1[3], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_14 = derivativesx(d_g_1[4], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_15 = derivativesx(d_g_1[5], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    auto dd_g_16 = derivativesx(d_g_1[6], wrt(p[0], p[1], p[2], p[3], p[4], p[5], p[6]));
    
    
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