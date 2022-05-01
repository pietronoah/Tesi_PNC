// In this file I will try to use the CasADi created file to evaluate functions 

#include "test_2_nlp.hpp"

#include <cassert>
#include <iostream>

#include "casadi_functions.c"

// Set delle variabile necessarie ad ipopt nelle chiamate alle funzioni generate da casadi
//------------------------------------------------------------------------------

    // Ora setto la variabile iw
    long long int* iw = (long long int*) malloc(sizeof(long long int));

    // Ora setto la variabile w
    double* w = (double*) malloc(sizeof(double));

    // Ora setto la variabile mem
    int mem = 10;

//------------------------------------------------------------------------------



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
  n = 4;

  // one equality constraint and one inequality constraint
  m = 2;

  // in this example the jacobian is dense and contains 8 nonzeros
  nnz_jac_g = 8;

  // the hessian is also dense and has 16 total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = 10;

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
  assert(n == 4);
  assert(m == 2);


  
    x_l[0] = 1;
    x_u[0] = 5;
  
    x_l[1] = 1;
    x_u[1] = 5;
  
    x_l[2] = 1;
    x_u[2] = 5;
  
    x_l[3] = 1;
    x_u[3] = 5;
  

  // the variables have upper bounds of 5
  
    g_l[0] = 25;
    g_u[0] = 2e+19;
  
    g_l[1] = 40;
    g_u[1] = 40;
  

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
  
    x[1] = 5;
  
    x[2] = 5;
  
    x[3] = 1;
  

  return true;
}

// returns the value of the objective function
bool HS071_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  assert(n == 4);

  double* obj_value1 = &obj_value;

  obj_f(&x, &obj_value1, iw, w, mem);


  return true;
}

// return the gradient of the objective function grad_{x} f(x)
bool HS071_NLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
  assert(n == 4);

  // Gradiente restituito come vettore colonna

  const long long int* sparsity_pattern = obj_f_grad_sparsity_out(0);
  int n_row = sparsity_pattern[0];
  int n_col = sparsity_pattern[1];

  double* grad_f1 = (double*) calloc(n_row*n_col,sizeof(double));

  obj_f_grad(&x, &grad_f1, iw, w, mem);

  int index = 0;
  int grad_index = 0;
  for(int i = 0; i < n_col; i++) {
    int nze = sparsity_pattern[3+i] - sparsity_pattern[2+i];   // Non zero elements for the i_column
    for(int j = 0; j < nze; j++) {
      grad_f[n_col * sparsity_pattern[2+(n_col+1)+index+j] + i] = grad_f1[grad_index];
      grad_index++;
    }
    index += nze;
  }


  return true;
}

// return the value of the constraints: g(x)
bool HS071_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  assert(n == 4);
  assert(m == 2);


  const long long int* sparsity_pattern = con_g_sparsity_out(0);
  int n_row = sparsity_pattern[0];
  int n_col = sparsity_pattern[1];

  double* g1 = (double*) calloc(n_row*n_col,sizeof(double));

  con_g(&x, &g1, iw, w, mem);

  int index = 0;
  int g_index = 0;
  for(int i = 0; i < n_col; i++) {
    int nze = sparsity_pattern[3+i] - sparsity_pattern[2+i];   // Non zero elements for the i_column
    for(int j = 0; j < nze; j++) {
      g[n_col * sparsity_pattern[2+(n_col+1)+index+j] + i] = g1[g_index];
      g_index++;
    }
    index += nze;
  }


  return true;
}

// return the structure or values of the jacobian
bool HS071_NLP::eval_jac_g(Index n, const Number* x, bool new_x,
                           Index m, Index nele_jac, Index* iRow, Index *jCol,
                           Number* values)
{
  if (values == NULL) {
    // return the structure of the jacobian

    const long long int* sparsity_pattern = con_g_jac_sparsity_out(0);
    int n_row = sparsity_pattern[0];
    int n_col = sparsity_pattern[1];

    int nz_elements = 0; // Numero elementi non zero trovati finora

    int index = 0; // Number of nz elements that is updated after every column iteration

    for(int k = 0; k < n_row; k++) { // Loop per ogni riga dello jacobiano
      index = 0;
      for(int i = 0; i < n_col; i++) {
        int nze_col = sparsity_pattern[3+i] - sparsity_pattern[2+i];   // Non zero elements for the i_column
        for(int j = 0; j < nze_col; j++) { // Loop per leggere l'indice di riga degli elementi non zero della colonna i
          long long int row_index = sparsity_pattern[2+(n_col+1)+index+j];
          long long int k_1 = k;
          if(row_index == k_1) {  // Qui da qualche problema di segmentation fault
            iRow[nz_elements] = (double) k;
            jCol[nz_elements] = (double) i;
            nz_elements++;
          }
        }
        index += nze_col;
      }
    }  
  }
  else {
    // return the values of the jacobian of the constraints

    //con_g_jac(&x, &values, iw, w, mem);

    const long long int* sparsity_pattern = con_g_jac_sparsity_out(0);
    int n_row = sparsity_pattern[0];
    int n_col = sparsity_pattern[1];


    double* values1 = (double*) calloc(n_row*n_col,sizeof(double));
    double* values3 = (double*) calloc(n_row*n_col,sizeof(double));

    con_g_jac(&x, &values1, iw, w, mem);

    int index = 0;
    int values_index = 0;
    for(int i = 0; i < n_col; i++) {
      int nze = sparsity_pattern[3+i] - sparsity_pattern[2+i];   // Non zero elements for the i_column
      for(int j = 0; j < nze; j++) {
        values3[n_col * sparsity_pattern[2+(n_col+1)+index+j] + i] = values1[values_index];
        values_index++;
      }
      index += nze;
    }


    int final_index = 0;
    for (int i = 0; i < (n_row*n_col); i++) {
      if (values3[i] != 0) {
        values[final_index] = values3[i];
        final_index++;
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
   


    double* x_path = (double *) calloc(n,sizeof(double));
    for (int i = 0; i < n; i++) {
      x_path[i] = rand();
    }

    const double* x_path1 = x_path;



    const long long int* obj_f_hes_sparsity = obj_f_hes_sparsity_out(0);
    int n_row = obj_f_hes_sparsity[0];
    int n_col = obj_f_hes_sparsity[1];

    double* values1 = (double*) calloc(n_row*n_col,sizeof(double));
    double* values2 = (double*) calloc(n_row*n_col,sizeof(double));
    double* values3 = (double*) calloc(n_row*n_col,sizeof(double));

    int position = 0;

    obj_f_hes(&x_path1, &values1, iw, w, mem);

    int index = 0;
    int values_index = 0;
    for(int i = 0; i < n_col; i++) {
      int nze = obj_f_hes_sparsity[3+i] - obj_f_hes_sparsity[2+i];   // Non zero elements for the i_column
      for(int j = 0; j < nze; j++) {
        
        Index line_index = obj_f_hes_sparsity[2+(n_col+1)+index+j];
        if(i <= line_index) { // Only if the element is in the lower part of the matrix (column_index lower than row_index)

          position = 0;
          for(int k = 1; k < (line_index + 1); k++) {
            position += k;
          }
          position += i;

          values3[position] = values1[values_index]; // In questo caso macherebbe obj_factor
        }
        values_index++;
      }
      index += nze;
    }



    const long long int* con_g_hes_sparsity = con_g_hes_sparsity_out(0);
    n_row = con_g_hes_sparsity[0];
    n_col = con_g_hes_sparsity[1];

    int n_constrain = n_row/n_col;

    con_g_hes(&x_path1, &values2, iw, w, mem);

    for(int w = 0; w < (n_constrain); w++) { // Itero sui vari constrains applicati alla funzione

      int index = 0;
      int values_index = 0;
      for(int i = 0; i < n_col; i++) {
        int nze = con_g_hes_sparsity[3+i] - con_g_hes_sparsity[2+i];   // Non zero elements for the i_column
        for(int j = 0; j < nze; j++) {

          Index line_index = con_g_hes_sparsity[2+(n_col+1)+index+j];
          if(i <= (line_index - n_col*w) && line_index >= n_col*w && line_index < n_col*(w+1)) { // Only if the element is in the lower part of the matrix (column_index lower than row_index)

            int position = 0;
            for(int k = 1; k < ((line_index + 1) - (n_col * w)); k++) { // In questo modo ottengo una posizione relativa al constrain in analisi
              position += k;
            }
            position += i;

            values3[position] += values2[values_index];
          }
          values_index++;
        }
        index += nze;
      }

    }

    

    Index idx = 0;
    for (Index i = 0; i < n_row*n_col; i++) {
      if (values3[i] != 0) {
        Index line_index = 0;
        Index k = i;
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


    const long long int* sparsity_pattern = obj_f_hes_sparsity_out(0);
    int n_row = sparsity_pattern[0];
    int n_col = sparsity_pattern[1];

    double* values1 = (double*) calloc(n_row*n_col,sizeof(double));
    double* values3 = (double*) calloc(n_row*n_col,sizeof(double));

    int position = 0;

    obj_f_hes(&x, &values1, iw, w, mem);

    int index = 0;
    int values_index = 0;
    for(int i = 0; i < n_col; i++) {
      int nze = sparsity_pattern[3+i] - sparsity_pattern[2+i];   // Non zero elements for the i_column
      for(int j = 0; j < nze; j++) {

        if(i <= sparsity_pattern[2+(n_col+1)+index+j]) { // Only if the element is in the lower part of the matrix (column_index lower than row_index)

          position = 0;
          for(int k = 1; k < (sparsity_pattern[2+(n_col+1)+index+j] + 1); k++) {
            position += k;
          }
          position += i;

          values3[position] = obj_factor * values1[values_index]; 
        }
        values_index++;
      }
      index += nze;
    }



    sparsity_pattern = con_g_hes_sparsity_out(0);
    n_row = sparsity_pattern[0];
    n_col = sparsity_pattern[1];

    double* values2 = (double*) calloc(n_row*n_col,sizeof(double));


    int n_constrain = n_row/n_col;

    con_g_hes(&x, &values1, iw, w, mem);

    for(int w = 0; w < (n_constrain); w++) { // Itero sui vari constrains applicati alla funzione

      int index = 0;
      int values_index = 0;
      for(int i = 0; i < n_col; i++) {
        int nze = sparsity_pattern[3+i] - sparsity_pattern[2+i];   // Non zero elements for the i_column
        for(int j = 0; j < nze; j++) {

          if(i <= (sparsity_pattern[2+(n_col+1)+index+j] - n_col*w) && sparsity_pattern[2+(n_col+1)+index+j] >= n_col*w && sparsity_pattern[2+(n_col+1)+index+j] < n_col*(w+1)) { // Only if the element is in the lower part of the matrix (column_index lower than row_index)

            int position = 0;
            for(int k = 1; k < ((sparsity_pattern[2+(n_col+1)+index+j] + 1) - (n_col * w)); k++) { // In questo modo ottengo una posizione relativa al constrain in analisi
              position += k;
            }
            position += i;

            values3[position] += lambda[w] * values1[values_index];
          }
          values_index++;
        }
        index += nze;
      }

    }

    int final_index = 0;
    for (int i = 0; i < (n*(n+1)/2); i++) {
      if (values3[i] != 0) {
        values[final_index] = values3[i];
        final_index++;
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