// In questo file è contenuta l'interfaccia principale di ipopt


#include "hs071_nlp.hpp"

#include <cassert>
#include <iostream>

#include "pattern_functions.h"
#include "CasADi_export/source.c"

// Set delle variabile necessarie ad ipopt nelle chiamate alle funzioni generate da casadi
//------------------------------------------------------------------------------

    // Ora setto la variabile iw
    long long int* iw = new long long int();

    // Ora setto la variabile w
    double* w = new double ();

    // Ora setto la variabile mem
    int mem = 10;

    // Ora setto le variabili del problema
    double n_variables_casadi;
    double* n_variables_pointer = &n_variables_casadi;

    double n_constrains_casadi;
    double* n_constrains_pointer = &n_constrains_casadi;

    double con_g_jac_nze_casadi;
    double* con_g_jac_nze_pointer = &con_g_jac_nze_casadi;

    double obj_f_hes_nze_casadi;
    double* obj_f_hes_nze_pointer = &obj_f_hes_nze_casadi;

    // Set vettore contenente il pattern finale della matrice hessiana complessiva
    // Lo imposto come variabile globale per non dover eseguire lo stesso processo di merge ad ogni iterazione
    // std::vector<long long int> final_pattern_hess;

//------------------------------------------------------------------------------



using namespace Ipopt;

// constructor
HS071_NLP::HS071_NLP()
{}

//destructor
HS071_NLP::~HS071_NLP()
{}


MyClass MyObj;


// returns the size of the problem
bool HS071_NLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, IndexStyleEnum& index_style)
{

  n_variables(NULL, &n_variables_pointer, iw, w, mem);
  n_constrains(NULL, &n_constrains_pointer, iw, w, mem);
  con_g_jac_nze(NULL, &con_g_jac_nze_pointer, iw, w, mem);
  obj_f_hes_nze(NULL, &obj_f_hes_nze_pointer, iw, w, mem);
  
  // The problem described in HS071_NLP.hpp has 4 variables, x[0] through x[3]
  n = (Index) n_variables_casadi;

  // one equality constraint and one inequality constraint
  m = (Index) n_constrains_casadi;

  // in this example the jacobian is dense and contains 8 nonzeros
  nnz_jac_g = (Index) con_g_jac_nze_casadi;

  // the hessian is also dense and has 16 total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = (Index) obj_f_hes_nze_casadi;

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
  assert(n == n_variables_casadi);
  assert(m == n_constrains_casadi);

  auto* x_l_vector = new double[n]();
  auto* x_u_vector = new double[n]();
  auto* g_l_vector = new double[m]();
  auto* g_u_vector = new double[m]();

  x_l_casadi(NULL, &x_l_vector, iw, w, mem);
  x_u_casadi(NULL, &x_u_vector, iw, w, mem);
  g_l_casadi(NULL, &g_l_vector, iw, w, mem);
  g_u_casadi(NULL, &g_u_vector, iw, w, mem);

  const long long int* sparsity_pattern_x_l = x_l_casadi_sparsity_out(0);
  auto* x_l_full = new double[n]();
  MyObj.pattern_value_match(x_l_vector, sparsity_pattern_x_l, x_l_full);

  const long long int* sparsity_pattern_x_u = x_u_casadi_sparsity_out(0);
  auto* x_u_full = new double[n]();
  MyObj.pattern_value_match(x_u_vector, sparsity_pattern_x_u, x_u_full);

  const long long int* sparsity_pattern_g_l = g_l_casadi_sparsity_out(0);
  auto* g_l_full = new double[m]();
  MyObj.pattern_value_match(g_l_vector, sparsity_pattern_g_l, g_l_full);

  const long long int* sparsity_pattern_g_u = g_u_casadi_sparsity_out(0);
  auto* g_u_full = new double[m]();
  MyObj.pattern_value_match(g_u_vector, sparsity_pattern_g_u, g_u_full);

  // the variables have lower bounds of 1
  for (Index i=0; i<n; i++) {
    x_l[i] = x_l_full[i];
    x_u[i] = x_u_full[i];
    std::cout << "Bounds for variable n: " << i << " are: " << x_l[i] << " , " << x_u[i] << std::endl;
  }

  // the variables have upper bounds of 5
  for (Index i=0; i<m; i++) {
    g_l[i] = g_l_full[i];
    g_u[i] = g_u_full[i];
    std::cout << "Bounds for constrain n: " << i << " are: " << g_l[i] << " , " << g_u[i] << std::endl;
  }

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

  // Get x starting point from casadi function
  auto* x_start_point_vector = new double[n]();
  x_start_point(NULL, &x_start_point_vector, iw, w, mem);

  const long long int* sparsity_pattern_x_start_point = x_start_point_sparsity_out(0);
  auto* x_start_point_full = new double[m]();
  MyObj.pattern_value_match(x_start_point_vector, sparsity_pattern_x_start_point, x_start_point_full);

  for (Index i=0; i<n; i++) {
    x[i] = x_start_point_full[i];
    std::cout << "Starting point for variable n: " << i << " is: " << x[i] << std::endl;
  }

  return true;
}

// returns the value of the objective function
bool HS071_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
  assert(n == n_variables_casadi);

  double* obj_value1 = &obj_value;

  obj_f(&x, &obj_value1, iw, w, mem);


  return true;
}

// return the gradient of the objective function grad_{x} f(x)
bool HS071_NLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
  assert(n == n_variables_casadi);

  const long long int* sparsity_pattern = obj_f_grad_sparsity_out(0);
  int n_row = (int) sparsity_pattern[0];
  int n_col = (int) sparsity_pattern[1];

  auto* grad_f1 = new double[n_row*n_col]();

  obj_f_grad(&x, &grad_f1, iw, w, mem);

  MyObj.pattern_value_match(grad_f1, sparsity_pattern, grad_f);

  return true;
}

// return the value of the constraints: g(x)
bool HS071_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  assert(n == n_variables_casadi);
  assert(m == n_constrains_casadi);


  for(int i = 0; i < m; i++) {
    g[i] = 0;
  }

  const long long int* sparsity_pattern = con_g_sparsity_out(0);
  int n_row = (int) sparsity_pattern[0];
  int n_col = (int) sparsity_pattern[1];

  auto* g1 = new double[n_row*n_col]();

  con_g(&x, &g1, iw, w, mem);

  MyObj.pattern_value_match(g1, sparsity_pattern, g);

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
      std::vector<long long int> sparsity_pattern1(sparsity_pattern, sparsity_pattern + (2 + sparsity_pattern[1] + 1 + nele_jac));
      MyObj.constrain_jac_structure(iRow,jCol,sparsity_pattern1);

  }
  else {
    // return the values of the jacobian of the constraints
    // Qui devo utilizzare una funzione che tenga conto della struttura dello jacobiano, restituendo un vettore di elementi compressi, ovvere semza gli elementi nulli del pattern

    const long long int* sparsity_pattern = con_g_jac_sparsity_out(0);
    auto* values1 = new double[nele_jac]();
    con_g_jac(&x, &values1, iw, w, mem);

    std::vector<long long int> sparsity_pattern_g_jac(sparsity_pattern, sparsity_pattern + (2 + sparsity_pattern[1] + 1 + nele_jac));

    MyObj.pattern_value_match_constrains_jacobian(sparsity_pattern_g_jac, values1, values, m);
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

    // Eseguo il merge dei due pattern, in modo da ottenere il pattern complessivo

    // Ottengo i pattern della matrice hessiano della funzione f e della matrice hessiana dei constrains
    const long long int* obj_f_hes_sparsity = obj_f_hes_sparsity_out(0);
    std::vector<long long int> obj_f_hes_sparsity_vect(obj_f_hes_sparsity, obj_f_hes_sparsity + (2+n+1+n*n));
    const long long int* con_g_hes_sparsity = con_g_hes_sparsity_out(0);
    std::vector<long long int> con_g_hes_sparsity_vect(con_g_hes_sparsity, con_g_hes_sparsity + (2+n+1+n*n*m));

    // Compatto il pattern della matrice dei constrains
    std::vector<long long int> merge_hes_con = MyObj.pattern_merge_constrains(con_g_hes_sparsity_vect);
      //std::cout << "Constrain pattern is: ";
      //print_array2(merge_hes_con);

    // Merge dei pattern delle matrici hessiane già compattate
    MyObj.final_pattern_hess = MyObj.pattern_merge(obj_f_hes_sparsity_vect,merge_hes_con);
      //std::cout << "Final pattern is: ";
      //print_array2(final_pattern_hess);

    // Nuova funzione per immagazzinare l'hessiano secondo la regola CCS
    // Risulta molto comodo per eseguire il match tra pattern e values durante la fase di valutazione
    MyObj.final_hess_structure(iRow,jCol,MyObj.final_pattern_hess);

    // Creo la mappatura del vettore delle values dei constrains
    MyObj.constrain_hess_map(con_g_hes_sparsity_vect, MyObj.final_pattern_hess, MyObj.cons_hess_map, MyObj.cons_vect);
    MyObj.print_array2(MyObj.cons_hess_map);

    Index idx = MyObj.final_pattern_hess[2+n]; // RIGUARDARE QUI!!

    //std::cout << "Idx is: " << idx << std::endl;

    assert(idx == nele_hess); // nele_hess == 10
  }
  else {
    // return the values. This is a symmetric matrix, fill the lower left triangle only

    auto* values1 = new double[n*n]();
    auto* values2 = new double[n*n]();

    obj_f_hes(&x, &values1, iw, w, mem);
    con_g_hes(&x, &values2, iw, w, mem);

    // Ottengo i pattern della matrice hessiano della funzione f e della matrice hessiana dei constrains
    const long long int* sparsity_pattern_f = obj_f_hes_sparsity_out(0);
    std::vector<long long int> sparsity_pattern_f_hes(sparsity_pattern_f, sparsity_pattern_f + (2+n+1+n*n));
    const long long int* sparsity_pattern_g = con_g_hes_sparsity_out(0);
    std::vector<long long int> sparsity_pattern_g_hes(sparsity_pattern_g, sparsity_pattern_g + (2+n+1+n*n*m));
    int n_constrain = (int) (sparsity_pattern_g_hes[0]/sparsity_pattern_g_hes[1]);

      /*
      // Compatto il pattern della matrice dei constrains
      std::vector<long long int> sparsity_pattern_g_hes2 = pattern_merge_constrains(sparsity_pattern_g_hes); // Restituisce il patter compresso

      // Merge dei pattern delle matrici hessiane già compattate
      std::vector<long long int> final_pattern = pattern_merge(sparsity_pattern_f_hes,sparsity_pattern_g_hes2);
      */


    long long int nze_final = MyObj.final_pattern_hess[2+n]; // Number of non-zero elements in the final pattern
    auto* final_values = new double[nze_final]();

    // Inserisco i valori delle singole matrici all'interno del vettore di valori finale
    MyObj.pattern_value_match_hessian(sparsity_pattern_f_hes,values1,MyObj.final_pattern_hess,final_values,1,&obj_factor);
    //MyObj.pattern_value_match_hessian(sparsity_pattern_g_hes,values2,MyObj.final_pattern_hess,final_values,n_constrain,lambda);

    MyObj.pattern_value_match_hessian_map(MyObj.cons_hess_map, MyObj.cons_vect, values2, final_values, lambda);

    for (int i = 0; i < nele_hess; i++) {
        values[i] = final_values[i];
    }

    //std::cout << "obj_factor: " << obj_factor << ", lambda: " << lambda[0] << ", " << lambda[1] << std::endl;
    //print_array_const(x,4);
    //print_array(values,10);
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
