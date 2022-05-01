// In this file I will try to use the CasADi created file to evaluate functions 


#include "hs071_nlp.hpp"

#include <cassert>
#include <iostream>

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

//------------------------------------------------------------------------------



using namespace Ipopt;

// constructor
HS071_NLP::HS071_NLP()
{}

//destructor
HS071_NLP::~HS071_NLP()
{}

void print_array(double* a, int n) {
    std::cout << "Array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void print_array2(std::vector <long long int> const &a) {
    std::cout << "The vector elements are : ";

    for(int i=0; i < a.size(); i++)
        std::cout << a.at(i) << ' ';
    std::cout << std::endl;
}



// Routine per il merge del pattern della matrice hessiana dei constrains
std::vector<long long int> pattern_merge_constrains(std::vector<long long int> a) {
    int n_row = (int) a[0];
    int n_col = (int) a[1];
    int n_constrains = n_row/n_col;

    std::vector<long long int> final_pattern;
    final_pattern.push_back(n_col);
    final_pattern.push_back(n_col);
    std::vector<int> col_ind_vect(n_col+1, 0);
    final_pattern.insert(final_pattern.end(), col_ind_vect.begin(), col_ind_vect.end()); // Concateno i due vettori contenti le dimensione e il numero di elementi non zero per ogni riga

    int index1 = 0;
    for(int i = 0; i <n_col; i++) {
        int nze1 = (int) (a[3+i] - a[2+i]);
        std::vector<long long int> patt1(a.begin() + (2+n_col+1) + index1,a.begin() + (2+n_col+1) + index1+nze1);
        index1 += nze1;

        for(std::vector<long long int>::iterator j = patt1.begin(); j != patt1.end(); ++j) {
            while(*j >= n_col) *j -= (long long int) n_col;
        }

        sort(patt1.begin(), patt1.end());
        patt1.erase( unique(patt1.begin(),patt1.end()),patt1.end() ); // Faccio il sort ed elimino i duplicati

        final_pattern[3+i] = (long long int) (final_pattern[2+i] + patt1.size());
        final_pattern.insert(final_pattern.end(), patt1.begin(), patt1.end()); // Concateno gli indici di riga con il pattern
    }
    return final_pattern;
}



// Routine per il merge di due pattern
// Routine per il merge di due pattern non triangolari per ottenerrne uno triangolare
std::vector<long long int> pattern_merge(std::vector<long long int> a, std::vector<long long int> b) {
    int n_row = (int) a[0];
    int n_col = (int) a[1];

    std::vector<long long int> final_pattern;
    final_pattern.push_back(n_row);
    final_pattern.push_back(n_col);
    std::vector<int> col_ind_vect(n_col+1, 0);
    final_pattern.insert(final_pattern.end(), col_ind_vect.begin(), col_ind_vect.end()); // Concateno i due vettori

    int index1 = 0;
    int index2 = 0;
    for(int i = 0; i <n_col; i++) {
        int nze1 = (int) (a[3+i] - a[2+i]);
        std::vector<long long int> patt1(a.begin() + (2+n_col+1) + index1,a.begin() + (2+n_col+1) + index1+nze1);
        index1 += nze1;

        int nze2 = (int) (b[3+i] - b[2+i]);
        std::vector<long long int> patt2(b.begin() + (2+n_col+1) + index2,b.begin() + (2+n_col+1) + index2+nze2);
        index2 += nze2;

        patt1.insert(patt1.end(), patt2.begin(), patt2.end());

        sort(patt1.begin(), patt1.end());
        patt1.erase( unique(patt1.begin(),patt1.end()),patt1.end() ); // Faccio il sort ed elimino i duplicati

        // Loop to select only the lower part of the matrix
        patt1.erase(std::remove_if(
                patt1.begin(), patt1.end(),
                [&i](const int& x) {
                    return x < i; // put your condition here
                }), patt1.end());

        final_pattern[3+i] = (long long int) (final_pattern[2+i] + patt1.size());
        final_pattern.insert(final_pattern.end(), patt1.begin(), patt1.end()); // Concateno gli indici di riga con il pattern
    }
    return final_pattern;
}

// Function to insert values from a pattern to a bigger pattern
void pattern_value_match_hessian(std::vector<long long int> short_patt, double* short_value, std::vector<long long int> long_patt, double* long_value, int n_cons, const Number* multiplier) {
    int q;
    for(q = 0; q < 2; q++) { // Loop over different constrains
        int n_row = (int) short_patt[0];
        int n_col = (int) short_patt[1];
        int index_short = 0;
        int index_long = 0;
        for(int i = 0; i < n_col; i++) {
            int nze_short = (int) (short_patt[3+i] - short_patt[2+i]);   // Non-zero elements for the i_column short pattern
            int nze_long = (int) (long_patt[3+i] - long_patt[2+i]);   // Non-zero elements for the i_column long pattern
            for(int j = 0; j < nze_short; j++) {
                int row_index = (int) (short_patt[2+(n_col+1)+index_short+j]);
                if(i <= (row_index-n_col*q)  &&  row_index >= (n_col*q)  &&  row_index < (n_col*(q+1))) {
                    int position = 0;
                    std::vector<long long int>::iterator itr = std::find(long_patt.begin()+(2+n_col+1+index_long), long_patt.begin()+(2+n_col+1+index_long+nze_long), row_index-(n_col*q));
                    for(int k = 1; k < ((row_index + 1) - (n_col * q)); k++) { // In questo modo ottengo una posizione relativa al constrain in analisi
                        position += k;
                    }
                    position += i;
                    long_value[position] += multiplier[q] * short_value[index_short+j];
                }
            }
            index_short += nze_short;
            index_long += nze_long;
        }
    }
}

void constrain_jac_structure(Index* a,Index* b, std::vector<long long int> sparsity_pattern) {
    int n_row = (int) sparsity_pattern[0];
    int n_col = (int) sparsity_pattern[1];

    int nz_elements = 0; // Numero elementi non zero trovati finora
    int index = 0; // Number of nz elements that is updated after every column iteration

    for(int k = 0; k < n_row; k++) { // Loop per ogni riga dello jacobiano
        index = 0;
        for(int i = 0; i < n_col; i++) {
            int nze_col = (int) (sparsity_pattern[3+i] - sparsity_pattern[2+i]);   // Non zero elements for the i_column
            for(int j = 0; j < nze_col; j++) { // Loop per leggere l'indice di riga degli elementi non zero della colonna i
                auto row_index = (long long int) sparsity_pattern[2+(n_col+1)+index+j];
                if(row_index == k) {
                    a[nz_elements] = (int) k;
                    b[nz_elements] = (int) i;
                    nz_elements++;
                }
            }
            index += nze_col;
        }
    }
}

// Insert values inside the final value vector from the pattern
// Used inside function like starting point..., where no previous structure is required
void pattern_value_match(double* a, const long long int* b, double* full) {
    int n_row = (int) b[0];
    int n_col = (int) b[1];
    int index = 0;
    int x_index = 0;
    for(int i = 0; i < n_col; i++) {
        int nze = (int) (b[3+i] - b[2+i]);   // Non zero elements for the i_column
        for(int j = 0; j < nze; j++) {
            int row_index = (int) b[2+(n_col+1)+index+j];
            full[n_col * row_index + i] = a[x_index];
            x_index++;
        }
        index += nze;
    }
}





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
  pattern_value_match(x_l_vector, sparsity_pattern_x_l, x_l_full);

  const long long int* sparsity_pattern_x_u = x_u_casadi_sparsity_out(0);
  auto* x_u_full = new double[n]();
  pattern_value_match(x_u_vector, sparsity_pattern_x_u, x_u_full);

  const long long int* sparsity_pattern_g_l = g_l_casadi_sparsity_out(0);
  auto* g_l_full = new double[m]();
  pattern_value_match(g_l_vector, sparsity_pattern_g_l, g_l_full);

  const long long int* sparsity_pattern_g_u = g_u_casadi_sparsity_out(0);
  auto* g_u_full = new double[m]();
  pattern_value_match(g_u_vector, sparsity_pattern_g_u, g_u_full);

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
  pattern_value_match(x_start_point_vector, sparsity_pattern_x_start_point, x_start_point_full);

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

  pattern_value_match(grad_f1, sparsity_pattern, grad_f);

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

  pattern_value_match(g1, sparsity_pattern, g);

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
      constrain_jac_structure(iRow,jCol,sparsity_pattern1);
  }
  else {
    // return the values of the jacobian of the constraints

    const long long int* sparsity_pattern = con_g_jac_sparsity_out(0);

    auto* values1 = new double[n*m]();

    con_g_jac(&x, &values1, iw, w, mem);
    pattern_value_match(values1, sparsity_pattern, values);
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
    const long long int* obj_f_hes_sparsity = obj_f_hes_sparsity_out(0);
    std::vector<long long int> obj_f_hes_sparsity_vect(obj_f_hes_sparsity, obj_f_hes_sparsity + (2+n+1+n*n));

    const long long int* con_g_hes_sparsity = con_g_hes_sparsity_out(0);
    std::vector<long long int> con_g_hes_sparsity_vect(con_g_hes_sparsity, con_g_hes_sparsity + (2+n+1+n*n*m));

    std::vector<long long int> merge_hes_con = pattern_merge_constrains(con_g_hes_sparsity_vect);
    std::vector<long long int> final_pattern = pattern_merge(obj_f_hes_sparsity_vect,merge_hes_con);
    constrain_jac_structure(iRow,jCol,final_pattern);
    Index idx = final_pattern[2+n];

    assert(idx == nele_hess); // nele_hess == 10
  }
  else {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only

    auto* values1 = new double[n*n]();
    auto* values2 = new double[n*n]();

    obj_f_hes(&x, &values1, iw, w, mem);
    con_g_hes(&x, &values2, iw, w, mem);

    const long long int* sparsity_pattern_f = obj_f_hes_sparsity_out(0);
    std::vector<long long int> sparsity_pattern_f_hes(sparsity_pattern_f, sparsity_pattern_f + (2+n+1+n*n));
    const long long int* sparsity_pattern_g = con_g_hes_sparsity_out(0);
    std::vector<long long int> sparsity_pattern_g_hes(sparsity_pattern_g, sparsity_pattern_g + (2+n+1+n*n*m));
    std::vector<long long int> sparsity_pattern_g_hes2 = pattern_merge_constrains(sparsity_pattern_g_hes); // Restituisce il patter compresso
    int n_constrain = (int) (sparsity_pattern_g_hes2[0]/sparsity_pattern_g_hes2[1]);
    std::vector<long long int> final_pattern = pattern_merge(sparsity_pattern_f_hes,sparsity_pattern_g_hes2);
    long long int nze_final = final_pattern[6]; // Number of non-zero elements in the final pattern
    auto* final_values = new double[nze_final]();
    pattern_value_match_hessian(sparsity_pattern_f_hes,values1,final_pattern,final_values,1,&obj_factor);
    pattern_value_match_hessian(sparsity_pattern_g_hes,values2,final_pattern,final_values,n_constrain,lambda);


    int final_index = 0;
    for (int i = 0; i < (n*(n+1)/2); i++) {
      if (final_values[i] != 0) {
        values[final_index] = final_values[i];
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
