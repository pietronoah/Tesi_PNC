// In questo file è contenuta l'interfaccia principale di ipopt


#include "test_7_nlp.hpp"

#include <cassert>
#include <iostream>

#include "casadi_functions.c"

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
    std::vector<long long int> final_pattern_hess;

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

void print_array_const(const double* a, int n) {
    std::cout << "Array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void print_array_ind(Index* a, int n) {
    std::cout << "Index array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void print_array_long(long long int* a, int n) {
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



// Inserisce i valori all'interno del vettore contente i valori finali a partire dal pattern
// Usato in funzioni come starting point, dove precedentemente non è richiesta la struttura
void pattern_value_match(double* a, const long long int* b, double* full) {
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


// Routine per ricavare la struttura della matrice jacobiana dei constrains
/*void constrain_jac_structure(Index* a,Index* b, std::vector<long long int> sparsity_pattern) {
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
}*/

void constrain_jac_structure(Index* a,Index* b, std::vector<long long int> sparsity_pattern) {
    int n_col = (int) sparsity_pattern[1];
    int index = 0;
    int position = 0;

    for(int i = 0; i < n_col; i++) {
        int nze = (int) (sparsity_pattern[3+i] - sparsity_pattern[2+i]);
        for(int j = 0; j < nze; j++) {
            int row_index = (int) (sparsity_pattern[2+(n_col+1)+index+j]);
            a[position] = row_index;
            b[position] = i;
            position++;
        }
        index += nze;
    }
}


// Function to insert values from a small pattern to a bigger pattern
/*void pattern_value_match_constrains_jacobian(std::vector<long long int> pattern, double* values, double* final_values, int n_cons) {
    int position = 0;
    for(int q = 0; q < n_cons; q++) { // Loop over different constrains
        int n_col = (int) pattern[1];
        int index = 0;
        for(int i = 0; i < n_col; i++) {
            int nze = (int) (pattern[3+i] - pattern[2+i]);   // Non-zero elements for the i_column pattern
            for(int j = 0; j < nze; j++) {
                int row_index = (int) (pattern[2+(n_col+1)+index+j]);
                if(row_index == q) {
                    final_values[position] = values[index + j];
                    position++;
                }
            }
            index += nze;
        }
    }
}*/


void pattern_value_match_constrains_jacobian(std::vector<long long int> pattern, double* values, double* final_values, int n_cons) {
    int position = 0;
    int n_col = (int) pattern[1];
    int index = 0;
    for(int i = 0; i < n_col; i++) {
        int nze = (int) (pattern[3+i] - pattern[2+i]);   // Non-zero elements for the i_column pattern
        for(int j = 0; j < nze; j++) {
            final_values[position] = values[index + j];
            position++;
        }
        index += nze;
    }
}






// Compatta il pattern della matrice hessiana dei constrains
// Il pattern in uscita è il pattern della matrice quadrata complessiva (non triangolare)
std::vector<long long int> pattern_merge_constrains(std::vector<long long int> a) {
    int n_row = (int) a[0];
    int n_col = (int) a[1];
    int n_constrains = n_row/n_col;

    // Creo un vettore pattern finale
    std::vector<long long int> final_pattern;
    final_pattern.push_back(n_col);
    final_pattern.push_back(n_col);

    // Vettore contenente gli elementi non zero per colonna
    std::vector<int> col_ind_vect(n_col+1, 0);
    final_pattern.insert(final_pattern.end(), col_ind_vect.begin(), col_ind_vect.end()); // Concateno i due vettori contenti le dimensione e il numero di elementi non zero per ogni riga

    int index1 = 0;
    for(int i = 0; i <n_col; i++) {
        int nze1 = (int) (a[3+i] - a[2+i]);
        std::vector<long long int> patt1(a.begin() + (2+n_col+1) + index1,a.begin() + (2+n_col+1) + index1+nze1);
        index1 += nze1;

        // Riduco gli indici di riga di un fattore n_col fino a farli rientrare nella matrice quadrata n * n
        // Da riguardare questo range-based for loop
        for(std::vector<long long int>::iterator j = patt1.begin(); j != patt1.end(); ++j) {
            while(*j >= n_col) *j -= (long long int) n_col;
        }

        // Faccio il sort ed elimino i duplicati
        sort(patt1.begin(), patt1.end());
        patt1.erase( unique(patt1.begin(),patt1.end()),patt1.end() );

        // Inserisco il pattern della colonna nel pattern finale
        final_pattern[3+i] = (long long int) (final_pattern[2+i] + patt1.size());
        final_pattern.insert(final_pattern.end(), patt1.begin(), patt1.end()); // Concateno gli indici di riga con il pattern
    }
    return final_pattern;
}



// Routine per il merge di due pattern non triangolari per ottenerne uno triangolare
// Utilizzata nel merge di due mattern nella funzione per ricavare l'hessiano complessivo
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

// FUnzione per riempire i vettori iRow e jCol secondo il metodo di storage CCS
void final_hess_structure(Index* a,Index* b, std::vector<long long int> sparsity_pattern) {
    int n_col = (int) sparsity_pattern[1];
    int index = 0;
    int position = 0;

    for(int i = 0; i < n_col; i++) {
        int nze = (int) (sparsity_pattern[3+i] - sparsity_pattern[2+i]);
        for(int j = 0; j < nze; j++) {
            int row_index = (int) (sparsity_pattern[2+(n_col+1)+index+j]);
            if(i <= row_index) {
                a[position] = row_index;
                b[position] = i;
                position++;
            }
        }
        index += nze;
    }
}


// Funzione per riempire il vettore di valori finali dell'hessiano
// Deve restituire solo i valori diversi da 0 (contenuti nel pattern) e appartenenti al traingolo basso della matrice
void pattern_value_match_hessian(std::vector<long long int> short_patt, double* short_value, std::vector<long long int> final_patt, double* final_value, int n_cons, const Number* multiplier) {
    for(int q = 0; q < n_cons; q++) { // Loop over different constrains
        int n_col = (int) short_patt[1];

        int position = 0;
        int index_short = 0;

        for(int i = 0; i < n_col; i++) {
            int nze_short = (int) (short_patt[3+i] - short_patt[2+i]);   // Non-zero elements for the i_column short pattern
            int nze_long = (int) (final_patt[3+i] - final_patt[2+i]);   // Non-zero elements for the i_column long pattern
            //std::cout << "Position: " << position << std::endl;
            for(int j = 0; j < nze_short; j++) {
                int row_index_short = (int) (short_patt[2+(n_col+1)+index_short+j]);
                for(int l = 0; l < nze_long; l++) {
                    if(i > (row_index_short-n_col*q)  ||  row_index_short < (n_col*q)  ||  row_index_short >= (n_col*(q+1))) {
                        break;
                    }
                    // Verifico che il valore in analisi sia nel range considerato (varia nel caso della matrice dei constrains in cui ho più righe che colonne)
                    // Verifico si trovi nella parte bassa triangolare della matrice
                    // Se il valore del pattern piccolo non corrisponde a quello del pattern finale, incremento la sua posizione e verifico la congruenza con l'elemento successivo
                    if(i <= (row_index_short-n_col*q)  &&  row_index_short >= (n_col*q)  &&  row_index_short < (n_col*(q+1))  &&  (row_index_short - n_col*q) == final_patt[2+n_col+1 + position]) {
                        final_value[position] += multiplier[q] *  short_value[index_short + j];
                        //std::cout << "Row index: " << row_index_short << ", Position: " << position << ", Short value: " << short_value[index_short + j] << ", multiplier: " << multiplier[q] << ", final value: " << final_value[position] << std::endl;
                        position++;
                        break;
                    }
                    //std::cout << "Row index: " << row_index_short << std::endl;
                    if(i <= (row_index_short-n_col*q)  &&  row_index_short >= (n_col*q)  &&  row_index_short < (n_col*(q+1))) {
                        position++;
                    }
                }
            }
            position = (int) final_patt[3+i];
            index_short += nze_short;
        }
    }
}




\
// returns the size of the problem
// returns the size of the problem
bool HS071_NLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, IndexStyleEnum& index_style)
{


    // The problem described in HS071_NLP.hpp has 4 variables, x[0] through x[3]
    n = 4;

    // one equality constraint and one inequality constraint
    m = 3;

    // in this example the jacobian is dense and contains 8 nonzeros
    nnz_jac_g = 10;

    // the hessian is also dense and has 16 total nonzeros, but we
    // only need the lower left corner (since it is symmetric)
    nnz_h_lag = 6;

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
    assert(m == 3);



    x_l[0] = -5;
    x_u[0] = 5;

    x_l[1] = -5;
    x_u[1] = 5;

    x_l[2] = -5;
    x_u[2] = 5;

    x_l[3] = -5;
    x_u[3] = 5;


    // the variables have upper bounds of 5

    g_l[0] = -2e+19;
    g_u[0] = 5;

    g_l[1] = -2e+19;
    g_u[1] = 4;

    g_l[2] = 1.5;
    g_u[2] = 2e+19;


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


    x[0] = 0.5;

    x[1] = 0.5;

    x[2] = 0.5;

    x[3] = 0.5;


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
    // Qui devo utilizzare una funzione che tenga conto della struttura dello jacobiano, restituendo un vettore di elementi compressi, ovvere semza gli elementi nulli del pattern

    const long long int* sparsity_pattern = con_g_jac_sparsity_out(0);
    auto* values1 = new double[nele_jac]();
    con_g_jac(&x, &values1, iw, w, mem);

    std::vector<long long int> sparsity_pattern_g_jac(sparsity_pattern, sparsity_pattern + (2 + sparsity_pattern[1] + 1 + nele_jac));

    pattern_value_match_constrains_jacobian(sparsity_pattern_g_jac, values1, values, m);
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
    std::vector<long long int> merge_hes_con = pattern_merge_constrains(con_g_hes_sparsity_vect);
      std::cout << "Constrain pattern is: ";
      print_array2(merge_hes_con);

    // Merge dei pattern delle matrici hessiane già compattate
    final_pattern_hess = pattern_merge(obj_f_hes_sparsity_vect,merge_hes_con);
      std::cout << "Final pattern is: ";
      print_array2(final_pattern_hess);

    // Nuova funzione per immagazzinare l'hessiano secondo la regola CCS
    // Risulta molto comodo per eseguire il match tra pattern e values durante la fase di valutazione
    final_hess_structure(iRow,jCol,final_pattern_hess);
    Index idx = final_pattern_hess[2+n]; // RIGUARDARE QUI!!

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


    long long int nze_final = final_pattern_hess[2+n]; // Number of non-zero elements in the final pattern
    auto* final_values = new double[nze_final]();

    // Inserisco i valori delle singole matrici all'interno del vettore di valori finale
    pattern_value_match_hessian(sparsity_pattern_f_hes,values1,final_pattern_hess,final_values,1,&obj_factor);
    pattern_value_match_hessian(sparsity_pattern_g_hes,values2,final_pattern_hess,final_values,n_constrain,lambda);

    for (int i = 0; i < nele_hess; i++) {
        values[i] = final_values[i];
    }

    std::cout << "obj_factor: " << obj_factor << ", lambda: " << lambda[0] << ", " << lambda[1] << std::endl;
    print_array_const(x,4);
    print_array(values,10);

    /*int final_index = 0;
    for (int i = 0; i < (n*(n+1)/2); i++) {
      if (final_values[i] != 0) {
        values[final_index] = final_values[i];
        final_index++;
      }
    }*/
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
