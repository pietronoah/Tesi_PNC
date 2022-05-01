//
// Created by Pietro Noah Crestaz on 01/05/22.
//

#ifndef FIRST_TEST_WITH_CASADI_IMPLEMENTATION_PATTERN_FUNCTIONS_H
#define FIRST_TEST_WITH_CASADI_IMPLEMENTATION_PATTERN_FUNCTIONS_H



#include <vector>
#include "hs071_nlp.hpp"
#include <iostream>

class MyClass {

public:

    std::vector<long long int> final_pattern_hess;
    std::vector<long long int> cons_hess_map;
    std::vector<long long int> cons_vect;

    void print_array(double *a, int n);

    void print_array_const(const double *a, int n);

    void print_array_ind(Index *a, int n);

    void print_array_long(long long int *a, int n);

    void print_array2(std::vector<long long int> const &a);


// Inserisce i valori all'interno del vettore contente i valori finali a partire dal pattern
// Usato in funzioni come starting point, dove precedentemente non è richiesta la struttura
    void pattern_value_match(double *a, const long long int *b, double *full);


// Routine per ricavare la struttura della matrice jacobiana dei constrains
    void constrain_jac_structure(Index *a, Index *b, std::vector<long long int> sparsity_pattern);


// Function to insert values from a small pattern to a bigger pattern
    void
    pattern_value_match_constrains_jacobian(std::vector<long long int> pattern, double *values, double *final_values, int n_cons);


// Compatta il pattern della matrice hessiana dei constrains
// Il pattern in uscita è il pattern della matrice quadrata complessiva (non triangolare)
    std::vector<long long int> pattern_merge_constrains(std::vector<long long int> a);


// Routine per il merge di due pattern non triangolari per ottenerne uno triangolare
// Utilizzata nel merge di due mattern nella funzione per ricavare l'hessiano complessivo
    std::vector<long long int> pattern_merge(std::vector<long long int> a, std::vector<long long int> b);


// FUnzione per riempire i vettori iRow e jCol secondo il metodo di storage CCS
    void final_hess_structure(Index *a, Index *b, std::vector<long long int> sparsity_pattern);

    void constrain_hess_map(std::vector<long long int> cons_patt, std::vector<long long int> final_patt, std::vector<long long int>& map_vector, std::vector<long long int>& cons_vector);


// Funzione per riempire il vettore di valori finali dell'hessiano
// Deve restituire solo i valori diversi da 0 (contenuti nel pattern) e appartenenti al traingolo basso della matrice
    void pattern_value_match_hessian(std::vector<long long int> short_patt, double *short_value,
                                     std::vector<long long int> final_patt, double *final_value, int n_cons,
                                     const Number *multiplier);

    void pattern_value_match_hessian_map(std::vector<long long int>& map_vector, std::vector<long long int>& cons_vector, double* cons_value, double* final_value, const Number* multiplier);

};




#endif //FIRST_TEST_WITH_CASADI_IMPLEMENTATION_PATTERN_FUNCTIONS_H
