//
// Created by Pietro Noah Crestaz on 01/05/22.
//

#include <vector>
#include "hs071_nlp.hpp"
#include <iostream>
#include "pattern_functions.h"




void MyClass::print_array(double* a, int n) {
    std::cout << "Array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void MyClass::print_array_const(const double* a, int n) {
    std::cout << "Array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void MyClass::print_array_ind(Index* a, int n) {
    std::cout << "Index array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void MyClass::print_array_long(long long int* a, int n) {
    std::cout << "Array is: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void MyClass::print_array2(std::vector <long long int> const &a) {
    std::cout << "The vector elements are : ";
    for(int i=0; i < a.size(); i++) std::cout << a.at(i) << ' ';
    std::cout << std::endl;
}



// Inserisce i valori all'interno del vettore contente i valori finali a partire dal pattern
// Usato in funzioni come starting point, dove precedentemente non è richiesta la struttura
void MyClass::pattern_value_match(double* a, const long long int* b, double* full) {
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
void MyClass::constrain_jac_structure(Index* a,Index* b, std::vector<long long int> sparsity_pattern) {
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
void MyClass::pattern_value_match_constrains_jacobian(std::vector<long long int> pattern, double* values, double* final_values, int n_cons) {
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
std::vector<long long int> MyClass::pattern_merge_constrains(std::vector<long long int> a) {
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
std::vector<long long int> MyClass::pattern_merge(std::vector<long long int> a, std::vector<long long int> b) {
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
void MyClass::final_hess_structure(Index* a,Index* b, std::vector<long long int> sparsity_pattern) {
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

// Funzione per creare la mappatura della matrice hessiana
// Il vettore mappatura contiene in uscita -1 se l'elemento i-esimo del vettore constrains non va considerato, la posizione nel vettore finale altrimenti
void MyClass::constrain_hess_map(std::vector<long long int> cons_patt, std::vector<long long int> final_patt, std::vector<long long int>& map_vector, std::vector<long long int>& cons_vector) {
    int n_row = (int) cons_patt[0];
    int n_col = (int) cons_patt[1];
    int n_constrains = n_row/n_col;

    //std::vector<long long int> long_vector =
    //vector1.insert( vector1.end(), vector2.begin(), vector2.end() )

    int position = 0;
    int index_cons = 0;
    int map_index = 0;

    for(int i = 0; i < n_col; i++) {
        int nze_cons = (int) (cons_patt[3+i] - cons_patt[2+i]);   // Non-zero elements for the i_column short pattern
        int nze_final = (int) (final_patt[3+i] - final_patt[2+i]);   // Non-zero elements for the i_column long pattern
        //std::cout << "Map_index: " << map_index << std::endl;
        for(int j = 0; j < nze_cons; j++) {
            int row_index_cons = (int) (cons_patt[2+(n_col+1)+index_cons+j]);

            // # constrain a cui fa riferimento il valore
            int cons_index = 0;
            while(row_index_cons >= n_col) {
                row_index_cons -= n_col;
                cons_index++;
            }

            position = (int) final_patt[2+i];
            for(int l = 0; l < nze_final; l++) {
                // Elemento fuori dall triangolo inferiore
                if(i > row_index_cons) {
                    map_vector.push_back(-1);
                    cons_vector.push_back(-1);
                    map_index++;
                    break;
                }
                // Verifico che il valore in analisi sia nel range considerato (varia nel caso della matrice dei constrains in cui ho più righe che colonne)
                // Verifico si trovi nella parte bassa triangolare della matrice
                // Se il valore del pattern piccolo non corrisponde a quello del pattern finale, incremento la sua posizione e verifico la congruenza con l'elemento successivo
                // Elemento nel triangolo inferiore
                if(i <= row_index_cons  &&  row_index_cons == final_patt[2+n_col+1 + position]) {
                    map_vector.push_back(position);
                    cons_vector.push_back(cons_index);
                    map_index++;
                    //std::cout << "Row index: " << row_index_short << ", Position: " << position << ", Short value: " << short_value[index_short + j] << ", multiplier: " << multiplier[q] << ", final value: " << final_value[position] << std::endl;
                    position++;
                    break;
                }
                //std::cout << "Row index: " << row_index_short << std::endl;
                if(i <= row_index_cons) {
                    position++;
                }
            }
        }
        position = (int) final_patt[3+i];
        index_cons += nze_cons;
    }
}



// Funzione per riempire il vettore di valori finali dell'hessiano
// Deve restituire solo i valori diversi da 0 (contenuti nel pattern) e appartenenti al traingolo basso della matrice
void MyClass::pattern_value_match_hessian(std::vector<long long int> short_patt, double* short_value, std::vector<long long int> final_patt, double* final_value, int n_cons, const Number* multiplier) {
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


void MyClass::pattern_value_match_hessian_map(std::vector<long long int>& map_vector, std::vector<long long int>& cons_vector, double* cons_value, double* final_value, const Number* multiplier) {
    int position = 0;
    for(int i = 0; i < map_vector.size(); i++) {
        if(map_vector[i] != -1) {
            final_value[map_vector[i]] += multiplier[cons_vector[i]] * cons_value[i];
        }
    }
}


