#include "source.c"
#include <iostream>
#include <vector>
#include <algorithm>

    // Ora setto la variabile iw
    long long int* iw = (long long int*) malloc(sizeof(long long int));

    // Ora setto la variabile w
    double* w = (double*) malloc(sizeof(double));

    // Ora setto la variabile mem
    int mem = 1;

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
void pattern_value_match_hessian(std::vector<long long int> short_patt, double* short_value, std::vector<long long int> long_patt, double* long_value, int n_cons) {
    int q;
    for(q = 0; q < 2; q++) {
        int n_row = (int) short_patt[0];
        int n_col = (int) short_patt[1];
        int index_short = 0;
        int index_long = 0;
        for(int i = 0; i < n_col; i++) {
            int nze_short = (int) (short_patt[3+i] - short_patt[2+i]);   // Non-zero elements for the i_column short pattern
            int nze_long = (int) (long_patt[3+i] - long_patt[2+i]);   // Non-zero elements for the i_column long pattern
            for(int j = 0; j < nze_short; j++) {
                int row_index = (int) (short_patt[2+(n_col+1)+index_short+j]);
                std::cout << "q: " << q << std::endl;
                if(i <= (row_index-n_col*q)  &&  row_index >= (n_col*q)  &&  row_index < (n_col*(q+1))) {
                    //std::cout << "Row index: " << row_index << std::endl;
                    int position = 0;
                    std::vector<long long int>::iterator itr = std::find(long_patt.begin()+(2+n_col+1+index_long), long_patt.begin()+(2+n_col+1+index_long+nze_long), row_index-(n_col*q));
                    for(int k = 1; k < ((row_index + 1) - (n_col * q)); k++) { // In questo modo ottengo una posizione relativa al constrain in analisi
                        position += k;
                    }
                    position += i;
                    //int position = std::distance(long_patt.begin()+(2+n_col+1), itr);
                    std::cout << "q: " << q << std::endl;
                    std::cout << "Position :" << position << std::endl;
                    std::cout << "Short value :" << short_value[index_short+j] << std::endl;
                    long_value[position] += short_value[index_short+j];
                }
            }
            index_short += nze_short;
            index_long += nze_long;
        }
    }
}






int main() {
    int n = 4;
//---------------Test----------------------------------------------------
    /*std::vector<long long int> vect1;

    vect1.push_back(2);
    vect1.push_back(2);
    vect1.push_back(0);
    vect1.push_back(1);
    vect1.push_back(2);
    vect1.push_back(0);
    vect1.push_back(1);

    std::vector<long long int> vect2;

    vect2.push_back(2);
    vect2.push_back(2);
    vect2.push_back(0);
    vect2.push_back(1);
    vect2.push_back(2);
    vect2.push_back(0);
    vect2.push_back(0);

    std::vector<long long int> test22 = pattern_merge(vect1,vect2);
    print_array2(test22);*/
//-----------------------------------------------------------------------



    auto* x_path = (double *) malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        x_path[i] = 1.0;
    }

    const double* x_path1 = x_path;



    const long long int* sparsity_pattern = obj_f_hes_sparsity_out(0);
    std::vector<long long int> test(sparsity_pattern, sparsity_pattern + (2+n+1+n*n));
    //int sizep = sizeof(sparsity_pattern) / sizeof(sparsity_pattern[0]);

    /*std::vector<long long int> test(sparsity_pattern, sparsity_pattern + (2+n+1+n*n));
    print_array2(test);
    std::vector<long long int> test2 = pattern_merge(test,test);
    print_array2(test2);*/

    int n_row = (int) sparsity_pattern[0];
    int n_col = (int) sparsity_pattern[1];

    auto* values1 = new double[16]();
    auto* values2 = (double*) malloc(16*sizeof(double));

    for(int i = 0; i < (n_row*n_col); i++) {
        values1[i] = 0;
        values2[i] = 0;
    }

    int position = 0;

    obj_f_hes(&x_path1, &values1, iw, w, mem);
    std::cout << "Values: ";
    print_array(values1, n*n);

    int index = 0;
    int values_index = 0;
    for(int i = 0; i < n_col; i++) {
        int nze = (int) (sparsity_pattern[3+i] - sparsity_pattern[2+i]);   // Non zero elements for the i_column
        for(int j = 0; j < nze; j++) {
            if(i <= sparsity_pattern[2+(n_col+1)+index+j]) { // Only if the element is in the lower part of the matrix (column_index lower than row_index)
                position = 0;
                for(int k = 1; k < (sparsity_pattern[2+(n_col+1)+index+j] + 1); k++) {
                    position += k;
                }
                position += i;
                values2[position] += values1[values_index]; // In questo caso mancherebbe obj_factor
            }
            values_index++;
        }
        index += nze;
    }



    values1 = (double*) malloc(16*sizeof(double));

    for(int i = 0; i < (n_row*n_col); i++) {
        values1[i] = 0;
    }


    int n_constrain = n_row/n_col;

    con_g_hes(&x_path1, &values1, iw, w, mem);

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

                    values2[position] += values1[values_index];
                }
                values_index++;
            }
            index += nze;
        }

    }

    //std::cout << "Array without zeroes:" << std::endl;
    //print_array(values2, (n*(n+1))/2 + 2);



    auto* iRow = (double*) malloc(16*sizeof(double));
    auto* jCol = (double*) malloc(16*sizeof(double));


    int idx = 0;
    for (int i = 0; i < (n*(n+1))/2; i++) {
        if (values2[i+1] != 0) {
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



    sparsity_pattern = con_g_hes_sparsity_out(0);

    for(int i = 0; i < (n_row*n_col); i++) {
        values1[i] = 0;
        values2[i] = 0;
    }
    obj_f_hes(&x_path1, &values1, iw, w, mem);
    con_g_hes(&x_path1, &values2, iw, w, mem);


    std::vector<long long int> test_hes(sparsity_pattern, sparsity_pattern + 30);
    std::vector<long long int> test_hes2 = pattern_merge_constrains(test_hes); // Restituisce il patter compresso
    std::vector<long long int> final_pattern = pattern_merge(test,test_hes2);
    std::cout << "Test pattern: ";
    print_array2(test);
    std::cout << "Hes_g pattern: ";
    print_array2(test_hes2);
    std::cout << "Final pattern: ";
    print_array2(final_pattern);

    long long int nze_final = final_pattern[6]; // Number of non-zero elements in the final pattern
    auto* final_values = new double[nze_final]();
    pattern_value_match_hessian(test,values1,final_pattern,final_values,1);
    std::cout << "-----------------------------------------------" << std::endl;
    pattern_value_match_hessian(test_hes,values2,final_pattern,final_values,2);
    std::cout << "Final values: ";
    print_array(final_values,nze_final);









    //std::cout << "iRow:" << std::endl;
    //print_array(iRow, (n*(n+1))/2);
    //std::cout << "jCol:" << std::endl;
    //print_array(jCol, (n*(n+1))/2);
    //std::cout << "idx:" << std::endl;
    //printf("%d", idx);

    //for (int i = -10; i < n_row*n_col; i++) {
    //    std::cout << values2[i] << " ";
    //}
    //std::cout << "" << std::endl;
    //std::cout << idx << std::endl;


    return 0;
}