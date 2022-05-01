#include "source.c"
using namespace std;
#include <iostream>

    // Ora setto la variabile iw
    long long int* iw = (long long int*) malloc(sizeof(long long int));

    // Ora setto la variabile w
    double* w = (double*) malloc(sizeof(double));

    // Ora setto la variabile mem
    int mem = 1;

void print_array(double* a, int n) {
    for (int i = 0; i < n; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
}

int main() {
    int n = 4;


    double* x_path = (double *) malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
      x_path[i] = 1.0;
    }

    const double* x_path1 = x_path;



    const long long int* sparsity_pattern_1 = obj_f_hes_sparsity_out(0);
    int n_row_1 = sparsity_pattern_1[0];
    int n_col_1 = sparsity_pattern_1[1];



    const long long int* sparsity_pattern_2 = con_g_hes_sparsity_out(0);
    int n_row_2 = sparsity_pattern_2[0];
    int n_col_2 = sparsity_pattern_2[1];
    int n_constrain = n_row_2/n_col_2;


    const int* final_pattern = new int[2 + n_col_1+1 + n_col_1*n_row_1];

    for(int i = 0; i < (2 + n_col_1+1 + n_col_1*n_row_1); i++) {
      final_pattern[i] = 0;
    }

    final_pattern[0] = sparsity_pattern_1[0];
    final_pattern[1] = sparsity_pattern_1[1];


    for(int w = 0; w < (n_constrain); w++) {
      for(int i = 0; i < n_col_1; i++) {
        for(int j = 0; j < n_col_1; j++) {
          
        }
      }
    }











    

    /* double* iRow = (double*) malloc(16*sizeof(double));
    double* jCol = (double*) malloc(16*sizeof(double));


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

    cout << "iRow:" << endl;
    print_array(iRow, (n*(n+1))/2);
    cout << "jCol:" << endl;
    print_array(jCol, (n*(n+1))/2);
    cout << "idx:" << endl;
    printf("%d", idx);

    for (int i = -10; i < n_row*n_col; i++) {
      std::cout << values2[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << idx << std::endl; */

    
    return 0;
}