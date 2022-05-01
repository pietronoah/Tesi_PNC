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
    int n = 5;


    double* x_path = (double *) malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
      x_path[i] = 1.0;
    }

    const double* x_path1 = x_path;



    const long long int* sparsity_pattern = obj_f_hes_sparsity_out(0);
    int n_row = sparsity_pattern[0];
    int n_col = sparsity_pattern[1];

    double* values1 = (double*) malloc(16*sizeof(double));
    double* values2 = (double*) malloc(16*sizeof(double));

    for(int i = 0; i < (n_row*n_col); i++) {
      values1[i] = 0;
      values2[i] = 0;
    }

    int position = 0;

    obj_f_hes(&x_path1, &values1, iw, w, mem);

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

          values2[position] += values1[values_index]; // In questo caso macherebbe obj_factor
        }
        values_index++;
      }
      index += nze;
    }

    



    sparsity_pattern = con_g_hes_sparsity_out(0);
    n_row = sparsity_pattern[0];
    n_col = sparsity_pattern[1];


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

    cout << "Array without zeroes:" << endl;
    print_array(values2, (n*(n+1))/2 );
    
    

    double* iRow = (double*) malloc(16*sizeof(double));
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


    std::cout << idx << std::endl;

    
    return 0;
}