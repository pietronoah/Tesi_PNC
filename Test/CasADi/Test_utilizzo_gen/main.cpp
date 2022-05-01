//#include "gen.h"
#include "source/gen.c"

#include <iostream>
#include <cstdlib>
using namespace std;


int main() {

    int x = 1, y = 1;

    // Creo il vettore in cui andrò a mettere le mie variabili del problema
    double* arg1 = (double*) malloc(2*sizeof(double));
    /* for(int i = 0; i < (sizeof(*arg1) / sizeof(int)); i++) {
        arg1[i] = 0;
    } */
    //cout << arg1[0] << endl;

    arg1[0] = x;
    arg1[1] = y;

    // Riporto questo vettore con un doppio puntatore 
    double** arg2 = &arg1;
    const double** arg;
    arg = (const double**) arg2;
    //cout << arg[0][0] << endl;


    // Ora imposto il vettore che conterrà il risultato delle evaluations

    double* res1 = (double*) malloc(2*sizeof(double));
    for(int i = 0; i < (sizeof(*arg1) / sizeof(double)); i++) {
        res1[i] = 0;
    }

    double** res = &res1;


    // Ora setto la variabile iw
    long long int* iw = (long long int*) malloc(sizeof(long long int));

    // Ora setto la variabile w
    double* w = (double*) malloc(sizeof(double));

    // Ora setto la variabile mem
    int mem;


    

    Function(arg, res, iw, w, mem);

    cout << res[0][0] << ", " << res[0][1] << endl;



    return 0;
}

