// C++ includes
#include <iostream>
#include <vector>

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;
using namespace std;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
var u(var* x)
{
    return x[1]*pow(x[0],2) + x[0]*pow(x[1],3) + x[1]*x[2];
}

int main()
{   

    var* x = (var*) calloc(3, sizeof(var));

    x[0] = 2;
    x[1] = 2;
    x[2] = 2;

    //var u = x[1]*pow(x[0],2) + x[0]*pow(x[1],3) + x[1]*x[2];  // the output variable u

    auto d_u = derivativesx(u(x), wrt(x[0],x[1],x[2]));

    array<detail::Variable<double>, 3>* vect = (array<detail::Variable<double>, 3>*) malloc(sizeof(array<detail::Variable<double>, 3>));

    vect[0] = derivativesx(d_u[0], wrt(x[0],x[1],x[2]));
    vect[1] = derivativesx(d_u[1], wrt(x[0],x[1],x[2]));
    vect[2] = derivativesx(d_u[2], wrt(x[0],x[1],x[2])); 




    cout << "u = " << u(x) << endl;  // print the evaluated output variable u

    cout << "ux = " << d_u[0] << endl;  // print the evaluated first order derivative ux
    cout << "uy = " << d_u[1] << endl;  // print the evaluated first order derivative uy
    cout << "uz = " << d_u[2] << endl;  // print the evaluated first order derivative uz 


    cout << "vect[0] = " << vect[0][0] << endl;
    cout << "vect[0] = " << vect[0][0] << vect[0][1] << vect[0][2] << endl;
    /* cout << "vect[1] = " << vect[1][0] << vect[1][1] << vect[1][2] << endl;
    cout << "vect[0] = " << vect[2][0] << vect[2][1] << vect[2][2] << endl; */

}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
In most cases, dual can be replaced by real, as commented in other examples.
However, computing higher-order cross derivatives has definitely to be done
using higher-order dual types (e.g., dual3rd, dual4th)! This is because real
types (e.g., real2nd, real3rd, real4th) are optimally designed for computing
higher-order directional derivatives.
-------------------------------------------------------------------------------------------------*/