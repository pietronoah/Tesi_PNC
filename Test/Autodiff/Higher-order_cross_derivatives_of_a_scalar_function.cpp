// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;
using namespace std;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
dual2nd f(dual2nd* x)
{
    return x[1]*pow(x[0],2) + x[0]*pow(x[1],3) + x[1]*x[2];
}

int main()
{

    dual2nd* p = (dual2nd*) calloc(3, sizeof(dual2nd));

    p[0] = 2;
    p[1] = 2;
    p[2] = 2;


    auto g = gradient(f, wrt(p[0],p[1],p[2]), at(p));

    cout << "ux = " << g[0] << endl;  // print the evaluated output variable u

    cout << "uy = " << g[1] << endl;  // print the evaluated first order derivative ux

    cout << "uz = " << g[2] << endl;  // print the evaluated first order derivative ux

    
    //auto d_g = gradient(g, wrt(x[0],x[1],x[2]), at(x));





    auto d_u = derivatives(f, wrt(p[0], p[0]), at(p));

    cout << "u = " << d_u[0] << endl;  // print the evaluated output variable u

    cout << "ux = " << d_u[1] << endl;  // print the evaluated first order derivative ux

    cout << "uxx = " << d_u[2] << endl;  // print the evaluated second order derivative uxx 
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