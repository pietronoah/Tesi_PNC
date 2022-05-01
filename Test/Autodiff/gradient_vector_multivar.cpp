// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
real f(const real x, const real y, const real z)
{
    return x * y + 2 * y; 
}

int main()
{
    using Eigen::VectorXd;

    real x = 7;                  
    real y = 3;
    real z = 3;

    real u = f(x,y,z);                                     // the output scalar u = f(x) evaluated together with gradient below

    VectorXd g = gradient(f, wrt(x,y,z), at(x,y,z), u); // evaluate the function value u and its gradient vector g = du/dx

    std::cout << "u = " << u << std::endl;      // print the evaluated output u
    std::cout << "The gradient vector is: " << "g = \n" << g << std::endl;    // print the evaluated gradient vector g = du/dx
}