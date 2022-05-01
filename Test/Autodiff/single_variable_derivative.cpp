// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
//using namespace autodiff;
using namespace autodiff;

real f (real x) {
    return pow(x,2); //This is a single variable function 
}

// Let's compute the derivative and them evaluate it

int main() {
    real x = 4; // Real is a data type included in autodiff 
    real u = f(x); 
    auto ux = derivative(f, wrt(x), at(x)); // I calculate the derivative and them evaluate it in a specific point
    std::cout << "The funtion evaluated in " << x << " is equal to: " << u << " and its derivative is: " << ux << std::endl;

    //auto uxx = jacobian(f, wrt(x), at(x));

    return 0;
}