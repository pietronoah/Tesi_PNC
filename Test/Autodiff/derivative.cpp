// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;


real f (real x, real y) {
    return pow(x,2) + y + atan(x) - cos(x-2*y); //This is a double variable function 
}

int main(){
    real x = 3, y = 1;
    real u = f(x,y);
    auto ux = derivative(f, wrt(x), at(x,y));

    std::cout << ux << std::endl;

    return 0;
}