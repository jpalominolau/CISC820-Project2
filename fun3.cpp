#include <iostream>
#include <vector>
#include <cmath>
#include "optimization.hpp"

double f(std::vector<double> x) {
    double term1 = 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
    double term2 = (1.0 - x[0]) * (1.0 - x[0]);
    return term1 + term2;
}

std::vector<double> get_gradient(std::vector<double> x) {
    std::vector<double> grad(2);
    grad[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    grad[1] = 200.0 * (x[1] - x[0] * x[0]);
    return grad;
}

std::vector<std::vector<double>> get_hessian(std::vector<double> x) {
    std::vector<std::vector<double>> hess(2, std::vector<double>(2, 0.0));
    hess[0][0] = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;
    hess[0][1] = -400.0 * x[0];
    hess[1][0] = -400.0 * x[0];
    hess[1][1] = 200.0;
    return hess;
}

int main() {
    int n = 2;
    std::vector<double> x0 = {-1.2, 1.0};

    double c_param = 0.1;
    double rho = 0.5;
    double alpha_init = 1.0;
    double tol = 0.000001;

    std::cout << "=== RUNNING GRADIENT DESCENT ===\n";
    gradient_descent(x0, c_param, rho, alpha_init, tol, 50000);

    std::cout << "\n=== RUNNING NEWTON METHOD ===\n";
    newton(x0, c_param, rho, alpha_init, tol, 100);

    std::cout << "\n=== RUNNING BFGS METHOD ===\n";
    bfgs(x0, c_param, rho, alpha_init, tol, 10000);

    plot_convergence("Function 3");

    return 0;
}