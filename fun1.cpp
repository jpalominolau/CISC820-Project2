#include <iostream>
#include <vector>
#include <cmath>
#include "optimization.hpp"

double f(std::vector<double> x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        sum += (i + 1) * x[i] * x[i];
    }
    return sum;
}

std::vector<double> get_gradient(std::vector<double> x) {
    int n = x.size();
    std::vector<double> grad(n);
    for (int i = 0; i < n; i++) {
        grad[i] = 2.0 * (i + 1) * x[i];
    }
    return grad;
}

std::vector<std::vector<double>> get_hessian(std::vector<double> x) {
    int n = x.size();

    std::vector<std::vector<double>> hess(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        hess[i][i] = 2.0 * (i + 1);
    }
    return hess;
}

int main() {
    int n = 100;
    std::vector<double> x0(n, 1.0);

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

    plot_convergence("Function 1");

    return 0;
}