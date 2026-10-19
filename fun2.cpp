#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "optimization.hpp"

int n = 100;
int m = 500;
std::vector<double> c;
std::vector<double> b;
std::vector<std::vector<double>> A;

double f(std::vector<double> x) {
    double term1 = 0.0;
    for (int j = 0; j < n; j++) {
        term1 += c[j] * x[j];
    }

    double term2 = 0.0;
    for (int i = 0; i < m; i++) {
        double Ax_i = 0.0;
        for (int j = 0; j < n; j++) {
            Ax_i += A[i][j] * x[j];
        }

        double diff = b[i] - Ax_i;
        if (diff <= 0.0) {
            return 1e18;
        }

        term2 += std::log(diff);
    }

    return term1 - term2;
}

std::vector<double> get_gradient(std::vector<double> x) {
    std::vector<double> grad = c;

    std::vector<double> d(m);
    for (int i = 0; i < m; i++) {
        double Ax_i = 0.0;
        for (int j = 0; j < n; j++) {
            Ax_i += A[i][j] * x[j];
        }
        
        double diff = b[i] - Ax_i;
        if (diff <= 1e-15) {
            diff = 1e-15;
        }
        d[i] = 1.0 / diff;
    }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            grad[j] += A[i][j] * d[i];
        }
    }

    return grad;
}

std::vector<std::vector<double>> get_hessian(std::vector<double> x) {
    std::vector<std::vector<double>> hess(n, std::vector<double>(n, 0.0));

    std::vector<double> w(m);
    for (int i = 0; i < m; i++) {
        double Ax_i = 0.0;
        for (int j = 0; j < n; j++) {
            Ax_i += A[i][j] * x[j];
        }
        w[i] = 1.0 / ((b[i] - Ax_i) * (b[i] - Ax_i));
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                hess[j][k] += w[i] * A[i][j] * A[i][k];
            }
        }
    }

    return hess;
}

int main() {
    c.resize(n);
    b.resize(m);
    A.assign(m, std::vector<double>(n));

    std::ifstream file_c("fun2_c.txt");
    if (!file_c.is_open()) {
        std::cerr << "Error opening c.txt\n";
        return 1;
    }
    for (int j = 0; j < n; j++) {
        file_c >> c[j];
    }
    file_c.close();

    std::ifstream file_b("fun2_b.txt");
    if (!file_b.is_open()) {
        std::cerr << "Error opening b.txt\n";
        return 1;
    }
    for (int i = 0; i < m; i++) {
        file_b >> b[i];
    }
    file_b.close();

    std::ifstream file_A("fun2_A.txt");
    if (!file_A.is_open()) {
        std::cerr << "Error opening A.txt\n";
        return 1;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            file_A >> A[i][j];
        }
    }
    file_A.close();

    std::cout << "Files loaded successfully!\n\n";

    std::vector<double> x0(n, 0.0);

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

    return 0;
}