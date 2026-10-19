#include "optimization.hpp"
#include <iostream>
#include <cmath>

double get_gradient_norm(std::vector<double> grad) {
    double sum = 0.0;
    for (int i = 0; i < grad.size(); i++) {
        sum += grad[i] * grad[i];
    }
    return std::sqrt(sum);
}

double armijo(std::vector<double> x, std::vector<double> p, double c, double rho, double alpha_init) {
    double alpha = alpha_init;
    std::vector<double> grad = get_gradient(x);

    double grad_dot_p = 0.0;
    for (int i = 0; i < x.size(); i++) {
        grad_dot_p += grad[i] * p[i];
    }

    while (true) {
        if (alpha < 1e-12) {
            return 0.0;
        }

        std::vector<double> x_new(x.size());
        for (int i = 0; i < x.size(); i++) {
            x_new[i] = x[i] + alpha * p[i];
        }

        if (f(x_new) <= f(x) + c * alpha * grad_dot_p) {
            break;
        }

        alpha *= rho;
    }

    return alpha;
}

std::vector<double> solve_linear_system(std::vector<std::vector<double>> H, std::vector<double> g) {
    int n = g.size();
    std::vector<std::vector<double>> A = H;
    std::vector<double> b(n);
    for (int i = 0; i < n; i++) {
        b[i] = -g[i];
    }

    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[k][i]) > std::abs(A[max_row][i])) {
                max_row = k;
            }
        }
        std::swap(A[i], A[max_row]);
        std::swap(b[i], b[max_row]);

        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    std::vector<double> p(n);
    for (int i = n - 1; i >= 0; i--) {
        p[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            p[i] -= A[i][j] * p[j];
        }
        p[i] /= A[i][i];
    }

    return p;
}

void gradient_descent(std::vector<double> x0, double c, double rho, double alpha_init, double tol, int max_iter) {
    std::vector<double> x = x0;

    for (int iter = 1; iter <= max_iter; iter++) {
        std::vector<double> grad = get_gradient(x);

        double norm = get_gradient_norm(grad);
        if (norm < tol) {
            std::cout << "Converged in " << iter - 1 << " iterations!\n\n";
            break;
        }

        std::vector<double> p(x.size());
        for (int i = 0; i < x.size(); i++) {
            p[i] = -grad[i];
        }

        double alpha = armijo(x, p, c, rho, alpha_init);

        if (alpha < 1e-8) {
            std::cout << "Line search failed (alpha < 1e-8) at iteration " << iter << "\n";
            break;
        }

        for (int i = 0; i < x.size(); i++) {
            x[i] += alpha * p[i];
        }

        std::vector<double> grad_new = get_gradient(x);
        double norm_new = get_gradient_norm(grad_new);

        std::cout << "Iteration " << iter 
                  << " | f(x): " << f(x) 
                  << " | norm: " << norm_new 
                  << " | alpha: " << alpha << "\n";
    }

    std::cout << "--- OPTIMAL POINT FOUND (GRADIENT DESCENT) ---\n";
    std::cout << "Minimum f(x*): " << f(x) << "\n";
    // std::cout << "At x*: (" << x[0] << ", " << x[1] << ")\n";
    std::cout << "First 3 entries of x*: (" << x[0] << ", " << x[1] << ", " << x[2] << " ...)\n";
}

void newton(std::vector<double> x0, double c, double rho, double alpha_init, double tol, int max_iter) {
    std::vector<double> x = x0;

    for (int iter = 1; iter <= max_iter; iter++) {
        std::vector<double> grad = get_gradient(x);

        double norm = get_gradient_norm(grad);
        if (norm < tol) {
            std::cout << "Converged in " << iter - 1 << " iterations!\n\n";
            break;
        }

        std::vector<std::vector<double>> hess = get_hessian(x);

        // std::vector<double> p(x.size());
        // for (int i = 0; i < x.size(); i++) {
        //     p[i] = -grad[i] / hess[i][i];
        // }

        std::vector<double> p = solve_linear_system(hess, grad);

        double alpha = armijo(x, p, c, rho, alpha_init);

        for (int i = 0; i < x.size(); i++) {
            x[i] += alpha * p[i];
        }

        std::vector<double> grad_new = get_gradient(x);
        double norm_new = get_gradient_norm(grad_new);

        std::cout << "Iteration " << iter 
                  << " | f(x): " << f(x) 
                  << " | norm: " << norm_new 
                  << " | alpha: " << alpha << "\n";
    }

    std::cout << "--- OPTIMAL POINT FOUND (NEWTON) ---\n";
    std::cout << "Minimum f(x*): " << f(x) << "\n";
    // std::cout << "At x*: (" << x[0] << ", " << x[1] << ")\n";
    std::cout << "First 3 entries of x*: (" << x[0] << ", " << x[1] << ", " << x[2] << " ...)\n";
}

void bfgs(std::vector<double> x0, double c, double rho, double alpha_init, double tol, int max_iter) {
    std::vector<double> x = x0;
    int n = x.size();

    std::vector<std::vector<double>> F(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) F[i][i] = 1.0;

    for (int iter = 1; iter <= max_iter; iter++) {
        std::vector<double> grad = get_gradient(x);

        double norm = get_gradient_norm(grad);
        if (norm < tol) {
            std::cout << "Converged in " << iter - 1 << " iterations!\n\n";
            break;
        }

        std::vector<double> p(n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p[i] -= F[i][j] * grad[j];
            }
        }

        double alpha = armijo(x, p, c, rho, alpha_init);

        if (alpha < 1e-8) {
            std::cout << "Line search failed (alpha < 1e-8) at iteration " << iter << "\n";
            break;
        }

        std::vector<double> s(n);
        std::vector<double> x_next(n);
        for (int i = 0; i < n; i++) {
            s[i] = alpha * p[i];
            x_next[i] = x[i] + s[i];
        }

        std::vector<double> grad_next = get_gradient(x_next);
        std::vector<double> y(n);
        for (int i = 0; i < n; i++) {
            y[i] = grad_next[i] - grad[i];
        }

        double y_dot_s = 0.0;
        for (int i = 0; i < n; i++) {
            y_dot_s += y[i] * s[i];
        }

        if (std::abs(y_dot_s) > 1e-12) {
            std::vector<double> Fy(n, 0.0);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Fy[i] += F[i][j] * y[j];
                }
            }

            double y_dot_Fy = 0.0;
            for (int i = 0; i < n; i++) {
                y_dot_Fy += y[i] * Fy[i];
            }

            std::vector<std::vector<double>> F_next = F;
            double term1_coeff = (y_dot_Fy + y_dot_s) / (y_dot_s * y_dot_s);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double t1 = term1_coeff * s[i] * s[j];
                    double t2 = (s[i] * Fy[j] + Fy[i] * s[j]) / y_dot_s;
                    F_next[i][j] += t1 - t2;
                }
            }
            F = F_next;
        }

        x = x_next;

        double norm_new = get_gradient_norm(grad_next);

        std::cout << "Iteration " << iter 
                  << " | f(x): " << f(x) 
                  << " | norm: " << norm_new 
                  << " | alpha: " << alpha << "\n";
    }

    std::cout << "--- OPTIMAL POINT FOUND (BFGS) ---\n";
    std::cout << "Minimum f(x*): " << f(x) << "\n";
    // std::cout << "At x*: (" << x[0] << ", " << x[1] << ")\n";
    std::cout << "First 3 entries of x*: (" << x[0] << ", " << x[1] << ", " << x[2] << " ...)\n";
}