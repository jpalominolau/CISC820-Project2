#pragma once

#include <vector>
#include <string>

double f(std::vector<double> x);
std::vector<double> get_gradient(std::vector<double> x);
std::vector<std::vector<double>> get_hessian(std::vector<double> x);

double get_gradient_norm(std::vector<double> grad);
double armijo(std::vector<double> x, std::vector<double> p, double c, double rho, double alpha_init);
std::vector<double> solve_linear_system(std::vector<std::vector<double>> H, std::vector<double> g);

void gradient_descent(std::vector<double> x0, double c, double rho, double alpha_init, double tol, int max_iter);
void newton(std::vector<double> x0, double c, double rho, double alpha_init, double tol, int max_iter);
void bfgs(std::vector<double> x0, double c, double rho, double alpha_init, double tol, int max_iter);

void plot_convergence(std::string title);