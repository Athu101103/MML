#import statements
from tabulate import tabulate
import sympy as sp0
import matplotlib.pyplot as plt
import math
import numpy as np

def f(x,y): #first order differential equation
  return 2*x - 3*y

def euler(f, x0, y0, h, n):
  x_points = [x0]
  y_points = [y0]

  for i in range(n):
    x_n = x_points[-1]
    y_n = y_points[-1]
    x_n_plus_1 = round(x_n + h, 2)
    slope_value = f(x_n, y_n)
    y_n_plus_1 = y_n + (h*slope_value)

    x_points.append(x_n_plus_1)
    y_points.append(y_n_plus_1)

  return x_points, y_points

def runge_kutta(f, x0, y0, h, n):
  x_points = [x0]
  y_points = [y0]

  for i in range(n):
    x_n = x_points[-1]
    y_n = y_points[-1]

    x_n_plus_1 = round(x_n + h, 2)
    k1 = h * f(x_n, y_n)
    k2 = h * f(x_n + (h/2), y_n + (k1/2))
    k3 = h * f(x_n + (h/2), y_n + (k2/2))
    k4 = h * f(x_n + h , y_n + k3)

    y_n_plus_1 = y_n + ((1/6)*(k1 + (2*k2) + (2*k3) + k4))

    x_points.append(x_n_plus_1)
    y_points.append(y_n_plus_1)

  return x_points, y_points

x0 = 0
y0 = 1
h = 0.1
n = 20  # 20 values → 21 points


print("\nDifferential Equation:")
print("dy/dx = 2x - 3y")
print(f"Initial Condition: y({x0}) = {y0}")
print(f"Step size (h): {h}, Number of steps: {n}")
print("-" * 50)

import sympy as sp

# Define variables
x = sp.Symbol('x')
y = sp.Function('y')

# Differential equation dy/dx = 2x - 3y
diffeq = sp.Eq(sp.diff(y(x), x), 2*x - 3*y(x))

# Solve ODE with initial condition y(0) = 1
solution = sp.dsolve(diffeq, y(x), ics={y(0): 1})

print("\nSolution with IC y(0)=1:")
print(solution)

# Convert to numerical function
y_analytical_fn = sp.lambdify(x, solution.rhs, 'numpy')

#Euler method
x_euler, y_euler = euler(f, x0, y0, h, n)

#Computing y_actual values using the function found using analytical method
y_analytical_values = [y_analytical_fn(x_val) for x_val in x_euler]


table_euler = []
for i in range(n+1):
  table_euler.append([i, x_euler[i], round(y_euler[i], 5), round(y_analytical_values[i],5)])

print("\nEuler Method Table:\n")
print(tabulate(table_euler, headers=["Step", "x", "y (Euler)","Analytical y"], tablefmt="fancy_grid"))

#RK method

x_rk, y_rk = runge_kutta(f, x0, y0, h, n)
table_rk = []
for i in range(n+1):
  table_rk.append([i, x_rk[i], round(y_rk[i], 5), round(y_analytical_values[i],5)])

print("\nRunge Kutta (4th order) Method: \n")
print(tabulate(table_rk, headers=["Step", "x", "y (RK-4)", "Analytical y"], tablefmt="fancy_grid"))

from scipy.stats import ttest_ind

# T-test: Euler vs Analytical
t_euler = ttest_ind(y_euler, y_analytical_values)
print("\nT-Test: Euler vs Analytical")
print(f"  t-statistic = {t_euler.statistic:.5f}, p-value = {t_euler.pvalue:.5f}")
if t_euler.pvalue < 0.05:
    print("   Euler is significantly different from the analytical solution.")
else:
    print("  Euler is NOT significantly different from the analytical solution.")

# T-test: RK4 vs Analytical
t_rk4 = ttest_ind(y_rk, y_analytical_values)
print("\nT-Test: RK4 vs Analytical")
print(f"  t-statistic = {t_rk4.statistic:.5f}, p-value = {t_rk4.pvalue:.5f}")
if t_rk4.pvalue < 0.05:
    print("  RK4 is significantly different from the analytical solution.")
else:
    print("  RK4 is NOT significantly different from the analytical solution.")

# Plot each method
plt.figure(figsize=(10, 6))
plt.plot(x_euler, y_analytical_values, label="Analytical Solution", linewidth=5)
plt.plot(x_euler, y_rk, label="Runge-Kutta (RK4)", linestyle='--', marker='o')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of RK with Analytical")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n\n")


plt.figure(figsize=(10, 6))
plt.plot(x_euler, y_analytical_values, label="Analytical Solution", linewidth=5)
plt.plot(x_euler, y_euler, label="Euler Method", linestyle='-.', marker='s')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Euler with Analytical")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n\n")