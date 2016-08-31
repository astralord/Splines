import numpy as np
import matplotlib.pyplot as plt
import Spline as splpckg
import random


class CurveFitter:
    def __init__(self, spline):
        self.spl = spline
        self.m_A_is_allocated = False
        self.m_norm_factors_are_counted = False
        self.m_gamma = self.m_beta = self.m_mu = 0
        self.m_delta = 0
        self.p = 0
        self.m_error = 0
        self.m_penalty = self.penalty()

    def penalty(self):
        k = self.spl.get_degree()
        g = self.spl.get_internal_knots_num()
        knots = self.spl.get_knots()
        self.m_penalty = 0
        for i in range(k, g + k + 1):
            self.m_penalty += 1.0 / (knots[i + 1] - knots[i])
        self.m_error = self.m_delta + self.p * self.m_penalty
        return self.m_penalty

    def interpolate_natural(self, points):
        k = self.spl.get_degree()
        n = self.spl.get_internal_knots_num() + 2
        knots = self.spl.get_knots()
        g = self.spl.get_internal_knots_num()

        f = points.copy()
        f[0] = -points[0] * self.spl.b_spline_derivative(knots[k], k, 0, 2)
        f[n - 1] = -points[n - 1] * self.spl.b_spline_derivative(knots[n - 1 + k], k, g + k, 2)

        a = [[0 for i in range(n)] for j in range(n)]
        a[0][0] = self.spl.b_spline_derivative(knots[k], k, 1, 2)
        a[0][1] = self.spl.b_spline_derivative(knots[k], k, 2, 2)
        for i in range(1, n - 1):
            a[i][i - 1] = self.spl.b_spline(knots[i + k], k, i)
            a[i][i] = self.spl.b_spline(knots[i + k], k, i + 1)
            a[i][i + 1] = self.spl.b_spline(knots[i + k], k, i + 2)
        a[n - 1][n - 1] = self.spl.b_spline_derivative(knots[n - 1 + k], k, g + k - 1, 2)
        a[n - 1][n - 2] = self.spl.b_spline_derivative(knots[n - 1 + k], k, g + k - 2, 2)

        res = np.linalg.solve(a, f)
        res = np.insert(res, 0, points[0])
        res = np.append(res, points[n - 1])

        self.spl.set_coefficients(res)
        return True

    def interpolate_clamped(self, points, left_bound, right_bound):
        k = self.spl.get_degree()
        n = self.spl.get_internal_knots_num() + 2
        knots = self.spl.get_knots()
        g = self.spl.get_internal_knots_num()

        f = points.copy()
        f[0] = left_bound - points[0] * self.spl.b_spline_derivative(knots[k], k, 0)
        f[n - 1] = right_bound - points[n - 1] * self.spl.b_spline_derivative(knots[n - 1 + k], k, g + k)

        a = [[0 for i in range(n)] for j in range(n)]
        a[0][0] = self.spl.b_spline_derivative(knots[k], k, 1)
        for i in range(1, n - 1):
            a[i][i - 1] = self.spl.b_spline(knots[i + k], k, i)
            a[i][i] = self.spl.b_spline(knots[i + k], k, i + 1)
            a[i][i + 1] = self.spl.b_spline(knots[i + k], k, i + 2)
        a[n - 1][n - 1] = self.spl.b_spline_derivative(knots[n - 1 + k], k, g + k - 1)

        res = np.linalg.solve(a, f)
        res = np.insert(res, 0, points[0])
        res = np.append(res, points[n - 1])

        self.spl.set_coefficients(res)

        return True

    # Thomas algorithm
    def solve_tridiag(self, f, left_diag, mid_diag, right_diag):
        n = self.spl.get_internal_knots_num() + 2
        x = self.spl.get_coefficients()
        for i in range(1, n):
            alpha = left_diag[i - 1] / mid_diag[i - 1]
            mid_diag[i] -= alpha * right_diag[i - 1]
            f[i] -= alpha * f[i - 1]
        x[n] = f[n - 1] / mid_diag[n - 1]
        for i in reversed(range(n - 1)):
            x[i + 1] = (f[i] - right_diag[i] * x[i + 2]) / mid_diag[i]


x = np.linspace(1, 10, 10)
y = np.random.rand(len(x))

# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 100)
y_new = [None] * 100

coefficients = [None] * (len(x) + 2)
s = splpckg.Spline(coefficients, x)
c = CurveFitter(s)
#c.interpolate_natural(y)
c.interpolate_clamped(y, 0, 0)

index = 0
for point in x_new:
    y_new[index] = s.get_value(point)
    index += 1

plt.plot(x, y, 'o', x_new, y_new)
plt.xlim([x[0] - 1, x[-1] + 1])
plt.show()