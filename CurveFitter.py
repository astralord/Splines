import numpy as np
import matplotlib.pyplot as plt
import Spline as splpckg
import Point as pnt

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
        n = len(points)
        x = points.x
        self.spl.set_knots(x)

        f = [None] * n
        for i in range(1, n - 1):
            f[i] = points[i].y
        f[0] = -points[0].y * self.spl.b_spline_derivative(x[0], k, 0, 2)
        f[n - 1] = -points[n - 1].y * self.spl.b_spline_derivative(x[n - 1], k, n - 2 + k, 2)

        a = [[0 for i in range(n)] for j in range(n)]
        a[0][0] = self.spl.b_spline_derivative(x[0], k, 1, 2)
        a[0][1] = self.spl.b_spline_derivative(x[0], k, 2, 2)
        for i in range(1, n - 1):
            a[i][i - 1] = self.spl.b_spline(x[i], k, i)
            a[i][i] = self.spl.b_spline(x[i], k, i + 1)
            a[i][i + 1] = self.spl.b_spline(x[i], k, i + 2)
        a[n - 1][n - 1] = self.spl.b_spline_derivative(x[n - 1], k, n + k - 3, 2)
        a[n - 1][n - 2] = self.spl.b_spline_derivative(x[n - 1], k, n + k - 4, 2)

        res = np.linalg.solve(a, f)
        res = np.insert(res, 0, points[0].y)
        res = np.append(res, points[n - 1].y)

        self.spl.set_coefficients(res)

        return True

    def interpolate_clamped(self, points, left_bound, right_bound):
        k = self.spl.get_degree()
        n = len(points)
        x = points.x
        self.spl.set_knots(x)

        f = [None] * n
        for i in range(1, n - 1):
            f[i] = points[i].y
        f[0] = left_bound - points[0].y * self.spl.b_spline_derivative(x[0], k, 0)
        f[n - 1] = right_bound - points[n - 1].y * self.spl.b_spline_derivative(x[n - 1], k, n - 2 + k)

        a = [[0 for i in range(n)] for j in range(n)]
        a[0][0] = self.spl.b_spline_derivative(x[0], k, 1)
        for i in range(1, n - 1):
            a[i][i - 1] = self.spl.b_spline(x[i], k, i)
            a[i][i] = self.spl.b_spline(x[i], k, i + 1)
            a[i][i + 1] = self.spl.b_spline(x[i], k, i + 2)
        a[n - 1][n - 1] = self.spl.b_spline_derivative(x[n - 1], k, n + k - 3)

        res = np.linalg.solve(a, f)
        res = np.insert(res, 0, points[0].y)
        res = np.append(res, points[n - 1].y)

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

    def delta(self, points, sw):
        self.m_delta = 0
        for i in range(len(points)):
            e = points.weight() * (points.y() - self.spl.get_value(points.x))
            self.m_delta += e * e
        if sw > 0:
            k = self.spl.get_degree()
            g = self.spl.get_internal_knots_num()
            coefficients = self.spl.get_coefficients()
            for q in range(k + 1, g + k + 1):
                e = 0
                for i in range(q - k - 1, q + 1):
                    e += coefficients[i] * self.spl.get_lead_derivative_difference(i, q)
                    self.m_delta += sw * e * e
        self.m_error = self.m_delta + self.p * self.m_penalty
        return self.m_delta

    def penalty_derivative(self, knot_id):
        knots = self.spl.get_knots()
        a = knots[knot_id - 1]
        b = knots[knot_id]
        c = knots[knot_id + 1]
        bma = b - a
        cmb = c - b
        return 1.0 / (cmb * cmb) - 1.0 / (bma * bma)

    def error(self, points, sw):
        return self.delta(points, sw) + self.p * self.penalty()

    def error_derivative(self, points, sw, knot_id):
        # least - square error
        sum = 0
        for i in range(len(points)):
            w_sq = points.weight()
            w_sq *= w_sq
            diff = points.y() - self.spl.get_value(points.x())
            sum -= self.spl.get_value_derivative_knot(points.x(), knot_id) * w_sq * diff

        # penalty error
        if self.p > 0:
            sum += 0.5 * self.p * self.penalty_derivative(knot_id)

        # smoothing error
        if sw > 0:
            k = self.spl.get_degree()
            g = self.spl.get_internal_knots_num()
            coefficients = self.spl.get_coefficients()
            for q in range(k + 1, g + k + 1):
                sum1 = 0
                sum2 = 0
                for i in range(q - k - 1, q + 1):
                    ci = coefficients[i]
                    lead_der_diff = self.spl.get_lead_derivative_difference(i, q)
                    sum1 += ci * lead_der_diff
                    sum2 += ci * self.spl.get_lead_der_diff_der_knot(lead_der_diff, i, q, knot_id)
                sum += 2 * sw * sum1 * sum2

        return 2 * sum

    def theta(self, points, sw, alpha):
        knots = self.spl.get_knots()
        k = self.spl.get_degree()
        g = self.spl.get_internal_knots_num()
        for i in range(g):
            knots[i + k + 1] = self.m_fixed_knots[i + k + 1] + alpha * self.m_dir[i]

        if self.approximate(points, sw):
            return self.error(points, sw)
        return None

    @staticmethod
    def norm(v):
        return sum(i * i for i in v)

    def approximate(self, points, sw):
        k = self.spl.get_degree()
        g = self.spl.get_internal_knots_num()
        n = len(points)
        coefs = [0] * (g + k + 1)
        A = [[0 for i in range(g + k + 1)] for j in range(g + k + 1)]

        # spline error
        l = 0
        for r in range(n):
            l = self.spl.get_left_node_index(points[r].x, l)
            if l < 0:
                return
            b_splines = self.spl.b_splines(points[r].x, k)
            for i in range(k + 1):
                w_sq = points[r].w * points[r].w
                for j in range(i + 1):
                    A[i + l - k][j + l - k] += w_sq * b_splines[i] * b_splines[j]
                coefs[i + l - k] += w_sq * points[r].y * b_splines[i]

        # smoothing
        if sw > 0:
            for q in range(g):
                for i in range(q, q + k + 2):
                    ai = self.spl.get_lead_derivative_difference(i, q + k + 1)
                    for j in range(q, i + 1):
                        A[i][j] += sw * ai * self.spl.get_lead_derivative_difference(j, q + k + 1)

        for i in range(g + k + 1):
            for j in range(i):
                A[j][i] = A[i][j]

        # solve equation
        L = np.linalg.cholesky(A)

        # solve Lx = r
        for i in range(g + k + 1):
            for j in range(i):
                coefs[i] -= L[i][j] * coefs[j]
            coefs[i] /= L[i][i]

        # solve Uy = x
        for i in reversed(range(g + k + 1)):
            for j in reversed(range(i + 1, g + k + 1)):
                coefs[i] -= L[j][i] * coefs[j]
            coefs[i] /= L[i][i]

        self.spl.set_coefficients(coefs)


x = range(12)
y = np.random.rand(len(x))
w = [1] * len(x)

knots = [0, 2, 5, 8, 10, 11]

points = pnt.Points(x, y, w)

# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 100)
y_new = [None] * 100

coefficients = [None] * (len(x) + 2)
s = splpckg.Spline(coefficients, knots)
c = CurveFitter(s)
#c.interpolate_natural(points)
#c.interpolate_clamped(points, 0, 0)
c.approximate(points, 0)

index = 0
for point in x_new:
    y_new[index] = s.get_value(point)
    index += 1

index = 0
knots_y = [0] * len(knots)
for point in knots:
    knots_y[index] = s.get_value(point)
    index += 1

plt.plot(x_new, y_new, x, y, 'o', knots, knots_y, 'o')
plt.xlim([x[0] - 1, x[-1] + 1])
plt.show()