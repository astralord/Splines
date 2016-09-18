import numpy as np
import matplotlib.pyplot as plt
import Spline as splpckg
import Point as pnt
import math


class CurveFitter:
    def __init__(self, spline):
        self.m_A_is_allocated = False
        self.m_norm_factors_are_counted = False
        self.m_gamma = self.m_beta = self.m_mu = 1
        self.m_delta = 0
        self.p = 0
        self.m_error = 0
        self.m_penalty = self.penalty(spline)

    def penalty(self, spline):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        knots = spline.get_knots()
        self.m_penalty = 0
        for i in range(k, g + k + 1):
            self.m_penalty += 1.0 / (knots[i + 1] - knots[i])
        self.m_error = self.m_delta + self.p * self.m_penalty
        return self.m_penalty

    @staticmethod
    def interpolate_natural(spline, points):
        k = spline.get_degree()
        n = len(points)
        x = points.x
        spline.set_knots(x)

        f = [None] * n
        for i in range(1, n - 1):
            f[i] = points[i].y
        f[0] = -points[0].y * spline.b_spline_derivative(x[0], k, 0, 2)
        f[n - 1] = -points[n - 1].y * spline.b_spline_derivative(x[n - 1], k, n - 2 + k, 2)

        a = [[0 for i in range(n)] for j in range(n)]
        a[0][0] = spline.b_spline_derivative(x[0], k, 1, 2)
        a[0][1] = spline.b_spline_derivative(x[0], k, 2, 2)
        for i in range(1, n - 1):
            a[i][i - 1] = spline.b_spline(x[i], k, i)
            a[i][i] = spline.b_spline(x[i], k, i + 1)
            a[i][i + 1] = spline.b_spline(x[i], k, i + 2)
        a[n - 1][n - 1] = spline.b_spline_derivative(x[n - 1], k, n + k - 3, 2)
        a[n - 1][n - 2] = spline.b_spline_derivative(x[n - 1], k, n + k - 4, 2)

        res = np.linalg.solve(a, f)
        res = np.insert(res, 0, points[0].y)
        res = np.append(res, points[n - 1].y)

        spline.set_coefficients(res)

        return True

    @staticmethod
    def interpolate_clamped(spline, points, left_bound, right_bound):
        k = spline.get_degree()
        n = len(points)
        x = points.x
        spline.set_knots(x)

        f = [None] * n
        for i in range(1, n - 1):
            f[i] = points[i].y
        f[0] = left_bound - points[0].y * spline.b_spline_derivative(x[0], k, 0)
        f[n - 1] = right_bound - points[n - 1].y * spline.b_spline_derivative(x[n - 1], k, n - 2 + k)

        a = [[0 for i in range(n)] for j in range(n)]
        a[0][0] = spline.b_spline_derivative(x[0], k, 1)
        for i in range(1, n - 1):
            a[i][i - 1] = spline.b_spline(x[i], k, i)
            a[i][i] = spline.b_spline(x[i], k, i + 1)
            a[i][i + 1] = spline.b_spline(x[i], k, i + 2)
        a[n - 1][n - 1] = spline.b_spline_derivative(x[n - 1], k, n + k - 3)

        res = np.linalg.solve(a, f)
        res = np.insert(res, 0, points[0].y)
        res = np.append(res, points[n - 1].y)

        spline.set_coefficients(res)

        return True

    def delta(self, spline, points, sw):
        self.m_delta = 0
        for i in range(len(points)):
            e = points[i].w * (points[i].y - spline.get_value(points[i].x))
            self.m_delta += e * e
        nu = 0
        if sw > 0:
            k = spline.get_degree()
            g = spline.get_internal_knots_num()
            coefficients = spline.get_coefficients()
            for q in range(k + 1, g + k + 1):
                e = 0
                for i in range(q - k - 1, q + 1):
                    e += coefficients[i] * spline.get_lead_derivative_difference(i, q)
                    nu += e * e
            nu *= sw
        self.m_delta += nu
        self.m_error = self.m_delta + self.p * self.m_penalty
        return self.m_delta

    @staticmethod
    def penalty_derivative(spline, knot_id):
        knots = spline.get_knots()
        a = knots[knot_id - 1]
        b = knots[knot_id]
        c = knots[knot_id + 1]
        bma = b - a
        cmb = c - b
        return 1.0 / (cmb * cmb) - 1.0 / (bma * bma)

    def error(self, spline, points, sw):
        return self.delta(spline, points, sw) + self.p * self.penalty(spline)

    def error_derivative(self, spline, points, sw, knot_id):
        # least-square error
        grad_error = 0
        for i in range(len(points)):
            w_sq = points[i].w * points[i].w
            diff = points[i].y - spline.get_value(points[i].x)
            grad_error -= spline.get_value_derivative_knot(points[i].x, knot_id) * w_sq * diff

        # penalty error
        if self.p > 0:
            grad_error += 0.5 * self.p * self.penalty_derivative(spline, knot_id)

        # smoothing error
        if sw > 0:
            sm_error = 0
            k = spline.get_degree()
            g = spline.get_internal_knots_num()
            coefficients = spline.get_coefficients()
            for q in range(k + 1, g + k + 1):
                sum1 = 0
                sum2 = 0
                for i in range(q - k - 1, q + 1):
                    ci = coefficients[i]
                    lead_der_diff = spline.get_lead_derivative_difference(i, q)
                    sum1 += ci * lead_der_diff
                    sum2 += ci * spline.get_lead_der_diff_der_knot(lead_der_diff, i, q, knot_id)
                sm_error += sum1 * sum2
            grad_error += sw * sm_error
        return 2 * grad_error

    def theta(self, spline, points, sw, alpha):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        knots = [0] * (g + 2)
        knots[0] = spline.get_left_bound()
        knots[g + 1] = spline.get_right_bound()
        for i in range(g):
            knots[i + 1] = self.m_fixed_knots[i + k + 1] + alpha * self.m_dir[i]
        spline.set_knots(knots)

        if self.approximate(spline, points, sw):
            return self.error(spline, points, sw)
        return -1

    @staticmethod
    def norm(v):
        return sum(i * i for i in v)

    @staticmethod
    def approximate(spline, points, sw):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        n = len(points)
        coefs = [0] * (g + k + 1)
        A = [[0 for i in range(g + k + 1)] for j in range(g + k + 1)]

        # spline error
        l = 0
        for r in range(n):
            l = spline.get_left_node_index(points[r].x, l)
            if l < 0:
                return
            b_splines = spline.b_splines(points[r].x, k)
            for i in range(k + 1):
                w_sq = points[r].w * points[r].w
                for j in range(i + 1):
                    A[i + l - k][j + l - k] += w_sq * b_splines[i] * b_splines[j]
                coefs[i + l - k] += w_sq * points[r].y * b_splines[i]

        # smoothing
        if sw > 0:
            for q in range(g):
                for i in range(q, q + k + 2):
                    ai = spline.get_lead_derivative_difference(i, q + k + 1)
                    for j in range(q, i + 1):
                        A[i][j] += sw * ai * spline.get_lead_derivative_difference(j, q + k + 1)

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

        spline.set_coefficients(coefs)

        return True

    @staticmethod
    def is_grid_valid(spline, points):
        knots = spline.get_knots()
        k = spline.get_degree()
        n = len(points)
        j = 0
        while j < n and points[j].x < knots[0]:
            j += 1

        for i in range(spline.get_internal_knots_num() + k + 1):
            while j < n and points[j].x < knots[i]:
                j += 1
            if points[j].x >= knots[i + k + 1]:
                return False

        return True

    def spec_dimensional_minimization(self, spline, points, sw):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        knots = spline.get_knots()
        alpha_max = math.inf
        a = spline.get_left_bound()
        b = spline.get_right_bound()

        if self.m_dir[0] < 0:
            alpha_max = (a - knots[k + 1]) / self.m_dir[0]
        for i in range(g - 1):
            if self.m_dir[i] > self.m_dir[i + 1]:
                alpha_max = min(alpha_max, (knots[i + k + 2] - knots[i + k + 1]) / (self.m_dir[i] - self.m_dir[i + 1]))
        if self.m_dir[g - 1] > 0:
            alpha_max = min(alpha_max, (b - knots[g + k]) / self.m_dir[g - 1])

        theta0 = self.m_error
        theta0_der = 0
        for i in range(len(self.m_dir)):
            theta0_der += self.m_dir[i] * self.m_error_deriv[i]

        alpha0 = 0
        alpha2 = alpha_max / (1 - theta0 / alpha_max / theta0_der)
        alpha1 = 0.5 * alpha2
        Q0 = self.m_delta
        R0 = self.m_penalty
        theta1 = self.theta(spline, points, sw, alpha1)
        if theta1 < 0:
            return False
        Q1 = self.m_delta
        R1 = self.m_penalty

        iter = 0
        max_num_of_iter = 10
        while theta1 >= theta0 and iter < max_num_of_iter:
            alpha_tilde = -0.5 * theta0_der * alpha1 * alpha1 / (theta1 - theta0 - theta0_der * alpha1)
            alpha1 = max(0.1 * alpha1, alpha_tilde)
            theta1 = self.theta(spline, points, sw, alpha1)
            if theta1 < 0:
                return False
            Q1 = self.m_delta
            R1 = self.m_penalty
            iter += 1

        if iter > 0:
            if theta1 > theta0:
                self.theta(spline, points, sw, alpha0)
                # should we return false in the case of if?
            return True

        theta2 = self.theta(spline, points, sw, alpha2)
        if theta2 < 0:
            return False
        Q2 = self.m_delta
        R2 = self.m_penalty

        while theta2 < theta1:
            alpha0 = alpha1
            Q0 = Q1
            R0 = R1
            alpha1 = alpha2
            theta1 = theta2
            Q1 = Q2
            R1 = R2

            alpha2 = min(2 * alpha1, 0.5 * (alpha_max + alpha1))
            theta2 = self.theta(spline, points, sw, alpha2)
            if theta2 < 0:
                return False

            Q2 = self.m_delta
            R2 = self.m_penalty

        # find Q coefficients
        a0 = Q0
        diff1 = alpha1 - alpha0
        diff2 = alpha2 - alpha0
        a2 = (Q1 - Q0) / diff1
        a2 -= (Q2 - Q0) / diff2
        a2 /= alpha1 - alpha2
        a1 = (Q1 - a0) / diff1
        a1 -= a2 * diff1

        # find R coefficients
        fraction = diff1 / diff2
        numerator = R1 - R0 - fraction * (R2 - R0)
        temp = math.log((alpha_max - alpha1) / (alpha_max - alpha0))
        denominator = temp - fraction * math.log((alpha_max - alpha2) / (alpha_max - alpha0))
        b2 = numerator / denominator
        b1 = (R1 - R0 - b2 * temp) / diff1

        # find coefficients of quadratic equation
        a = -2 * a2
        b = -a * (alpha_max + alpha0) - self.p * b1 - a1
        c = (self.p * b1 + a1 + a * alpha0) * alpha_max - self.p * b2

        root1 = -0.5 * (b + math.sqrt(b * b - 4 * a * c)) / a
        root2 = -b / a - root1

        alpha_res = 0
        if 0 < root1 < alpha_max:
            alpha_res = root1
        elif 0 < root2 < alpha_max:
            alpha_res = root2

        theta_res = self.theta(spline, points, sw, alpha_res)

        if theta_res < 0:
            return False
        return True

    @staticmethod
    def initiate_grid(spline, points):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        knots = [0] * (g + 2)
        knots[0] = spline.get_left_bound()
        knots[g + 1] = spline.get_right_bound()
        n = len(points)

        unique_size = 0
        index = 0

        while index < n and points[index].x < knots[g + 1]:
            if index != 0 and points[index].x != points[index - 1].x:
                unique_size += 1
            index += 1

        # not enough data points
        if unique_size <= 0:
            return False

        # number of knots should be less than n - k for n points with unique x
        if unique_size < g + k + 1:
            return False

        points_per_knot = unique_size / (g + 1)
        knot_index = 1
        i = 1
        counter = 0

        while knot_index < g + 1:
            while counter < knot_index * points_per_knot or points[i].x == points[i - 1].x:
                if points[i].x != points[i - 1].x:
                    counter += 1
                i += 1
            knots[knot_index] = 0.5 * (points[i].x + points[i - 1].x)
            knot_index += 1

        spline.set_knots(knots)
        return True

    def count_norm_factors(self, points):
        # todo: should we think that points.x is sorted? what if all weights are equal?
        self.m_gamma = max(points.x) - min(points.x)
        self.m_beta = max(points.y) - min(points.y)
        self.m_mu = max(points.w) - min(points.w)

    def approximate_with_optimal_grid(self, spline, points, smooth_weight, eps1, eps2):
        if not self.initiate_grid(spline, points):
            return False

        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        #self.count_norm_factors(points)
        #smooth_weight *= 1e-10 * math.pow(self.m_gamma, 2 * k) * self.m_mu * self.m_mu

        if not self.approximate(spline, points, smooth_weight):
            return False

        self.m_dir = [0] * g
        self.m_error_deriv = [0] * g
        self.m_fixed_knots = [0] * (g + 2 * k + 2)

        self.delta(spline, points, smooth_weight)
        self.p = eps1 * self.m_delta * (spline.get_right_bound() - spline.get_left_bound()) / (g + 1) / (g + 1)
        self.m_error = self.m_delta + self.p * self.penalty(spline)

        for i in range(g):
            self.m_error_deriv[i] = self.error_derivative(spline, points, smooth_weight, i + k + 1)
            self.m_dir[i] = -self.m_error_deriv[i]

        old_norm = self.norm(self.m_dir)
        crit1 = eps1 + eps2
        crit2 = crit1
        max_num_of_iter = 1000

        counter = 0
        j = 0
        eps2_sq = eps2 * eps2

        while (crit1 >= eps1 or crit2 >= eps2_sq) and j < max_num_of_iter:
            self.m_fixed_knots = spline.get_knots().copy()

            old_error = self.m_error

            # can't minimize further, knots are too close to each other
            if not self.spec_dimensional_minimization(spline, points, smooth_weight):
                spline.set_knots(self.m_fixed_knots[k:(g + k + 2)])
                return self.approximate(spline, points, smooth_weight)

            for i in range(g):
                self.m_error_deriv[i] = self.error_derivative(spline, points, smooth_weight, i + k + 1)

            new_norm = self.norm(self.m_error_deriv)

            if j % g == 0:
                for i in range(g):
                    self.m_dir[i] = -self.m_error_deriv[i]
            else:
                temp = new_norm / old_norm
                for i in range(g):
                    self.m_dir[i] *= temp
                    self.m_dir[i] -= self.m_error_deriv[i]

            numerator = 0
            denominator = 0

            knots = spline.get_knots()
            for i in range(k + 1, g + k + 1):
                temp = knots[i] - self.m_fixed_knots[i]
                numerator += temp * temp
                denominator += self.m_fixed_knots[i] * self.m_fixed_knots[i]

            crit1 = math.fabs(old_error - self.m_error) / old_error
            crit2 = numerator / denominator

            old_norm = new_norm

        return True


def main():
    x = np.zeros(60)
    for i in range(len(x)):
        x[i] = i + 1 / (i + 1) * np.random.rand()
    x = np.concatenate((x, x), 0)
    x = np.sort(x)

    y = [0] * len(x)
    w = [1] * len(x)
    for i in range(0, len(x), 2):
        y[i] = np.cos(0.005 * x[i] * x[i])
        err = np.random.rand() * 15 / (x[i] + 10)
        y[i + 1] = y[i] + err
        y[i] -= err
        w[i] = 1.0 / math.fabs(x[i] - x[i - 1])
        w[i + 1] = w[i]

    knots = [x[0], 2, 6, 7, 10, 15, x[len(x) - 1]]

    points = pnt.Points(x, y, w)

    # calculate new x's and y's
    x_curve = np.linspace(x[0], x[-1], 1000)
    y_curve = [None] * 1000
    y_curve_opt = [None] * 1000

    coefficients = [None] * (len(x) + 2)
    s = splpckg.Spline(coefficients, knots)
    c = CurveFitter(s)
    c.initiate_grid(s, points)
    c.approximate(s, points, 0)

    index = 0
    for point in x_curve:
        y_curve[index] = s.get_value(point)
        index += 1

    index = 0
    knots = s.get_knots()
    knots_y_smoothed = [0] * len(knots)
    for point in knots:
        knots_y_smoothed[index] = s.get_value(point)
        index += 1

    c.approximate_with_optimal_grid(s, points, 0, 1e-2, 1e-4)

    index = 0
    for point in x_curve:
        y_curve_opt[index] = s.get_value(point)
        index += 1

    index = 0
    knots2 = s.get_knots()
    knots_y = [0] * len(knots2)
    for point in knots2:
        knots_y[index] = s.get_value(point)
        index += 1

    y_real = np.cos([0.001 * i * i for i in x_curve]) + 1

    plt.plot(x_curve, y_curve, x_curve, y_curve_opt, x, y, 'o', knots2, knots_y, 's', knots, knots_y_smoothed, 's')
    plt.xlim([x[0] - 1, x[-1] + 1])
    plt.legend(["Spline on uniform grid", "Spline on optimal grid", "Data"])
    plt.show()

main()