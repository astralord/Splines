# -*- coding: utf-8 -*-
import math


class Spline:
    def __init__(self, coefficients, horizontal_knots, spline_degree=3):
        self.k = spline_degree
        self.k_fact = math.factorial(self.k)
        self.coefficients = coefficients
        horizontal_knots.sort()
        self.g = len(horizontal_knots) - 2
        self.a = horizontal_knots[0]
        self.b = horizontal_knots[self.g + 1]
        self.knots = [self.a] * (self.k + 1) + [0] * self.g + [self.b] * (self.k + 1)
        for i in range(self.g):
            self.knots[i + self.k + 1] = horizontal_knots[i + 1]

    def get_left_node_index(self, point, min_id=0):
        if point < self.a or point > self.b:
            return -1

        l = min_id
        while l < self.g + self.k and (self.knots[l] > point or self.knots[l + 1] <= point):
            l += 1
        return l

    # evaluate B-spline of degree deg on interval [λ_{knot_id}, λ_{knot_id+deg+1}) at given point
    def b_spline(self, point, deg, knot_id):
        if point < self.knots[knot_id] or point > self.knots[knot_id + deg + 1]:
            return 0

        if deg == 0:
            return point != self.knots[knot_id + deg + 1]

        # if there are k + 1 coincident points on the left side
        if self.knots[knot_id + deg] < self.knots[knot_id + deg + 1]:
            j = 0
            while j < deg and self.knots[knot_id + j] == self.knots[knot_id + j + 1]:
                j += 1
            if j == deg:
                return pow((self.knots[knot_id + deg + 1] - point) /
                           (self.knots[knot_id + deg + 1] - self.knots[knot_id]), deg)

        # if there are k + 1 coincident points on the right side
        if self.knots[knot_id] < self.knots[knot_id + 1]:
            j = 1
            while j <= deg and self.knots[knot_id + j] == self.knots[knot_id + j + 1]:
                j += 1
            if j == deg + 1:
                return pow((point - self.knots[knot_id]) /
                           (self.knots[knot_id + deg + 1] - self.knots[knot_id]), deg)

        l = self.get_left_node_index(point, knot_id)
        buff = [0] * (deg + 1)
        buff[knot_id - l + deg] = 1

        for j in range(1, deg + 1):
            for i in reversed(range(l - deg + j, l + 1)):
                alpha = (point - self.knots[i]) / (self.knots[i + 1 + deg - j] - self.knots[i])
                buff[i - l + deg] = alpha * buff[i - l + deg] + (1 - alpha) * buff[i - 1 - l + deg]

        return buff[deg]

    # evaluate all B-splines of degree deg < k on interval at given point
    def b_splines(self, point, deg):
        if deg > self.k:
            return False

        l = self.get_left_node_index(point)
        buff = [0] * (self.k + 1)
        buff[deg] = 1

        for r in range(1, deg + 1):
            v = l - r + 1
            w2 = (self.knots[v + r] - point) / (self.knots[v + r] - self.knots[v])
            buff[deg - r] = w2 * buff[deg - r + 1]
            for i in range(deg - r + 1, deg):
                w1 = w2
                v += 1
                w2 = (self.knots[v + r] - point) / (self.knots[v + r] - self.knots[v])
                buff[i] = (1 - w1) * buff[i] + w2 * buff[i + 1]
            buff[deg] *= (1 - w2)
        return buff

    def get_internal_knots_num(self):
        return self.g

    def get_degree(self):
        return self.k

    def get_left_bound(self):
        return self.a

    def get_right_bound(self):
        return self.b

    def get_degree_factorial(self):
        return self.k_fact

    # evaluate dD-th derivative of B-spline of degree l on interval [λ_i, λ_{i+l+1}) at given point
    def b_spline_derivative(self, point, l, i, der_degree=1):
        if der_degree == 0:
            return self.b_spline(point, l, i)

        if l == 0:
            return 0

        spline = 0
        c1 = self.knots[i + l] - self.knots[i]
        c2 = self.knots[i + l + 1] - self.knots[i + 1]
        if c1 != 0:
            spline += self.b_spline_derivative(point, l - 1, i, der_degree - 1) / c1
        if c2 != 0:
            spline -= self.b_spline_derivative(point, l - 1, i + 1, der_degree - 1) / c2
        return l * spline

    # get value of built spline at given point
    def get_value(self, point):
        if point < self.a or point > self.b:
            return 0  # todo: return continuation of spline

        l = self.get_left_node_index(point)
        if l < 0:
            return 0

        buff = [0] * (self.k + 1)
        # De Boor Algorithm
        for i in range(self.k + 1):
            buff[i] = self.coefficients[i + l - self.k]

        for j in range(1, self.k + 1):
            for i in reversed(range(l - self.k + j, l + 1)):
                alpha = (point - self.knots[i]) / (self.knots[i + 1 + self.k - j] - self.knots[i])
                buff[i - l + self.k] = alpha * buff[i - l + self.k] + (1 - alpha) * buff[i - 1 - l + self.k]

        return buff[self.k]

    # get value of derivative of built spline at point x
    def get_value_derivative(self, point, der_degree=1):
        if der_degree < 0:
            return None

        if der_degree == 0:
            return self.get_value(point)

        if der_degree > self.k:
            return 0

        l = self.get_left_node_index(point)

        # if point is out of [a, b]
        if l < 0:
            if der_degree > 1:
                return 0
            return -1  # todo: replace with derivatives outside of knots

        alpha = 1
        spline = 0

        for i in range(der_degree):
            alpha *= (self.k - i)

        buff = [0] * (self.k + 1)
        for i in range(self.k + 1):
            buff[i] = self.coefficients[i + l - self.k]

        for j in range(1, der_degree):
            for i in reversed(range(l - self.k + j, j + 1)):
                buff[i - l + self.k] = (buff[i - l + self.k] - buff[i - l + self.k - 1]) / (self.knots[i + 1 + self.k - j] - self.knots[i])

        for i in range(der_degree, self.k + 1):
            spline += buff[i] * self.b_spline(point, self.k - der_degree, l + i - self.k)

        return alpha * spline

    # get difference between leading derivatives of B-splines on interval [λ_{i}, λ_{i+k+1}] in λ_{q-} and λ_{q+}
    def get_lead_derivative_difference(self, i, q):
        if i < q - self.k - 1 or i > q:
            return 0
        numerator = (2 * (self.k % 2) - 1) * self.k_fact * (self.knots[i + self.k + 1] - self.knots[i])
        denominator = 1
        for j in range(i, i + self.k + 2):
            if j != q:
                denominator *= self.knots[q] - self.knots[j]

        return numerator / denominator

    # get derivative of difference between leading derivatives
    def get_lead_der_diff_der_knot(self, i, q, l):
        if l < i or l > i + self.k + 1:
            return 0
        return self.get_lead_der_diff_der_knot(self.get_lead_derivative_difference(i, q), i, q, l)

    # get derivative of difference between leading derivatives, if this difference is counted
    def get_lead_der_diff_der_knot(self, lddk, i, q, l):
        if l < i or l > i + self.k + 1:
            return 0

        if l != i and l != q and l != i + self.k + 1:
            return lddk / (self.knots[q] - self.knots[l])

        c = (2 * (self.k % 2) - 1) * self.k_fact
        product = c / lddk
        total_sum = 0

        if q != i and q != i + self.k + 1:
            if l == i:
                return lddk / (self.knots[q] - self.knots[i]) / (self.knots[i + self.k + 1] - self.knots[i]) * (self.knots[i + self.k + 1] - self.knots[q])
            if l == i + self.k + 1:
                return lddk / (self.knots[q] - self.knots[i + self.k + 1]) / (self.knots[i + self.k + 1] - self.knots[i]) * (self.knots[q] - self.knots[i])
            if q == l:
                product *= self.knots[i + self.k + 1] - self.knots[i]
                for j in range(i, i + self.k + 2):
                    if j != q:
                        total_sum += product / (self.knots[q] - self.knots[j])
                product *= product
                return -c * (self.knots[i + self.k + 1] - self.knots[i]) * total_sum / product
        else:
            for j in range(i + 1, i + self.k + 1):
                total_sum += product / (self.knots[q] - self.knots[j])
            return -c * total_sum / (product * product)
        return 0

    # get value of derivative dS/dλ
    def get_value_derivative_knot(self, point, knot_id):
        if point < self.a:
            if knot_id == self.k + 1:
                numerator = -self.k * (self.coefficients[1] - self.coefficients[0]) * (point - self.a)
                denominator = (self.knots[knot_id] - self.a) * (self.knots[knot_id] - self.a)
                return numerator / denominator
            return 0

        if point > self.b:
            if knot_id == self.g + self.k:
                numerator = -self.k * (self.coefficients[self.g + self.k] - self.coefficients[self.g + self.k - 1]) * (point - self.b)
                denominator = (self.knots[knot_id] - self.b) * (self.knots[knot_id] - self.b)
                return numerator / denominator
            return 0

        if point <= self.knots[knot_id - self.k] or point >= self.knots[knot_id + self.k]:
            return 0

        l = self.get_left_node_index(point)
        if l < 0:
            return 0

        if l >= knot_id:
            l += 1

        buff = [0] * (self.k + 1)
        # De Boor algorithm
        for i in range(self.k + 1):
            if i < knot_id - l or i > knot_id - l + self.k:
                buff[i] = 0
            else:
                buff[i] = self.coefficients[i + l - self.k - 1] - self.coefficients[i + l - self.k]
                if i + l + 1 <= knot_id:
                    buff[i] /= self.knots[i + l + 1] - self.knots[i + l - self.k]
                elif i <= knot_id:
                    buff[i] /= self.knots[i + l] - self.knots[i + l - self.k]
                else:
                    buff[i] /= self.knots[i + l] - self.knots[i + l - self.k - 1]

        for j in range(1, self.k + 1):
            for i in reversed(range(l - self.k + j, l + 1)):
                if i + 1 + self.k - j <= knot_id:
                    alpha = (point - self.knots[i]) / (self.knots[i + 1 + self.k - j] - self.knots[i])
                elif i <= knot_id:
                    alpha = (point - self.knots[i]) / (self.knots[i + self.k - j] - self.knots[i])
                else:
                    alpha = (point - self.knots[i - 1]) / (self.knots[i + self.k - j] - self.knots[i - 1])
                buff[i - l + self.k] = alpha * buff[i - l + self.k] + (1 - alpha) * buff[i - 1 - l + self.k]
        return buff[self.k]

    def insert_node(self, coordinate):
        j = self.get_left_node_index(coordinate)

        if j < 0 or self.knots[j] == coordinate:
            return

        self.knots.append(coordinate)
        self.knots.sort()
        self.coefficients.append(self.coefficients[self.g + self.k])

        for i in reversed(range(j + 1, self.g + self.k + 1)):
            self.coefficients[i] = self.coefficients[i - 1]

        for i in reversed(range(j - self.k + 1, j + 1)):
            ri = (coordinate - self.knots[i]) / (self.knots[i + self.k + 1] - self.knots[i])
            self.coefficients[i] = ri * self.coefficients[i] + (1 - ri) * self.coefficients[i - 1]

        self.g += 1

    def get_knots(self):
        return self.knots[self.k:self.g + self.k + 2]

    def set_knots(self, horizontal_knots):
        horizontal_knots.sort()
        self.g = len(horizontal_knots) - 2
        self.a = horizontal_knots[0]
        self.b = horizontal_knots[self.g + 1]
        self.knots = [self.a] * (self.k + 1) + [0] * self.g + [self.b] * (self.k + 1)
        for i in range(self.g):
            self.knots[i + self.k + 1] = horizontal_knots[i + 1]

    def get_coefficients(self):
        return self.coefficients

    def set_coefficients(self, coefficients):
        self.coefficients = coefficients

    def set_left_edge(self, left_edge):
        self.a = left_edge

    def set_right_edge(self, right_edge):
        self.b = right_edge
