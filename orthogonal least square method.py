import numpy
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve


def function(x):
    return x * numpy.log(x + 1)

def generate_data(function, a, b, N, n_repeats):
    x = numpy.linspace(a, b, N)
    x_notes = []
    y_notes = []
    for _ in range(n_repeats):
        x_notes.append(x)
        y_notes.append(function(x) + numpy.random.normal(0, 0.2, N))

    return numpy.array(x_notes).flatten(), numpy.array(y_notes).flatten()




def deg(points, deg):
    res = 0
    for point in points:
        res += point[0] ** deg
    return res


def alpha(index1, points, matrix):
    j = index1 - 1
    num = 0
    denom = 0

    for i in range(len(points)):
        t = matrix[j][i] ** 2
        num += points[i][0] * t
        denom += t

    return num / denom


def betta(index, points, matrix):
    num = 0
    denom = 0

    for i in range(len(points)):
        t = matrix[index - 1][i]
        num += points[i][0] * matrix[index][i] * t
        denom += t ** 2

    return num / denom


def q (j, x, points, q_matrix, for_matrix):
    if j == 0:
        return 1
    if j == 1:
        return x - (1 / len(points)) * deg(points, 1)

    return x * q_matrix[j - 1][for_matrix] - alpha(j, points, q_matrix) * q_matrix[j - 1][for_matrix] - betta(j - 1, points,
                                                                                                            q_matrix) * \
           q_matrix[j - 2][for_matrix]

def coef_a (k, points, q_matrix):
    num = 0
    denom = 0

    for i in range(len(points)):
        tmp = q_matrix[k][i]
        num += tmp * points[i][1]
        denom += tmp ** 2

    return num / denom

def ort_pol_value(x, n, points, q_matrix, alpha_list, betta_list):
    q_x_list = [1, x - (1 / len(points)) * deg(points, 1)]

    for i in range(2, n + 1):
      q_x_list.append(x * q_x_list[i - 1] - alpha_list[i] * q_x_list[i - 1] - betta_list[i - 1] * q_x_list[i - 2])

    a_list = []

    for i in range(n + 1):
        a_list.append(coef_a(i, points, q_matrix))

    res = 0

    for i in range(n + 1):
        res += a_list[i] * q_x_list[i]

    return res




def orthogonal(n, points):
    q_matrix = []
    alpha_list = []
    betta_list = []
    for i in range(n + 1):
        row = []
        for j in range(len(points)):
            row.append(q(i, points[j][0], points, q_matrix, j))
        q_matrix.append(row)

    for i in range(n + 1):
        alpha_list.append(alpha(i + 1, points, q_matrix))
        betta_list.append(betta(i, points, q_matrix))

    error = 0

    values = []

    for i in range(len(points)):
        x = points[i][0]
        val = ort_pol_value(x, n, points, q_matrix, alpha_list, betta_list)
        values.append(val)
        error += (val - points[i][1]) ** 2
    return values, error



a, b = 1, 3
x_notes, y_notes = generate_data(function, a, b, 50, 3)
points = numpy.array([x_notes, y_notes]).T
for i in range(1, 6):
    values, error = orthogonal(i, points)
    plt.plot(x_notes[:49], values[:49])
    print(error)
plt.scatter(x_notes, y_notes)
plt.show()

'''

Синий - полином 1 степени

Оранжевый - полином 2 степени

Зеленый - полином 3 степени

Красный - полином 4 степени

Фиолетовый - полином 5 степени

'''