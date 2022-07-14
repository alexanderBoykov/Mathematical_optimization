import numpy
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve


def function(x):
    return x *  numpy.log(x + 1)


#Умножение матриц
def multiply(first, second):
    result = numpy.zeros((first.shape[0], second.shape[1]))
    for i in range(first.shape[0]):
        for j in range(second.shape[1]):
            for k in range(len(second)):
                result[i, j] += first[i, k] * second[k, j]
    return result
#Генерируем данные
def generate_data(function, a, b, N, n_repeats):
    x = numpy.linspace(a, b, N)
    x_notes = []
    y_notes = []
    for _ in range(n_repeats):
        x_notes.append(x)
        y_notes.append(function(x) + numpy.random.normal(0, 0.2, N))

    return numpy.array(x_notes).flatten(), numpy.array(y_notes).flatten()

def eq(n,coef, x):
    P = numpy.zeros(x.shape)
    for i in range(0, n + 1):
        P += coef[i] * x ** i
    return P






a, b = 1,3
x_notes, y_notes = generate_data(function, a, b, 50, 3)

www = plt.axes()
plt.scatter(x_notes, y_notes)
sq_m = 0
for i in range(1, 6):
    e = numpy.ones((x_notes.shape[0], 1))
    for j in range(1, i + 1):
        e = numpy.hstack((e, numpy.reshape(x_notes ** j, (-1, 1))))
    coef_k = solve(multiply(e.T, e), multiply(e.T, numpy.reshape(y_notes, (-1, 1))))
    sq_mistakes = sum(((eq(i, coef_k, x_notes) - y_notes)) ** 2)
    print(sq_mistakes)
    www.plot(numpy.linspace(a, b, 5), eq(i, coef_k, numpy.linspace(a, b, 5)))
plt.show()

'''
Синий - полином 1 степени

Оранжевый - полином 2 степени

Зеленый - полином 3 степени

Красный - полином 4 степени

Фиолетовый - полином 5 степени





'''