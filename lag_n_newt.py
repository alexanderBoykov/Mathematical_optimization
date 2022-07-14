from math import tan, cos, log
import numpy as np
import matplotlib.pyplot as plt

def ctg(x):
    return 1 / tan(x)

def function(x):
    return x**2*log(x+3)

def difference(some_list, number):
    new_list = []
    for i in range(0, len(some_list)):
        new_list.append(some_list[i] - number)
    return new_list


def dividing(some_list, other_list):
    new_list = []
    for i in range(0, len(some_list)):
        new_list.append(some_list[i]/other_list[i])
    return new_list


def transpose(A):
    res = np.copy(A)
    for i in range(len(A)):
        for j in range(len(A)):
            res[i][j] = A[j][i]

    return res

def mul_matrix(A, B):
    res = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            for z in range(len(A)):
                res[i][j] += A[i][z] * B[z][j]
    return res


def mul_vector(A, b):
    res = np.zeros((len(A), 1))
    for i in range(len(A)):
        for j in range(len(A)):
            res[i] += A[i][j] * b[j]
    return res

def mul_v_v(a, b):
    res = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(a)):
            res[i][j] += a[i] * b[j]
    return res

def diag(A):
    check_list = []

    for i in range(len(A)):
        check_list.append(2 * abs(A[i][i]) - sum(abs(A[i][:])) > 0)

    return check_list == [True for i in range(len(A))]

def back_step(R, y, inv=True):
    if inv:
        dg = [R[-i][-i] for i in range(1, len(R) + 1)]
        x = [y[2][0] / dg[0]]

        for i in range(1, len(R)):
            sm = sum(x * R[len(R) - (i + 1), len(R) - i:][::-1])
            x.append((y[len(R) - (i + 1)][0] - sm) / dg[i])
    else:
        dg = [R[i][i] for i in range(0, len(R))]
        x = [y[0][0] / dg[0]]

        for i in range(1, len(R)):
            sm = sum(x * R[i][:i])
            x.append((y[i][0] - sm) / dg[i])

    return np.array(x)

def deviation_newton(nodes, test_points):
     deviation_table = []
     for i in range(0, len(test_points[0])):
         x = test_points[0][i]
         f = test_points[1][i]
         p = interpolate_Lagrange(x,nodes)
         deviation_table.append(abs(f-p))
     return max(deviation_table)+0.1

def mul_1(a,b):
    x=np.zeros(len(a))
    for i in range(len(a)):
        x[i]=a[i]*b[i]
    return x

def tru_slv(A,b):
    x = np.linalg.solve(A, b)

    return x

def LU_P(A, b):

        dim = len(A)
        M = np.copy(A)
        P = np.eye(dim)

        for ind in range(dim - 1):
            max_el = -np.inf
            max_ind = -1

            for i in range(ind, dim):
                if np.abs(M[i][ind]) > max_el:
                    max_el = np.abs(M[i][ind])
                    max_ind = i

            t_M, t_P = np.copy(M[ind][:]), np.copy(P[ind][:])
            M[ind][:], P[ind][:] = M[max_ind][:], P[max_ind][:]
            M[max_ind][:], P[max_ind][:] = t_M, t_P

            for j in range(ind + 1, dim):
                M[j][ind] = M[j][ind] / M[ind][ind]
                for k in range(ind + 1, dim):
                    M[j][k] = M[j][k] - M[j][ind] * M[ind][k]

        L_U = M + np.eye(dim)
        L = np.eye(dim) + np.tril(L_U, k=-1)
        U = L_U - L

        c = mul_vector(P, b)
        x = back_step(L, c, inv=False)
        x = x.reshape(-1, 1)
        x = back_step(U, x, inv=True)
        return np.flip(x)
def interpolate_Lagrange(x,nodes):
    P = 0
    def basis(k):
        basis = 1.0
        for i in range(0, len(nodes[0])):
            if i!=k:
                basis *= (x - nodes[0][i]) / (nodes[0][k] - nodes[0][i])
        return basis
    for i in range(0, len(nodes[0])):
        P += basis(i)*nodes[1][i]
    return P


def deviation(nodes, test_points):
    deviation_table = []
    for i in range(0, len(test_points[0])):
        x = test_points[0][i]
        f = test_points[1][i]
        p = interpolate_Lagrange(x, nodes)
        deviation_table.append(abs(f - p))
    return max(deviation_table)


test_points = [[], []]
step = 0.01
x = 1
for i in range(1, 1000):
    test_points[0].append(x)
    test_points[1].append(function(x))
    x += step



a = 1
b = 10
n = 100 #количество узлов!
nodes = [[],[]]
step = (b-a)/n
x = a
while x<b:
    nodes[0].append(x)
    nodes[1].append(function(x))
    x += step


opt_nodes = [[],[]]
for i in range(0,n):
    x = 0.5*((b-a)*cos((2*i+1)/(2*(n+1))*3.14)+(b+a))
    opt_nodes[0].append(x)
    opt_nodes[1].append(function(x))


poly_table = []
poly_table_opt = []
for i in range(0, len(nodes[0])):
    x = nodes[0][i]
    poly_table.append(interpolate_Lagrange(x, nodes))

for i in range(0, len(opt_nodes[0])):
    x = opt_nodes[0][i]
    poly_table_opt.append(interpolate_Lagrange(x,opt_nodes))

plt.plot(test_points[0], test_points[1], 'r.')
plt.plot(nodes[0], poly_table, 'b')
plt.show()

plt.plot(test_points[0], test_points[1], 'r.')
plt.plot(opt_nodes[0], poly_table_opt, 'b')
plt.show()





print('Newt')

def div_diff(nodes):
    x_nodes = nodes[0]
    y_nodes= nodes[1]
    n = len(nodes[1])
    a = []
    for i in range(0, n):
        a.append(nodes[1][i])

    for j in range(1, n):

        for i in range(n-1, j-1, -1):
            a[i] = float(a[i] - a[i-1])/float(x_nodes[i] - x_nodes[i-j])
    return a


def interpolate_Newton(x, nodes):
    d = div_diff(nodes)
    n = len(d) - 1
    P = d[n] + (x - nodes[0][n])
    for i in range( n - 1, -1, -1 ):
        P = P * ( x - nodes[0][i] ) + d[i]
    return P

poly_table_Newton = []
poly_table_Newton_opt = []
for i in range(0, len(nodes[0])):
    x = nodes[0][i]
    poly_table_Newton.append(interpolate_Newton(x, nodes))
for i in range(0, len(opt_nodes[0])):
    x = opt_nodes[0][i]
    poly_table_Newton_opt.append(interpolate_Newton(x,opt_nodes))


plt.plot(test_points[0], test_points[1], 'r.')
plt.plot(nodes[0], poly_table_Newton, 'b')
plt.show()
plt.plot(test_points[0], test_points[1], 'r.')
plt.plot(opt_nodes[0], poly_table_Newton_opt, 'b')
plt.show()

