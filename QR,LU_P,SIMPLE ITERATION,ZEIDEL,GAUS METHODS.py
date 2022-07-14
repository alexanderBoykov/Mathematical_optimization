import numpy as np
import math

def ex_0():
    A = np.array([[0., 2., 3.],
                  [1., 2., 4.],
                  [4., 5., 6.]])
    b = np.array([[13.],
                  [17.],
                  [32.]])



    return A, b


def ex_1():
    A = np.array([[11., 1., 1.],
                  [1., 13., 1.],
                  [1., 1., 15.]])
    b = np.array([[13.],
                  [15.],
                  [17.]])

    return A, b
def str_int(a,b):
    for i in range(len(a)):
        a[i]/=b
    return a

def ex_2():
    A = np.array([[-11., 1., 1.],
                  [1., -13., 1.],
                  [4., 5., -15.]])
    b = np.array([[-13.],
                  [-15.],
                  [-17.]])

    return A, b


def ex_3():
    A = np.array([[-11., 12., 13.],
                  [14., -13., 10.],
                  [13., 14., -15.]])
    b = np.array([[13.],
                  [15.],
                  [17.]])

    return A, b


def ex_4():
    A = np.array([[11., 10., 10.],
                  [10., 13., 10.],
                  [10., 10., 15.]])
    b = np.array([[13.],
                  [15.],
                  [17.]])

    return A, b


def ex_5(dim=4, eps=10e-3):
    A = (-1) * np.triu(np.ones(dim), k=1) + np.eye(dim) + \
        eps * 9 * ((-1) * np.triu(np.ones(dim), k=1) + np.tril(np.ones(dim)))
    b = [1.0 for i in range(dim - 1)]
    b.append(-1)
    b = np.array(b).reshape(-1, 1)

    return A, b


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
def mul_1(a,b):
    x=np.zeros(len(a))
    for i in range(len(a)):
        x[i]=a[i]*b[i]
    return x

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


def zeidel():
    print('Zeidels Method: ')
    print()

    ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]
    eps = 10e-3

    for z in range(6):
        A, b = ex_list[z]

        if not diag(A):
            A_t = transpose(A)
            A = mul_matrix(A_t, A)
            b = mul_vector(A_t, b)

        dim = len(A)
        C = np.zeros((dim, dim))
        d = np.zeros((dim, 1))
        it = 0

        for i in range(dim):
            if A[i][i] != 0:
                for j in range(dim):
                    if i != j:
                        C[i][j] = (-1) * A[i][j] / A[i][i]
                    else:
                        C[i][j] = 0
                        d[i] = b[i] / A[i][j]

        x0 = np.copy(d)

        while True:
            x = np.zeros((dim, 1))
            Ax = np.zeros((dim, 1))

            for i in range(dim):
                x[i] += d[i]
                for j in range(dim):
                    if j < i:
                        x[i] += C[i][j] * x[j]
                    if j > i:
                        x[i] += C[i][j] * x0[j]

            for i in range(dim):
                Ax[i] -= b[i]
                for j in range(dim):
                    Ax[i] += A[i][j] * x[j]

            if np.max(np.abs(Ax)) <= eps:
                print('Example {}'.format(z), 'converges to: ', x.reshape(1, -1))
                print('Error: ', np.max([np.abs(true_solve(z) - x.reshape(1, -1))]))
                print('Eps: ', eps)
                print('Iter: ', it)
                print()
                break
            if it == 1000:
                print('Example {}'.format(z), 'not converge.')
                print()
                # print(x.reshape(1, -1))
                break

            x0 = np.copy(x)
            it += 1
def jakobi():

        print('Jakobi Method: ')
        print()
        ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]
        eps = 10e-3

        for z in range(6):
            A, b = ex_list[z]

            if not diag(A):
                print('Example {}'.format(z), 'not converge.')
                print()
                continue

            dim = len(A)
            C = np.zeros((dim, dim))
            d = np.zeros((dim, 1))
            it = 0

            for i in range(dim):
                if A[i][i] != 0:
                    for j in range(dim):
                        if i != j:
                            C[i][j] = (-1) * A[i][j] / A[i][i]
                        else:
                            C[i][j] = 0
                            d[i] = b[i] / A[i][j]

            x0 = np.copy(d)

            while True:
                x = np.zeros((dim, 1))
                Ax = np.zeros((dim, 1))

                x=mul_vector(C,x0)+d
                for i in range(dim):
                    Ax[i] -= b[i]
                    for j in range(dim):
                        Ax[i] += A[i][j] * x[j]


                if np.max(np.abs(Ax)) <= eps:
                    print('Example {}'.format(z), 'converges to: ', x.reshape(1, -1))
                    print('Error: ', np.max([np.abs(true_solve(z) - x.reshape(1, -1))]))
                    print('Eps: ', eps)
                    print('Iter: ', it)
                    print()
                    break
                if it == 1000:
                    print('Example {}'.format(z), 'not converge.')
                    print()
                    # print(x.reshape(1, -1))
                    break

                x0 = np.copy(x)
                it += 1


def simple_dimple():
    print('Method simple iterations: ')
    print()
    ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]
    eps = 10e-3

    for z in range(4,5):
        A, b = ex_list[z]

        A_t = transpose(A)
        A = mul_matrix(A_t, A)
        b = mul_vector(A_t, b)

        dim = len(A)
        mu = 1 / (np.max(A)+eps)
        B = np.eye(dim) - mu * A
        c = mu * b
        it = 0
        x0 = np.copy(c)
        print(A,b)

        while True:
            x = np.zeros((dim, 1))
            Ax = np.zeros((dim, 1))

            for i in range(dim):
                x[i] += c[i]
                for j in range(dim):
                    x[i] += B[i][j] * x0[j]

            for i in range(dim):
                Ax[i] -= b[i]
                for j in range(dim):
                    Ax[i] += A[i][j] * x[j]

            if np.max(np.abs(Ax)) <= eps:
                print('Example {}'.format(z), 'converges to: ', x.reshape(1, -1))
                print('Error: ', np.max([np.abs(true_solve(z) - x.reshape(1, -1))]))
                print('Eps: ', eps)
                print('Iter: ', it)
                print()
                break
            if it == 2000:
                print('Example {}'.format(z), 'not converge.')
                print()
                # print(x.reshape(1, -1))

                break

            x0 = np.copy(x)
            it += 1


def QR():
    print('QR Method: ')
    print()
    ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]


    for zz in range(6):
        eps = 10e-6
        A, b = ex_list[zz]

        dim = len(A)
        Q = np.eye(dim)
        R = np.copy(A)

        Q_list = [Q]
        R_list = [R]

        it = 0

        while True:


            y, z = ((R_list[it])[it:, it:])[:, 0], np.eye(dim - it)[:][0]
            alpha = np.sqrt(sum([y[i] ** 2 for i in range(len(y))]))
            ro = np.sqrt(sum([(y[i] - alpha * z[i]) ** 2 for i in range(len(y))]))
            w = (y - alpha * z) / (ro + eps)

            Q_list.append(np.eye(dim - it) - 2 * mul_v_v(w, w))
            R_list.append(mul_matrix(Q_list[it + 1], R_list[it][int(it>=1):, int(it>=1):]))

            if it == dim - 2:
                for i in range(len(Q_list)):
                    if len(Q_list[i]) != dim:
                        t = np.copy(Q_list[i])
                        Q_list[i] = np.eye(dim)
                        Q_list[i][-len(t):, -len(t):] = t

                R_res = R_list[1]

                for i in range(it):
                    R_res[i + 1:, i + 1:] = R_list[i + 2]

                Q_list = Q_list[::-1]
                Q_fin = Q_list[0]

                for i in range(1, len(Q_list)):
                    Q_fin = mul_matrix(Q_fin, Q_list[i])

                Q_fin = transpose(Q_fin)
                R_fin = R_res

                y = mul_vector(transpose(Q_fin), b)
                x = back_step(R_fin, y)
                print('Example {}'.format(zz), 'converges to: ', x[::-1])
                print('Error: ', np.max([np.abs(true_solve(zz) - x[::-1])]))



                break

            it += 1




def pop_it():
    print('Gaus Method: ')
    print()
    ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]
    for z in range(6):

        A, b = ex_list[z]
        dim = len(A)



        for k in range(dim):
            if A[k][k]==0.0 and k!=dim-1:
                c=A.copy()
                e=b.copy()
                b[k]=e[k+1]
                b[k+1]=e[k]
                A[k]=c[k+1]
                A[k+1]=c[k]

            b[k][0]/= A[k][k]
            A[k]/=A[k][k]


            for i in range(1,dim-k):
                b[k + i] -= b[k] * A[k + i][k]
                A[k + i] -= A[k] * A[k + i][k]



        x = np.zeros(dim)
        x[-1]=b[-1]/A[-1][-1]

        for i in range(2,dim+1):
            x[-i]=b[-i]-np.sum(mul_1(x,A[-i]))
       # print('Example {}'.format(z), 'converges to: ', x.reshape(1, -1))


        print( x.reshape(1, -1))










def LU_P():
    print('LUP Method: ')
    print()
    ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]

    for z in range(6):
        A, b = ex_list[z]
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
        print(x)





def true_solve(ind):
    ex_list = [ex_0(), ex_1(), ex_2(), ex_3(), ex_4(), ex_5()]
    A, b = ex_list[ind]
    x = np.linalg.solve(A, b).reshape(1, -1)

    return x
A = np.array([[ 2., 2.,3.],
                  [4.,3.,5.]])
simple_dimple()
