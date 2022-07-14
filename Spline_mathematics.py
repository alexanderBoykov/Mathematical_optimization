import numpy as np
import matplotlib.pyplot as plt


from math import tan, cos,pi



def ctg(x):
    return 1 / tan(x)



def function(x):
    return cos(x)



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






def spline1(n,funpoint,testpoint):
    A=np.zeros((2*n-2,2*n-2))
    b=np.zeros((2*n-2,1))
    for i in range(0,n):
        A[i*2-1-1][i*2-1-1]=funpoint[0][i-1]
        A[i * 2 - 1 ][i * 2 - 1 - 1] = funpoint[0][i ]
        A[i * 2 - 1][i * 2 - 1 ] = 1
        A[i * 2 - 1-1][i * 2 - 1] = 1
        b[i * 2 - 1]=funpoint[1][i ]
    for i in range(0, n-1):
        b[i * 2 ] = funpoint[1][i]
    spl=np.linalg.solve(A,b)

    graf=[]
    for i in range(n):
        graf.append(spl[i*2-2]*funpoint[0][i]+spl[i*2-1])

    max=0
    chekkkor=[]
    for i in range(n):
        for j in range(n):
            if testpoint[0][i]<funpoint[0][j]:
                key=j
                break
        chekkkor.append(spl[key*2-2]*testpoint[0][i]+spl[key*2-1])
        if max<abs(spl[key*2-2]*testpoint[0][i]+spl[key*2-1]-testpoint[1][i]):

            max=abs(spl[key*2-2]*testpoint[0][i]+spl[key*2-1]-testpoint[1][i])
    print(max)
    plt.plot(testpoint[0],chekkkor,'b')
    plt.plot(testpoint[0],testpoint[1],'r.')
    plt.show()



def spline2(n,funpoint,testpoint):

        A = np.zeros((3 * (n - 1), 3 * (n - 1)))
        b = np.zeros((3 * (n - 1), 1))
        for i in range(0, n - 1):
            A[i * 3][i * 3] = funpoint[0][i] ** 2
            A[i * 3 + 1][i * 3] = funpoint[0][i + 1] ** 2
            A[i * 3 + 1][i * 3 + 1] = funpoint[0][i + 1]
            A[i * 3][i * 3 + 1] = funpoint[0][i]
            A[i * 3 + 1][i * 3 + 2] = 1
            A[i * 3][i * 3 + 2] = 1
            A[i * 3 + 2][i * 3] = funpoint[0][i + 1] * 2
            A[i * 3 + 2][i * 3 + 1] = 1
            b[i * 3] = funpoint[1][i]
            b[i * 3 + 1] = funpoint[1][1 + i]
            try:
                A[i * 3 + 2][i * 3 + 3] = -1
            except:
                break
            A[i * 3 + 2][i * 3 + 4] = -2 * funpoint[0][i + 1]

        spl = tru_slv(A, b)

        chekkkor = []
        max = 0
        for i in range(n):
            for j in range(n):

                if testpoint[0][i] < funpoint[0][j]:
                    key = j
                    break

            chekkkor.append(
                spl[key * 3 - 3] * testpoint[0][i] ** 2 + spl[key * 3 - 2] * testpoint[0][i] + spl[key * 3 - 1])
            if max < abs(
                    spl[key * 3 - 3] * testpoint[0][i] ** 2 + spl[key * 3 - 2] * testpoint[0][i] + spl[key * 3 - 1] -
                    testpoint[1][i]):
                max = abs(
                    spl[key * 3 - 3] * testpoint[0][i] ** 2 + spl[key * 3 - 2] * testpoint[0][i] + spl[key * 3 - 1] -
                    testpoint[1][i])

        plt.plot(testpoint[0], chekkkor, 'b')
        plt.plot(testpoint[0], testpoint[1], 'r')
        plt.plot(xlim=(30,100))
        plt.show()
        print('deviation =', max)



def spline3(n,funpoint,testpoint):
    A = np.zeros((n-2, n-2))
    gama=np.zeros((n-2,1))
    y_vtor=np.zeros((n-1,1))
    C=np.zeros((n-1,1))

    for i in range(0,n-2):
        A[i][i]=2*(funpoint[0][i+2]-funpoint[0][i])
        gama[i]=6*( (funpoint[1][i+2]-funpoint[1][i+1])/(funpoint[0][i+2]-funpoint[0][i+1])-(funpoint[1][i+1]-funpoint[1][i])/(funpoint[0][i+1]-funpoint[0][i]) )
        try:
            A[i+1][i] =(funpoint[0][i + 2] - funpoint[0][i+1])
        except:
            break
        A[i][i+1] = (funpoint[0][i + 2] - funpoint[0][i + 1])
    spl=LU_P(A,gama)
    spl=np.append(spl,0.0)
    spl=np.insert(spl,0,0.0)
    for i in range(n-1):
        y_vtor[i]=((funpoint[1][i+1]-funpoint[1][i])/(funpoint[0][i+1]-funpoint[0][i]))-spl[i+1]*((funpoint[0][i+1]-funpoint[0][i])/6)-spl[i]*((funpoint[0][i+1]-funpoint[0][i])/3)

    #for i in range(n-1):
    resi=[]
    chekkkor = []
    max = 0
    for i in range(n):
        for j in range(n-1):

            if testpoint[0][i] < funpoint[0][j]:
                key = j
                break

        chekkkor.append(  funpoint[1][key-1]+y_vtor[key-1][0]*(testpoint[0][i]-funpoint[0][key-1])+spl[key-1]*((testpoint[0][i]-funpoint[0][key-1])**2)/2+(spl[key]-y_vtor[key-1][0])*((testpoint[0][i]-funpoint[0][key-1])**3)/(6*(funpoint[0][key]-funpoint[0][key-1])))
        if max < abs( funpoint[1][key-1]+y_vtor[key-1]*(testpoint[0][i]-funpoint[0][key-1])+spl[key-1]*((testpoint[0][i]-funpoint[0][key-1])**2)/2+(spl[key]-y_vtor[key-1])*((testpoint[0][i]-funpoint[0][key-1])**3)/(6*(funpoint[0][key]-funpoint[0][key-1]))-testpoint[1][i]):
            max =  abs( funpoint[1][key-1]+y_vtor[key-1]*(testpoint[0][i]-funpoint[0][key-1])+spl[key-1]*((testpoint[0][i]-funpoint[0][key-1])**2)/2+(spl[key]-y_vtor[key-1])*((testpoint[0][i]-funpoint[0][key-1])**3)/(6*(funpoint[0][key]-funpoint[0][key-1]))-testpoint[1][i])
        resi.append( abs( funpoint[1][key-1]+y_vtor[key-1]*(testpoint[0][i]-funpoint[0][key-1])+spl[key-1]*((testpoint[0][i]-funpoint[0][key-1])**2)/2+(spl[key]-y_vtor[key-1])*((testpoint[0][i]-funpoint[0][key-1])**3)/(6*(funpoint[0][key]-funpoint[0][key-1]))-testpoint[1][i]))
    return resi


n=100
testpoint=[[], []]
x=2.26
for i in range(1,n+1):
    testpoint[0].append(x)
    testpoint[1].append(function(x))
    x += 0.02

a = 2.25
b = 4
opt_nodes = [[],[]]
for i in range(0,n):
    x = 0.5*((b-a)*cos((2*i+1)/(2*(n+1))*pi)+(b+a))
    opt_nodes[0].append(x)
    opt_nodes[1].append(function(x))



opt_nodes[1].reverse()
opt_nodes[0].reverse()

nodes = [[],[]]
step = (b-a)/n
x = a
while x<b:
    nodes[0].append(x)
    nodes[1].append(function(x))
    x += step

max_div = spline1(n,nodes,testpoint)

plt.show()








