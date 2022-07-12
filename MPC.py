import numpy as np
import cvxopt
import matplotlib.pyplot as plt
class LTI_system:
    A = np.array([[1, 1],
                [0, 1]])
    B = np.array( [[0],
                [1]])
    P = np.array([[1, 0],
                [0, 1]])
    P = P/2
    Q = P
    R = np.array([5])
    x = np.zeros([1000,2])
    u = np.zeros(1000)
    x[0] = np.array([-4.5 ,2])
    def systemgo(self,xk,uk):
        return self.A@xk + self.B*uk

def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    n_var = H.shape[1]
    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')
    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])
        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])
        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')
    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')
    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)
    return np.array(sol['x'])
class MPC:
    def Prediction(x,E,H,N):
        lb1 =np.array([[-0.5],
                  [-0.5],
                  [-0.5]])
        ub1 =np.array([[0.5],
                  [0.5],
                  [0.5]])
        return quadprog(H, E@x, L=None, k=None, Aeq=None, beq=None, lb=lb1, ub=ub1)
    def get_GEH(A,B,Q,R,N):
        nA = A.shape[0]
        M = np.eye(nA)
        C = B
        b_zero= np.array([[0],
                        [0]])
        a = M
        for i in range(N):
            a = a@A
            M = np.vstack((M,a))
        for i in range(N):
            crow = b_zero
            for j in range(N):
                b = B
                if j <= i:
                    for k in range (i-j):
                        b = A@b
                    crow = np.hstack((crow,b))
                else :
                    crow = np.hstack((crow,b_zero))
            if i != 0:
                C = np.vstack((C,crow))
            else :
                C = crow
        C = C[:,1:]
        C = np.vstack((np.zeros([nA,N]),C))
        Q_ = np.kron(np.eye(N+1),Q)
        G = M.T@Q_@M
        E = C.T@Q_@M
        H = C.T@Q_@C+np.kron(np.eye(N),R)
        return G,E,H

if __name__ == '__main__':
    N = 3 #设置预测步数
    xx=[]
    yy=[]
    x4=[]
    y4=[]
    x2=[]
    y2=[]
    [G,E,H] = MPC.get_GEH(LTI_system.A,LTI_system.B,LTI_system.Q,LTI_system.R,N)
    for i in range(30):
        U = MPC.Prediction(LTI_system.x[i],E,H,N)
        uu = U[0]
        print(uu)
        LTI_system.x[i+1] = LTI_system.A@LTI_system.x[i]  + LTI_system.B@uu
        plt.scatter(LTI_system.x[i][0],LTI_system.x[i][1],color='black')

        xx.append(LTI_system.x[i][0])
        yy.append(LTI_system.x[i][1])
        # plt.plot(LTI_system.x[i][0],LTI_system.x[i][1])
    plt.plot(xx,yy)
    ####################################################################################

    # LTI_system.x[0] = np.array([-4.5 ,3])#设置初始状态
    # [G,E,H] = MPC.get_GEH(LTI_system.A,LTI_system.B,LTI_system.Q,LTI_system.R,N)
    # for i in range(50):
    #     U = MPC.Prediction(LTI_system.x[i],E,H,N)
    #     uu = U[0]
    #     print(uu)
    #     LTI_system.x[i+1] = LTI_system.A@LTI_system.x[i]  + LTI_system.B@uu
    #     plt.scatter(LTI_system.x[i][0],LTI_system.x[i][1],color='red')
    #     plt.plot(LTI_system.x[i][0],LTI_system.x[i][1])
    ###################################################################################
    N = 4 #设置预测步数
    [G,E,H] = MPC.get_GEH(LTI_system.A,LTI_system.B,LTI_system.Q,LTI_system.R,N)
    for i in range(30):
        U = MPC.Prediction(LTI_system.x[i],E,H,N)
        uu = U[0]
        print(uu)
        LTI_system.x[i+1] = LTI_system.A@LTI_system.x[i]  + LTI_system.B@uu
        plt.scatter(LTI_system.x[i][0],LTI_system.x[i][1],color='red')
        x4.append(LTI_system.x[i][0])
        y4.append(LTI_system.x[i][1])

    N = 2 #设置预测步数
    [G,E,H] = MPC.get_GEH(LTI_system.A,LTI_system.B,LTI_system.Q,LTI_system.R,N)
    for i in range(30):
        U = MPC.Prediction(LTI_system.x[i],E,H,N)
        uu = U[0]
        print(uu)
        LTI_system.x[i+1] = LTI_system.A@LTI_system.x[i]  + LTI_system.B@uu
        plt.scatter(LTI_system.x[i][0],LTI_system.x[i][1],color='blue',label = 'i = {}')
        x2.append(LTI_system.x[i][0])
        y2.append(LTI_system.x[i][1])
    plt.plot(x2,y2,label='N=2')
    plt.plot(x4,y4,label='N=4')
    plt.legend()
################################################################################################
    plt.show()
