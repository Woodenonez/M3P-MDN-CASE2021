import numpy as np
from numpy import dot, linalg
from numpy.linalg import inv

class KalmanFilter():
    
    def __init__(self, state_space, P0, Q, R):
        super().__init__()
        self.ss = state_space # [X0, A, B, C, D]
        self.X = state_space[0]
        self.P = P0
        self.Q = Q
        self.R = R
        self.tK = 0 # discrete time step

        self.Xs = state_space[0]

    def one_step(self, U, Y):
        self.tK += 1
        self.predict(U)
        self.update(U, Y)
        self.Xs = np.concatenate((self.Xs, self.X), axis=1)
        return self.X

    def append_state(self, X):
        self.Xs = np.concatenate((self.Xs, X), axis=1)
        self.tK += 1

    def predict_more(self, T, evolve_P=True):
        for _ in range(T):
            self.predict(np.zeros((np.shape(self.ss[2])[1],1)))
            self.Xs = np.concatenate((self.Xs, self.X), axis=1)

    def predict(self, U, evolve_P=True):
        A = self.ss[1]
        B = self.ss[2]
        self.X = dot(A, self.X) + dot(B, U)
        if evolve_P:
            self.P = dot(A, dot(self.P, A.T)) + self.Q
        return self.X

    def update(self, U, Y):
        C = self.ss[3]
        D = self.ss[4]
        Yh = dot(C, self.X) + dot(D, U)
        S = self.R + dot(C, dot(self.P, C.T)) # innovation: covariance of Yh
        K = dot(self.P, dot(C.T, inv(S))) # Kalman gain
        self.X = self.X + dot(K, (Y-Yh))
        self.P = self.P - dot(K, dot(S, K.T))
        return (self.X,K,S,Yh)

def model_CV(X0, Ts=1):
    A = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])
    B = np.zeros((4,1))
    C = np.array([[1,0,0,0],[0,1,0,0]])
    D = np.zeros((2,1))
    return [X0, A, B, C, D]

def fill_diag(diag):
    M = np.zeros((len(diag),len(diag)))
    for i in range(len(diag)):
        M[i,i] = diag[i]
    return M


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torch.tensor as ts
    import mdn_base

    X0 = np.array([[0,0,0,0]]).transpose()
    model = model_CV(X0)

    P0 = fill_diag((1,1,1,1))
    Q = fill_diag((1,1,100,1))
    Q2 = np.eye(4)
    R = np.eye(2)
    KF  = KalmanFilter(model,P0,Q,R)
    KF2 = KalmanFilter(model,P0,Q2,R)

    Y = [(1,0),(2.3,0),(2.5,1),(2.5,2),(2.5,3),(3,3),(4,3)]
    # Y = [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)]
    U = np.array([[0]])

    fig,ax = plt.subplots()

    for i in range(len(Y)+2):
        # ax.cla()
        if i<len(Y):
            KF.one_step(U, np.array(Y[i]).reshape(2,1))
            KF2.one_step(U, np.array(Y[i]).reshape(2,1))
        else:
            KF.predict(U,evolve_P=1)
            KF2.predict(U,evolve_P=1)
            KF.append_state(KF.X)
            KF2.append_state(KF2.X)
        print([KF2.X[:2].reshape(-1)],[KF2.P[0,0],KF2.P[1,1]])
        mdn_base.draw_GauEllipse(ax, ts([KF.X[:2].reshape(-1)]), ts([[KF.P[0,0],KF.P[1,1]]]), fc='g', nsigma=1, extend=False)
        mdn_base.draw_GauEllipse(ax, ts([KF2.X[:2].reshape(-1)]), ts([[KF2.P[0,0],KF2.P[1,1]]]), fc='y', nsigma=1, extend=False)
        plt.plot(KF.Xs[0,:],  KF.Xs[1,:],  'bo-')
        plt.plot(KF2.Xs[0,:], KF2.Xs[1,:], 'go-')
        plt.plot(np.array(Y)[:,0], np.array(Y)[:,1], 'rx')
        plt.pause(0.9)
    # KF.predict_more(2,evolve_P=False)
    # KF2.predict_more(2,evolve_P=False)

    plt.show()

