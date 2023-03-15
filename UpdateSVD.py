import numpy as np
from numpy import diag
from numpy.linalg import eig


def UpdateSVD(W: np.ndarray):
    b, n = W.shape
    m = np.mean(W, axis=1, keepdims=True)
    JW = W - m
    JW = JW.T
    ss, P = eig(np.dot(JW.T, JW))
    sorted_indices = np.argsort(ss)
    ss = ss[sorted_indices]
    P = P[:, sorted_indices]
    zeroidx = (ss <= 1e-10)
    if zeroidx.sum() == 0:
        diag_ss = diag(1.0 / np.sqrt(ss))
        JWP = np.dot(JW, P)
        H_v = np.sqrt(n) * np.dot(P, np.dot(JWP, diag_ss).transpose())
    else:
        ss = ss[ss > 1e-10]
        diag_ss = diag(1.0 / np.sqrt(ss))
        Q = np.dot(np.dot(JW, P[:, ~zeroidx]), diag_ss)
        Q = my_MGS(Q, b)
        H_v = np.sqrt(n) * np.dot(P, Q.T)
    return H_v


def my_MGS(U, K):
    m, n = U.shape
    U = np.hstack((U, np.zeros(shape=(m, K - n))))
    for i in range(n, K):
        v = np.random.rand(m, 1)
        v = v - v.mean()
        for j in range(0, i):
            Uj = np.ascontiguousarray(U[:, j])
            v = v - (np.dot(Uj.T, v) * Uj).reshape((m, 1))
        norm_value = np.linalg.norm(v)
        v = v / norm_value
        U[:, i] = v.T
    return U
