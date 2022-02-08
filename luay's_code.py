from typing import Tuple
import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import eig, svd
import sympy
import operator as op
from functools import reduce
from timeit import default_timer as timer
from scipy.sparse import lil_matrix
from scipy.optimize import lsq_linear
import scipy
import matplotlib.pyplot as plt
# import cvxpy as cp
import statistics

littled = None
TOL = 1e-3


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


# qp = ncr(n + DD - 1, DD - 1)  # number of monomials up to order DD-1
# qD_d = ncr(n + DD - d, DD - d)

def sumOfAP(a: int, d: int, n: int) -> int:
    """
    Function to find sum of series.
    :param a: The first term of series
    :param d: common difference
    :param n: number of terms in the series
    :return: sum of series
    """
    sum = 0
    i = 0
    while i < n:
        sum = sum + a
        a = a + d
        i = i + 1
    return sum


def nullspace(A, atol: float = 1e-9, rtol=0):
    """
    :param A:
    :param atol:
    :param rtol:
    :return:
    """
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def BuildMat(P: np.array) -> np.array:
    """
    :param P: A 2x2 matrix that
    :return:
    """
    A = np.zeros((2 + P.shape[0], 15))
    A[0][0] = -1
    for i in range(P.shape[0]):
        A[0][i + 1] = 1
        A[i + 1][12] = -P[i][0] ** 2
        A[i + 1][14] = -P[i][1] ** 2
        A[i + 1][13] = -P[i][0] * P[i][1] * 2
        if i == 0:
            A[i + 1, 5] = 1
        else:
            A[i + 1, 9] = 1
    A[P.shape[0] + 1, [12, 14]] = 1
    A[P.shape[0] + 1][0] = -1
    return A


def NumberOfMon(n: int, d: int) -> int:
    """
    :param n: number of variables
    :param d: degree
    :return: number of monomials of total degree d in n variables
    """
    return ncr(n + d, n)


def FindDegree(r: int, vec: np.array) -> int:
    """
    :param r: integer that indicates the monomial location
    :param vec: vector that indicates the start location of each degree
    :return: degree of the monomial
    """
    u = vec
    u -= r
    return np.where(u < 0)[0]



def startOfMonomal(d: int, n: int, i: int) -> int:
    """
    :param d: degree of that monomonial
    :param n: number of terms
    :param i: monomnial number
    :return:  index of the start of monomonial i of degree d
    """
    return ncr(n + d - 1, d - 1) + sumOfAP((ncr(n + d - 1, d - 1) - ncr(n + d - 2, d - 2)), -1, i)  # maybe i-1 in last


def rangeOfDegree(d: int, n: int):
    """
    :param d: degree
    :param n: number of terms
    :return: range of monomonials of degree d
    """
    if d == 1:
        return 1, ncr(n + 1, 1) - 1
    else:
        return ncr(n + d - 1, d - 1), ncr(n + d, d) - 1


def constructMaculayMatrix(P: np.array, l=1.0) -> np.array:
    n = d = 2 + P.shape[0]  # d* , n.
    littled = ncr(n + 2, n)
    pD = ((n - 1) * ncr(n + d - 2, d - 2)) + ncr(n + d - 1, d - 1)  # sum of ncr(n+d-d_i,d-d_i) for i in (1,n)
    qD = ncr(n + d, d)
    M = lil_matrix((n, littled), dtype=np.float)
    # Building regular matrix
    for i in range(1, P.shape[0] + 1):  # pi = 1
        M[0, i] = 1
    M[0, 0] = -l
    a = n + 1
    x1 = n + 1 + sumOfAP(n, -1, n - 2)  # index of x1^2
    x2 = x1 + 2
    x12 = x1 + 1
    for i in range(1, P.shape[0] + 1):  # di^2 - (PiX)^2
        M[i, a] = 1
        M[i, x1] = -P[i - 1, 0] ** 2
        M[i, x2] = -P[i - 1, 1] ** 2
        M[i, x12] = -(2 * P[i - 1, 0] * P[i - 1, 1])
        a += (n - i + 1)
    M[P.shape[0] + 1, 0] = -1
    M[P.shape[0] + 1, x1] = 1
    M[P.shape[0] + 1, x2] = 1
    return M


def generateRandomShiftMatrix(n: np.int) -> np.ndarray:
    return np.hstack((0, np.random.normal(0, 1, size=(n,))))
    # m = n-3
    # eye = np.ones((n, ))
    # z = np.random.choice(np.arange(n), size=m, replace=False)
    # eye[z] = 0
    # return np.hstack((0, eye))


def shiftMatrixByShiftFunction(A: np.ndarray, sigma_function: np.array, starting_indices: np.ndarray):
    num_poly = A.shape[0]
    degree_per_row = np.zeros((num_poly,))
    A = np.asarray(A)
    for i in range(num_poly):
        max_non_zero_index = np.max(np.nonzero(A[i, :].flatten())[0])
        if max_non_zero_index == 0:
            degree_per_row[i] = -1
        else:
            degree_per_row[i] = getRangeOfidx(max_non_zero_index, starting_indices)[-3]

    degree_of_sigma_function = getRangeOfidx(np.max(np.nonzero(sigma_function)[0]), starting_indices)[-3]

    S2 = np.zeros(A.shape)

    for i in range(num_poly):
        for idx in np.nonzero(sigma_function)[0]:
            if idx == 0:
                S2[i, :A.shape[1]] += (A[i, :] * sigma_function[idx])
            else:
                for non_zero_idx in np.nonzero(A[i])[0]:
                    if non_zero_idx == 0:
                        S2[i, idx] += (A[i, non_zero_idx] * sigma_function[idx])
                    else:
                        S2[i, shiftIndex(np.array([non_zero_idx, idx], dtype=np.int), starting_indices)[-1]] += \
                            (A[i, non_zero_idx] * sigma_function[idx])

    return S2


def shift_matrix_moqaren(vh, crk, qp, qD, starting_indices):
    A = vh[-crk:]
    A0 = np.concatenate((A[:, :qp], np.zeros((crk, qD - qp))))
    shift_function = np.random.normal(0, 1, size=(5,))
    shift_function[0] = 0
    B = shiftMatrixByShiftFunction(A, shift_function, starting_indices)
    return A, A0, B


def findBreakingPoint(v: np.array):
    for i in range(v.shape[0] - 1):
        if v[i + 1] / v[i] <= 1e-4:
            return i + 1

    return None


def roundingMatrix(A):
    U, D, V = np.linalg.svd(A)
    breaking_point = findBreakingPoint(D)
    if breaking_point is None:
        return A
    else:
        D[breaking_point:] = 0
    # return np.where(np.abs(A) <= 1e-10, 0, A)
    return U.dot(np.diag(D).dot(V))

def find_li_vectors(dim, R):
    r = matrix_rank(R)
    index = np.zeros(r)  # this will save the positions of the li columns in the matrix
    counter = 0
    index[0] = 0  # without loss of generality we pick the first column as linearly independent
    j = 0  # therefore the second index is simply 0

    for i in range(R.shape[0]):  # loop over the columns
        if i != j:  # if the two columns are not the same
            inner_product = np.dot(R[:, i], R[:, j])  # compute the scalar product
            norm_i = np.linalg.norm(R[:, i])  # compute norms
            norm_j = np.linalg.norm(R[:, j])

            # inner product and the product of the norms are equal only if the two vectors are parallel
            # therefore we are looking for the ones which exhibit a difference which is bigger than a threshold
            if np.abs(inner_product - norm_j * norm_i) > 1e-4:
                counter += 1  # counter is incremented
                index[counter] = i  # index is saved
                j = i  # j is refreshed
            # do not forget to refresh j: otherwise you would compute only the vectors li with the first column!!

    R_independent = np.zeros((r, dim))

    i = 0
    # now save everything in a new matrix
    while (i < r):
        R_independent[i, :] = R[index[i], :]
        i += 1

    return R_independent


def gauss(A):
    m = len(A)
    n = m + 1

    for k in range(m):
        pivots = [abs(A[i][k]) for i in range(k, m)]
        i_max = pivots.index(max(pivots)) + k

        # Check for singular matrix
        assert A[i_max][k] != 0, "Matrix is singular!"

        # Swap rows
        A[k], A[i_max] = A[i_max], A[k]

        for i in range(k + 1, m):
            f = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= A[k][j] * f

            # Fill lower triangular matrix with zeros:
            A[i][k] = 0

    # Solve equation Ax=b for an upper triangular matrix A
    x = []
    for i in range(m - 1, -1, -1):
        x.insert(0, A[i][m] / A[i][i])
        for k in range(i - 1, -1, -1):
            A[k][m] -= A[k][i] * x[0]
    return x


def checkValidityAboveZero(v, idxs):
    for i in idxs:
        if v[i] < -1e-5:
            return False

    return True



def BTTRFinal(P: np.array, ineq=None, inequalities_on_variables=None, args=None):
    n = 4
    d = 4
    pD, qD = 80, 70
    DD = 4
    M, starting_indices = constructShiftMatrixMurad(P, args)
    Z = scipy.linalg.null_space(M.todense())
    norms = np.linalg.norm(Z, axis=1)
    idxs = np.where(norms >= statistics.median(norms))[0]
    reduced_form, inds = sympy.SparseMatrix(Z[idxs]).T.rref()
    ma = len(inds)
    # B = Z[idxs[list(inds)], :]
    # B = B[:ma, :]
    indices = [(i, idxs[inds[i]]) for i in range(ma)]
    S1 = np.zeros((ma, Z.shape[0]))
    for idx in indices:
        S1[idx] = 1
    B = S1.dot(Z)
    W = Z.dot(np.linalg.pinv(roundingMatrix(B)))
    shift_function = generateRandomShiftMatrix(
        starting_indices[args["d"] - getRangeOfidx(idxs[inds[-1]], starting_indices)[-3] - 2, -1])
    shift_function[np.where(shift_function != 0)[0]] = 1
    try:
        S2 = shiftMatrixByShiftFunction(S1, shift_function, starting_indices)
    except IndexError:
        return None
    A = S2.dot(W)
    A = A[:, :ma]
    U, Q = scipy.linalg.schur(A)
    solutions = []
    for i in range(Q.shape[1]):
        A = W.dot(Q[:, i])
        A = np.asarray(A).flatten()
        A = np.multiply(A, 1.0 / A[0])
        r = np.linalg.norm(P.dot(A[:args["littled"]]))
        # if r <= 1e-2 and np.all(A[inequalities_on_variables] >= -1e-19) and not(np.any(np.isnan(A))) :
        # if r <= 1e-2 and checkValidityAboveZero(A, inequalities_on_variables) and not(np.any(np.isnan(A))) :
        if True and checkValidityAboveZero(A, inequalities_on_variables) and not (np.any(np.isnan(A))):
            solutions.append(A)
    for i in range(Q.shape[0]):
        A = W.dot(Q[i, :])
        A = np.asarray(A).flatten()
        A = np.multiply(A, 1.0 / A[0])
        r = np.linalg.norm(P.dot(A[:args["littled"]]))
        # if r <= 1e-2 and np.all(A[inequalities_on_variables] >= -1e-19) and not(np.any(np.isnan(A))) :
        # if r <= 1e-2 and checkValidityAboveZero(A, inequalities_on_variables) and not(np.any(np.isnan(A))) :
        if True and checkValidityAboveZero(A, inequalities_on_variables) and not (np.any(np.isnan(A))):
            solutions.append(A)
    return None if len(solutions) == 0 else np.array(solutions)

def BinarySearchOnBTTR(P: np.array, inequalities_on_variables=None, ineq=None, args=None):
    low = 0.0
    oldwlow = low
    high = 100.0
    M1 = constructMaculayMatrix(P, high)
    for i in np.linspace(0.0, 100.0, 10):
        M1[0, 0] = -np.round(i)
        Mh = BTTRFinal(M1, inequalities_on_variables=inequalities_on_variables, args=args)
        if Mh is not None:
            high = np.round(i)
            break
    M1[0, 0] = -high
    Mh = BTTRFinal(M1, inequalities_on_variables=inequalities_on_variables, args=args)
    M1[0, 0] = -low
    Ml = BTTRFinal(M1, inequalities_on_variables=inequalities_on_variables, args=args)
    while high - low >= 1e-6:
        if Ml is not None:
            Mh = Ml
            high = low
            low = oldlow
            M1[0, 0] = -low
            Ml = BTTRFinal(M1, inequalities_on_variables=inequalities_on_variables, args=args)
        if Ml is None and Mh is None:
            Ml = Mh
            high -= (high - low) / 2
            M1[0, 0] = -high
            Mh = BTTRFinal(M1, inequalities_on_variables=inequalities_on_variables, args=args)
        if Ml is None and Mh is not None:
            oldlow = low
            low += ((high - low) / 2)
            M1[0, 0] = -low
            Ml = BTTRFinal(M1, inequalities_on_variables=inequalities_on_variables, args=args)
        # if Mh1 is None and Ml1 is None:
        #     low = high
        #     high *= 2
        # elif Mh1 is not None and Ml1 is None:
        #     oldlow = low
        #     low += ((high-low)/2)
        # elif Ml1 is not None:
        #     high = low
        #     if low > oldlow:
        #         low -= (low-oldlow)/2
    return Mh, high


def processSolutions(solutions):
    try:
        real_entries_idxs = np.where(np.any(np.logical_not(np.iscomplex(solutions))), axis=1)[0][0]
    except IndexError:  # it means all values are non-complex
        return None
    return real_entries_idxs


def constructIndicesMatrix(n, d):
    starting_indices = np.zeros((d, n))
    for i in range(d):
        for j in range(n):
            if i == 0:
                starting_indices[i, j] = j + 1
            elif j == 0:
                starting_indices[i, j] = np.math.factorial(n + i) / np.math.factorial(i) / np.math.factorial(n)
            else:
                starting_indices[i, j] = starting_indices[i, j - 1] + \
                                         (starting_indices[i - 1, -1] - starting_indices[i - 1, 0] + 1) - \
                                         (starting_indices[i - 1, j - 1] - starting_indices[i - 1, 0])
    starting_indices = starting_indices.astype(int)
    print(starting_indices)
    return starting_indices.astype(np.int)


def getRangeOfidx(idx, starting_indices):
    degree_idx = np.argmax(np.where(starting_indices - idx <= 0, starting_indices - idx, -np.inf), axis=0)
    v = np.choose(degree_idx, starting_indices - idx)
    idx_monomial = np.argmax(v[np.where(v <= 0)[0]])
    return starting_indices[degree_idx[idx_monomial], idx_monomial], degree_idx[idx_monomial], idx_monomial, idx


def dissectIndex(idx, starting_indices):
    dissected_monomials = []
    temp_idx = idx
    starting_index, degree, monomial_idx, _ = getRangeOfidx(temp_idx, starting_indices)
    distance_to_start = idx - starting_index
    keep_dissecting = True
    if distance_to_start == 0:
        dissected_monomials.append(getRangeOfidx(temp_idx, starting_indices))
    else:
        # dissected_monomials.append(getRangeOfidx(starting_indices[degree - 1, monomial_idx], starting_indices))
        monomial_idx_dissected = monomial_idx
        while keep_dissecting:
            starting_index, degree_second_part, monomial_idx_dissected, _ = getRangeOfidx(temp_idx, starting_indices)
            if monomial_idx_dissected != monomial_idx:
                dissected_monomials.append(
                    getRangeOfidx(starting_indices[degree - 1 - degree_second_part, monomial_idx], starting_indices))
                keep_dissecting = False
            else:
                temp_idx = starting_indices[degree_second_part - 1, monomial_idx_dissected] + distance_to_start

        dissected_monomials.append(getRangeOfidx(temp_idx, starting_indices))
    return dissected_monomials


def dissectUniqueMonomial(monomial, starting_indices):
    return [getRangeOfidx(starting_indices[0,
                                           monomial[-2]],
                          starting_indices) for i in
            range(monomial[-3] + 1)] if monomial[-3] > 0 else [monomial]


def shiftIndex(idxs, starting_indices):
    ### We can make it faster using combinatorial analysis
    if idxs[0] == 1 and idxs[-1] == 19:
        print('Stop')
    ranges = np.array([getRangeOfidx(idx, starting_indices) for idx in idxs])
    min_idx = np.argsort([x[-2] for x in ranges]).astype(np.int)
    idxs = idxs[min_idx]
    ranges = ranges[min_idx]
    obtained_degree = ranges[0][1] + ranges[0][1]
    # if idxs[0] - ranges[0][0] == 0:
    #     pass
    #     # return (idxs[min_idx[-1]] - starting_indices[ranges[min_idx[-1]][1]], ranges[min_idx[0]][-1]]) + \
    #     #                starting_indices[obtained_degree, ranges[min_idx[0]]]
    # else:
    shifted_idx = 0
    monomial_to_multiply = []

    for i in range(len(idxs)):
        keep_dissecting = True
        temp_idx = idxs[i]
        while keep_dissecting:
            dissected_temp_idxs = dissectIndex(temp_idx, starting_indices)
            if len(dissected_temp_idxs) == 1 or dissected_temp_idxs[0][-3] >= 0:
                monomial_to_multiply.extend(dissectUniqueMonomial(dissected_temp_idxs[0], starting_indices))
                if len(dissected_temp_idxs) == 1:
                    keep_dissecting = False
                else:
                    temp_idx = dissected_temp_idxs[1][-1]
            else:
                # if dissected_temp_idxs[0][-3] > 0:
                #     monomial_to_multiply.extend([getRangeOfidx(starting_indices[0,
                #                                                                 dissected_temp_idxs[0][-2]],
                #                                                starting_indices) for i in
                #                                  range(dissected_temp_idxs[0][-3] + 1)])
                # else:
                #     monomial_to_multiply.append(dissected_temp_idxs[0])

                if dissected_temp_idxs[1][-1] - dissected_temp_idxs[1][0] != 0:
                    temp_idx = dissected_temp_idxs[1][-1]
                else:
                    monomial_to_multiply.extend(dissectUniqueMonomial(dissected_temp_idxs[1], starting_indices))
                    keep_dissecting = False
    monomial_to_multiply = np.array(monomial_to_multiply)
    sort_dissected = np.argsort([x[-2] for x in monomial_to_multiply])
    obtained_index = monomial_to_multiply[sort_dissected]
    while len(obtained_index) > 1:
        x, y = obtained_index[-2], obtained_index[-1]
        degree_of_y = y[-3]
        distance_to_same_degree_monomial = y[-1] - starting_indices[degree_of_y, x[-2]]
        obtained_index = obtained_index[:-1, :]  # remove last entry
        obtained_index[-1] = getRangeOfidx(starting_indices[degree_of_y + 1, x[-2]] +
                                           distance_to_same_degree_monomial, starting_indices)

    return obtained_index.flatten()

    # while keep_descending:
    #     temp_idxs = dissectIndex(idxs[min_idx[0]], starting_indices)
    #     monomial_to_multiply.extend(temp_idxs)
    #     if len(temp_idxs) == 1:
    #         if temp_idxs[0][1] >= ranges[min_idx[-1]][1]:
    #             pass


def plotSolutions(P, sols):
    plt.scatter(P[:, 0], P[:, 1], marker='o')
    X = np.random.randn(100, P.shape[1]) * np.linalg.norm(P)
    for normal in sols:
        subspace = nullspace(normal)
        points_on_subspace = X.dot(subspace.dot(subspace.T))
        plt.plot(points_on_subspace[:, 0], points_on_subspace[:, 1])
    plt.xlim([np.min(P[:, 0]) - 0.5, np.max(P[:, 0]) + 0.5])
    plt.ylim([np.min(P[:, 1]) - 0.5, np.max(P[:, 1]) + 0.5])
    plt.savefig("sols_loay.png")


def constructShiftMatrixMurad(matrix, args):
    num_poly = matrix.shape[0]
    meculay_matrix = lil_matrix((args["pD"], args["qD"]), dtype=np.float)
    starting_indices = constructIndicesMatrix(args["n"], args["d"])
    degree_per_row = np.zeros((num_poly,))
    for i in range(num_poly):
        max_non_zero_index = np.max(np.nonzero(matrix[i, :])[1])
        degree_per_row[i] = getRangeOfidx(max_non_zero_index, starting_indices)[-3]
    DD = np.sum(degree_per_row + 1) - args["n"] + 1
    meculay_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix

    num_rows = matrix.shape[0]
    for i in range(num_poly):
        monomial_shifted = np.eye(N=args["qD"], M=1, dtype=np.int).flatten()
        max_num_monomials_attained = int(
            ncr(int(args["n"] + DD - degree_per_row[i] - 1), int(DD - degree_per_row[i] - 1)))
        for j in range(1, max_num_monomials_attained):
            if num_rows == 40:
                print('Stop')
            if j == 0:
                meculay_matrix[num_rows, :matrix.shape[1]] = matrix[i, :]
            else:
                monomial_shifted = np.roll(monomial_shifted, 1)
                idx_monomial = np.where(monomial_shifted)[0][0]
                for non_zero_idx in np.nonzero(matrix[i])[1]:
                    if non_zero_idx == 0:
                        meculay_matrix[num_rows, idx_monomial] = matrix[i, non_zero_idx]
                    else:
                        meculay_matrix[num_rows, shiftIndex(
                            np.array([non_zero_idx, idx_monomial], dtype=np.int), starting_indices)[-1]] = matrix[
                            i, non_zero_idx]
            num_rows += 1

    return meculay_matrix, starting_indices


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def diameter(P):
    distance = np.inf
    for p1 in range(P.shape[0]):
        A = np.delete(range(P.shape[0]), p1)
        for p2 in A:
            a = np.sqrt(((P[p1, 0] - P[p2, 0]) ** 2) + ((P[p1, 1] - P[p2, 1]) ** 2))
            if a < distance:
                distance = a
    return a
if __name__ == '__main__':
## add P and the BinarySearch