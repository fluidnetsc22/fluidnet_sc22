import torch
from . import CellType
import scipy
from scipy.sparse import *
from scipy import *
import numpy as np
from timeit import default_timer


def CreateCSR_Direct(flags):

    start = default_timer()

    assert (flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (flags.size(1) == 3)
    if (not is3D):
        assert d == 1, '2D velocity field but zdepth > 1'

    assert (flags.is_contiguous()), 'Input is not contiguous'

    length = h * w
    A = np.zeros((length))

    A_val_l = []
    I_A = torch.zeros((length + 1), dtype=torch.int32)
    J_A_l = []

    I_A[0] = 0
    counter = 0

    for j in range(h):
        for i in range(w):

            if flags[0, 0, 0, j, i].eq(CellType.TypeFluid):

                # Filling order = down, left, diag, right, top

                if flags[0, 0, 0, j - 1, i].eq(CellType.TypeFluid):
                    A_down = -1
                    counter += 1
                    A_val_l.append(A_down)
                    J_A_l.append(j * w + i - w)
                if flags[0, 0, 0, j, i - 1].eq(CellType.TypeFluid):
                    A_left = -1
                    counter += 1
                    A_val_l.append(A_left)
                    J_A_l.append(j * w + i - 1)

                Diag = 4

                if flags[0, 0, 0, j, i - 1].eq(CellType.TypeObstacle):
                    Diag -= 1
                if flags[0, 0, 0, j, i + 1].eq(CellType.TypeObstacle):
                    Diag -= 1
                if flags[0, 0, 0, j - 1, i].eq(CellType.TypeObstacle):
                    Diag -= 1
                if flags[0, 0, 0, j + 1, i].eq(CellType.TypeObstacle):
                    Diag -= 1

                counter += 1
                A_val_l.append(Diag)
                J_A_l.append(j * w + i)

                if flags[0, 0, 0, j, i + 1].eq(CellType.TypeFluid):
                    A_right = -1
                    counter += 1
                    A_val_l.append(A_right)
                    J_A_l.append(j * w + i + 1)
                if flags[0, 0, 0, j + 1, i].eq(CellType.TypeFluid):
                    A_up = -1
                    counter += 1
                    A_val_l.append(A_up)
                    J_A_l.append(j * w + i + w)

            I_A[(j * w) + i + 1] = counter

        if j % 40 == 0:
            print(" Completion {0:6.3f} % ".format(100 * j / h))

    A_val = torch.FloatTensor(A_val_l)
    J_A = torch.IntTensor(J_A_l)

    print("A val ", A_val)
    print("I_A ", I_A)
    print("J_A ", J_A)

    end = default_timer()
    time = (end - start)
    print("A matrix creation time", time)

    return A_val, I_A, J_A


def createMatrixA(flags):

    start = default_timer()

    #cuda = torch.device('cuda')
    assert (flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (flags.size(1) == 3)
    if (not is3D):
        assert d == 1, '2D velocity field but zdepth > 1'

    assert (flags.is_contiguous()), 'Input is not contiguous'

    # We declare the super tensor A. The declaration order is very importan
    # as we will have to flatten the divergency consecuently!

    # This module might be one of the most inefficient codes in the entire FluidNet
    # Just have to run it once
    length = h * w
    A = np.zeros((length, length))

    counter = 0
    for j in range(h):
        for i in range(w):

            if flags[0, 0, 0, j, i].eq(CellType.TypeFluid):

                A[counter, counter] = 4

                if flags[0, 0, 0, j, i - 1].eq(CellType.TypeObstacle):
                    A[counter, counter] -= 1
                if flags[0, 0, 0, j, i + 1].eq(CellType.TypeObstacle):
                    A[counter, counter] -= 1
                if flags[0, 0, 0, j - 1, i].eq(CellType.TypeObstacle):
                    A[counter, counter] -= 1
                if flags[0, 0, 0, j + 1, i].eq(CellType.TypeObstacle):
                    A[counter, counter] -= 1

                if flags[0, 0, 0, j, i - 1].eq(CellType.TypeFluid):
                    A[counter, counter - 1] -= 1
                if flags[0, 0, 0, j, i + 1].eq(CellType.TypeFluid):
                    A[counter, counter + 1] -= 1
                if flags[0, 0, 0, j - 1, i].eq(CellType.TypeFluid):
                    A[counter, counter - w] -= 1
                if flags[0, 0, 0, j + 1, i].eq(CellType.TypeFluid):
                    A[counter, counter + w] -= 1

            counter += 1

    print("A ", A)

    end = default_timer()
    time = (end - start)
    print("A matrix creation time", time)

    return A


def CreateCSR(A):

    h = A.size(0)
    w = A.size(1)

    A_val_l = []
    I_A = torch.zeros((w + 1), dtype=torch.int32)
    J_A_l = []

    I_A[0] = 0
    counter = 0

    for i in range(0, w):

        for j in range(0, h):
            if A[j, i] != 0:
                A_val_l.append(A[j, i])
                J_A_l.append(j)
                counter += 1
        I_A[i + 1] = counter

    A_val = torch.FloatTensor(A_val_l)
    J_A = torch.IntTensor(J_A_l)

    print("A val ", A_val)
    print("I_A ", I_A)
    print("J_A ", J_A)

    return A_val, I_A, J_A


def CreateCSR_scipy(A):

    start = default_timer()
    h = A[0].size
    w = A[1].size

    A_val_l = csr_matrix(A).data
    I_A_l = csr_matrix(A).indptr
    J_A_l = csr_matrix(A).indices

    I_A = torch.IntTensor(I_A_l)
    A_val = torch.FloatTensor(A_val_l)
    J_A = torch.IntTensor(J_A_l)

    print("A val scipy ", A_val)
    print("I_A scipy ", I_A)
    print("J_A scipy ", J_A)

    end = default_timer()
    time = (end - start)
    print("CSR conversion time", time)

    return A_val, I_A, J_A
