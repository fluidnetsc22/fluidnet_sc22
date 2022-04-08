import torch
from . import CellType
import scipy
from scipy.sparse import *
from scipy import *
import numpy as np
from timeit import default_timer


def CreateCSR_Direct_3D(flags):

    start = default_timer()

    assert (flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (d > 1)
    if (not is3D):
        assert d == 1, '2D velocity field but zdepth > 1'

    assert (flags.is_contiguous()), 'Input is not contiguous'

    length = h * w * d
    A = np.zeros((length))

    A_val_l = []
    I_A = torch.zeros((length + 1), dtype=torch.int32)
    J_A_l = []

    I_A[0] = 0
    counter = 0
    z = 0

    for k in range(d):
        for j in range(h):
            for i in range(w):
                z += 1
                if flags[0, 0, k, j, i].eq(CellType.TypeFluid):

                    # Filling order = back, down, left, diag, front, right, top,
                    if d > 1:
                        if flags[0, 0, k - 1, j, i].eq(CellType.TypeFluid):
                            A_back = -1
                            counter += 1
                            A_val_l.append(A_back)
                            J_A_l.append((h * w) * (k - 1) + w * j + i)

                    if flags[0, 0, k, j - 1, i].eq(CellType.TypeFluid):
                        A_down = -1
                        counter += 1
                        A_val_l.append(A_down)
                        J_A_l.append((h * w) * k + w * (j - 1) + i)
                    if flags[0, 0, k, j, i - 1].eq(CellType.TypeFluid):
                        A_left = -1
                        counter += 1
                        A_val_l.append(A_left)
                        J_A_l.append((h * w) * k + w * j + (i - 1))

                    if d > 1:
                        Diag = 6
                    else:
                        Diag = 4

                    if flags[0, 0, k, j, i - 1].eq(CellType.TypeObstacle):
                        Diag -= 1
                    if flags[0, 0, k, j, i + 1].eq(CellType.TypeObstacle):
                        Diag -= 1
                    if flags[0, 0, k, j - 1, i].eq(CellType.TypeObstacle):
                        Diag -= 1
                    if flags[0, 0, k, j + 1, i].eq(CellType.TypeObstacle):
                        Diag -= 1
                    if d > 1:
                        if flags[0, 0, k - 1, j, i].eq(CellType.TypeObstacle):
                            Diag -= 1
                        if flags[0, 0, k + 1, j, i].eq(CellType.TypeObstacle):
                            Diag -= 1

                    counter += 1
                    A_val_l.append(Diag)
                    J_A_l.append((h * w) * k + w * j + (i))

                    if flags[0, 0, k, j, i + 1].eq(CellType.TypeFluid):
                        A_right = -1
                        counter += 1
                        A_val_l.append(A_right)
                        J_A_l.append((h * w) * k + w * j + (i + 1))
                    if flags[0, 0, k, j + 1, i].eq(CellType.TypeFluid):
                        A_up = -1
                        counter += 1
                        A_val_l.append(A_up)
                        J_A_l.append((h * w) * k + w * (j + 1) + i)
                    if d > 1:
                        if flags[0, 0, k + 1, j, i].eq(CellType.TypeFluid):
                            A_front = -1
                            counter += 1
                            A_val_l.append(A_front)
                            J_A_l.append((h * w) * (k + 1) + w * j + i)

                I_A[z] = counter

        if k % 2 == 0:
            print(" Completion {0:6.3f} % ".format(100 * k / d))

    A_val = torch.FloatTensor(A_val_l)
    J_A = torch.IntTensor(J_A_l)

    print("A val ", A_val, A_val.shape)
    print("I_A ", I_A, I_A.shape)
    print("J_A ", J_A, J_A.shape)

    end = default_timer()
    time = (end - start)
    print("A matrix creation time", time)

    return A_val, I_A, J_A
