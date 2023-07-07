# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:59:17 2023

@author: anton
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import mpmath

data = np.genfromtxt("data/passpunktliste.txt")
#data = data[:,1:4]
X18 = np.array([data[0, 1],
                data[1, 1],
                data[2, 1],
                data[3, 1],
                data[4, 1],
                data[5, 1],
                data[6, 1],
                data[7, 1],
                data[8, 1],
                data[9, 1],
                data[11, 1],
                data[12, 1],
                data[14, 1],
                data[15, 1],
                data[17, 1],
                data[18, 1],
                data[20, 1],
                data[21, 1]])

Y18 = np.array([data[0, 2],
                data[1, 2],
                data[2, 2],
                data[3, 2],
                data[4, 2],
                data[5, 2],
                data[6, 2],
                data[7, 2],
                data[8, 2],
                data[9, 2],
                data[11, 2],
                data[12, 2],
                data[14, 2],
                data[15, 2],
                data[17, 2],
                data[18, 2],
                data[20, 2],
                data[21, 2]])

Z18 = np.array([data[0, 3],
                data[1, 3],
                data[2, 3],
                data[3, 3],
                data[4, 3],
                data[5, 3],
                data[6, 3],
                data[7, 3],
                data[8, 3],
                data[9, 3],
                data[11, 3],
                data[12, 3],
                data[14, 3],
                data[15, 3],
                data[17, 3],
                data[18, 3],
                data[20, 3],
                data[21, 3]])
#X18 = X18[:,np.newaxis]
#Y18 = Y18[:,np.newaxis]
#Z18 = Z18[:,np.newaxis]

def rotmat_AR(omega, phi, kappa):
    r11 = np.cos(phi) * np.cos(kappa)
    r12 = -np.cos(phi) * np.sin(kappa)
    r13 = np.sin(phi)
    r21 = np.cos(omega) * np.sin(kappa) + np.sin(omega) * np.sin(phi) * np.cos(kappa)
    r22 = np.cos(omega) * np.cos(kappa) - np.sin(omega) * np.sin(phi) * np.sin(kappa)
    r23 = -np.sin(omega) * np.cos(phi)
    r31 = np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(phi) * np.cos(kappa)
    r32 = np.sin(omega) * np.cos(kappa) + np.cos(omega) * np.sin(phi) * np.sin(kappa)
    r33 = np.cos(omega) * np.cos(phi)

    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    return R

def rotmat_AR_2(omega, phi, kappa):
    r11 =  math.cos(phi) *  math.cos(kappa)
    r12 = - math.cos(phi) *  math.sin(kappa)
    r13 =  math.sin(phi)
    r21 =  math.cos(omega) *  math.sin(kappa) +  math.sin(omega) *  math.sin(phi) *  math.cos(kappa)
    r22 =  math.cos(omega) *  math.cos(kappa) -  math.sin(omega) *  math.sin(phi) *  math.sin(kappa)
    r23 = - math.sin(omega) *  math.cos(phi)
    r31 =  math.sin(omega) *  math.sin(kappa) -  math.cos(omega) *  math.sin(phi) *  math.cos(kappa)
    r32 =  math.sin(omega) *  math.cos(kappa) +  math.cos(omega) *  math.sin(phi) *  math.sin(kappa)
    r33 =  math.cos(omega) *  math.cos(phi)

    R =  np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    
    return R

def rotmat_AR_3(omega, phi, kappa):
    omega = omega.item()
    phi = phi.item()
    kappa = kappa.item()
    mpmath.mp.dps = 10
    r11 =  mpmath.cos(phi) *  mpmath.cos(kappa)
    r12 = - mpmath.cos(phi) *  mpmath.sin(kappa)
    r13 =  mpmath.sin(phi)
    r21 =  mpmath.cos(omega) *  mpmath.sin(kappa) +  mpmath.sin(omega) *  mpmath.sin(phi) *  mpmath.cos(kappa)
    r22 =  mpmath.cos(omega) *  mpmath.cos(kappa) -  mpmath.sin(omega) *  mpmath.sin(phi) *  mpmath.sin(kappa)
    r23 = - mpmath.sin(omega) *  mpmath.cos(phi)
    r31 =  mpmath.sin(omega) *  mpmath.sin(kappa) -  mpmath.cos(omega) *  mpmath.sin(phi) *  mpmath.cos(kappa)
    r32 =  mpmath.sin(omega) *  mpmath.cos(kappa) +  mpmath.cos(omega) *  mpmath.sin(phi) *  mpmath.sin(kappa)
    r33 =  mpmath.cos(omega) *  mpmath.cos(phi)

    R =  np.array([[float(r11), float(r12), float(r13)], [float(r21), float(r22), float(r23)], [float(r31), float(r32), float(r33)]])
    
    return R

def DLT(X, Y, Z, x, y):
    anz_points = len(X)

    A = np.zeros((anz_points*2, 11))
    b = np.zeros((anz_points*2, 1))

    c = 0

    for k in range(0, 2*anz_points, 2):
        A[k, 0] = X[c]
        A[k, 1] = Y[c]
        A[k, 2] = Z[c]
        A[k, 3] = 1
        A[k, 8] = -x[c] * X[c]
        A[k, 9] = -x[c] * Y[c]
        A[k, 10] = -x[c] * Z[c]

        A[k+1, 4] = X[c]
        A[k+1, 5] = Y[c]
        A[k+1, 6] = Z[c]
        A[k+1, 7] = 1
        A[k+1, 8] = -y[c] * X[c]
        A[k+1, 9] = -y[c] * Y[c]
        A[k+1, 10] = -y[c] * Z[c]

        b[k] = x[c]
        b[k+1] = y[c]

        c += 1
    N = np.dot(A.T, A)
    L = np.dot(np.linalg.inv(N), np.dot(A.T, b))
    #print(L.shape)
    v = np.dot(A, L) - b
    
    tmp = np.array([L[0], L[1], L[2], L[4], L[5], L[6], L[8], L[9], L[10]])
    tmp = tmp.reshape(3,3)
    X0 = -1 * np.linalg.inv(tmp) @ np.array([[L[3]], [L[7]], [1]])
    
    Lh = -1 / np.sqrt(L[8]**2 + L[9]**2 + L[10]**2)
    x0 = Lh * (L[0] * L[8] + L[1] * L[9] + L[2] * L[10])
    y0 = Lh * (L[4] * L[8] + L[5] * L[9] + L[6] * L[10])
    
    cx = np.sqrt(Lh**2 * (L[0]**2 + L[1]**2 + L[2]**2) - x0**2)
    cy = np.sqrt(Lh**2 * (L[4]**2 + L[5]**2 + L[6]**2) - y0**2)
    
    r11 = Lh * (x0 * L[8] - L[0]) / cx
    r21 = Lh * (x0 * L[9] - L[1]) / cx
    r31 = Lh * (x0 * L[10] - L[2]) / cx
    
    r12 = Lh * (y0 * L[8] - L[4]) / cy
    r22 = Lh * (y0 * L[9] - L[5]) / cy
    r32 = Lh * (y0 * L[10] - L[6]) / cy
    
    r13 = Lh * L[8]
    r23 = Lh * L[9]
    r33 = Lh * L[10]
    
    R = np.array([r11, r12, r13,
                  r21, r22, r23,
                  r31, r32, r33])
    R = R.reshape(3,3)
    
    # Normalisierung der Rotationsmatrix
    c1 = np.cross(R[:, 1], R[:, 2])
    c1 = c1 / np.linalg.norm(c1)
    
    c2 = np.cross(R[:, 2], R[:, 0])
    c2 = c2 / np.linalg.norm(c2)
    
    c3 = R[:, 2] / np.linalg.norm(R[:, 2])
    
    R = np.column_stack((c1, c2, c3))
    
    # Berechnung der Rotationswinkel
    om = np.arctan2(-R[1, 2], R[2, 2])
    kap = np.arctan2(-R[0, 1], R[0, 0])
    phi = np.arctan2(R[0, 2], np.sqrt(R[1, 2]**2 + R[2, 2]**2))
    
    return X0, R, x0, y0, om, phi, kap

def main(X18,Y18,Z18,x_coord,y_coord):
    pixelgroesse = 4.52 * 1e-3
    bildgroesse_x = 5168
    bildgroesse_y = 3448
    
    pixelkoord_pp18 = np.column_stack((np.arange(0, 10).tolist() + [12, 13, 15, 16, 18, 19, 21, 22], x_coord, y_coord))
    bildkoord_pp18 = np.copy(pixelkoord_pp18.astype(np.float32))
    bildkoord_pp18[:, 1] = ((bildkoord_pp18[:, 1] - bildgroesse_x / 2 - 1) * pixelgroesse + pixelgroesse / 2) * 7952 / 5168
    bildkoord_pp18[:, 2] = ((-bildkoord_pp18[:, 2] + bildgroesse_y / 2 - 1) * pixelgroesse + pixelgroesse / 2) * 7952 / 5168
    
    # Plot zur Kontrolle der umgerechneten Bildkoordinaten der Passpunkte
    
    plt.plot(bildkoord_pp18[:, 1], bildkoord_pp18[:, 2], 'rx')
    plt.title("Bildkoordinaten der Passpunkte")
    for i in range(len(pixelkoord_pp18)):
        plt.text(bildkoord_pp18[i, 1], bildkoord_pp18[i, 2], str(pixelkoord_pp18[i, 0]))
    
    A1 = -0.0648 * pixelgroesse
    A2 = 0.0720 * pixelgroesse
    B1 = 3.9245e-05 * pixelgroesse
    B2 = 0.0015 * pixelgroesse
    
    # Näherungswerte bestimmen
    X0, R, x0, y0, om, phi, kap = DLT(X18, Y18, Z18, bildkoord_pp18[:, 1], bildkoord_pp18[:, 2])
    
    x_o = np.zeros((6, 1))
    x_o[0] = X0[0]
    x_o[1] = X0[1]
    x_o[2] = X0[2]
    x_o[3] = om
    x_o[4] = phi
    x_o[5] = kap
    
    verzeichnung = False
    
    b = np.reshape(bildkoord_pp18[:, 1:], (-1, 1))  # Struktur [x1;y1;xn;yn]
    
    c = 5.3701e+03 * pixelgroesse
    
    n = len(b)
    u = 6
    
    Pbb = np.eye(n)
    Qbb = np.eye(n)
    
    f_x_o = np.zeros((n, 1))
    f_x_o = f_x_o.reshape(-1)
    f_x_dach = np.zeros((n, 1))
    f_x_dach = f_x_dach.reshape(-1)
    A = np.zeros((n, u))
    
    iter = 0
    
    while 1:
        iter = iter + 1
    
        R1 = rotmat_AR(x_o[3], x_o[4], x_o[5])
        R = rotmat_AR_2(x_o[3], x_o[4], x_o[5])
        R3 = rotmat_AR_3(x_o[3], x_o[4], x_o[5])
        R = R.reshape(3,3)
        #print(R.shape)
            
        Zx = R[0, 0] * (X18 - x_o[0]) + R[1, 0] * (Y18 - x_o[1]) + R[2, 0] * (Z18 - x_o[2])
        Zy = R[0, 1] * (X18 - x_o[0]) + R[1, 1] * (Y18 - x_o[1]) + R[2, 1] * (Z18 - x_o[2])
        N = R[0, 2] * (X18 - x_o[0]) + R[1, 2] * (Y18 - x_o[1]) + R[2, 2] * (Z18 - x_o[2])
        #Zx = Zx[:,np.newaxis]
        #Zy = Zy[:,np.newaxis]
        #N = N[:,np.newaxis]
        
        xs = x0 - c * Zx / N
        ys = y0 - c * Zy / N
        f_x_o[0::2] = xs
        f_x_o[1::2] = ys
        
        w = b - f_x_o[:,np.newaxis]
        
        A[::2, 0] = -c / N ** 2 * (R[0, 2] * Zx - R[0, 0] * N)
        A[::2, 1] = -c / N ** 2 * (R[1, 2] * Zx - R[1, 0] * N)
        A[::2, 2] = -c / N ** 2 * (R[2, 2] * Zx - R[2, 0] * N)
        A[::2, 3] = -c / N * (Zx / N * (R[2, 2] * (Y18 - x_o[1]) - R[1, 2] * (Z18 - x_o[2])) - R[2, 0] * (Y18 - x_o[1]) + R[1, 0] * (Z18 - x_o[2]))
        A[::2, 4] = c / N * (Zx / N * (Zx * np.cos(x_o[5]) - Zy * np.sin(x_o[5])) + N * np.cos(x_o[5]))
        A[::2, 5] = -c / N * Zy
        
        A[1::2, 0] = -c / N ** 2 * (R[0, 2] * Zy - R[0, 1] * N)
        A[1::2, 1] = -c / N ** 2 * (R[1, 2] * Zy - R[1, 1] * N)
        A[1::2, 2] = -c / N ** 2 * (R[2, 2] * Zy - R[2, 1] * N)
        A[1::2, 3] = -c / N * (Zy / N * (R[2, 2] * (Y18 - x_o[1]) - R[1, 2] * (Z18 - x_o[2])) - R[2, 1] * (Y18 - x_o[1]) + R[1, 1] * (Z18 - x_o[2]))
        A[1::2, 4] = c / N * (Zx / N * (Zx * np.cos(x_o[5]) - Zy * np.sin(x_o[5])) + N * np.cos(x_o[5]))
        A[1::2, 5] = -c / N * Zy
        
        A[1::2, 0] = -c / N ** 2 * (R[0, 2] * Zy - R[0, 1] * N)
        A[1::2, 1] = -c / N ** 2 * (R[1, 2] * Zy - R[1, 1] * N)
        A[1::2, 2] = -c / N ** 2 * (R[2, 2] * Zy - R[2, 1] * N)
        A[1::2, 3] = -c / N * (Zy / N * (R[2, 2] * (Y18 - x_o[1]) - R[1, 2] * (Z18 - x_o[2])) - R[2, 1] * (Y18 - x_o[1]) + R[1, 1] * (Z18 - x_o[2]))
        A[1::2, 4] = c / N * (Zy / N * (Zx * np.cos(x_o[5]) - Zy * np.sin(x_o[5])) - N * np.sin(x_o[5]))
        A[1::2, 5] = c / N * Zx
        
        A_transposed = np.transpose(A)
        part1 = np.dot(np.dot(A_transposed, Pbb), A)
        part2 = np.dot(np.dot(A_transposed, Pbb), w)
    
        delta_x_dach = np.linalg.solve(part1, part2)
        #delta_x_dach = np.linalg.inv(A.T @ Pbb @ A) @ (A.T @ Pbb @ w)
        x_dach = x_o + delta_x_dach
        v_dach = A @ delta_x_dach - w
        b_dach = b + v_dach
        
        R = rotmat_AR(x_dach[3], x_dach[4], x_dach[5])
        Zx = R[0, 0] * (X18 - x_dach[0]) + R[1, 0] * (Y18 - x_dach[1]) + R[2, 0] * (Z18 - x_dach[2])
        Zy = R[0, 1] * (X18 - x_dach[0]) + R[1, 1] * (Y18 - x_dach[1]) + R[2, 1] * (Z18 - x_dach[2])
        N = R[0, 2] * (X18 - x_dach[0]) + R[1, 2] * (Y18 - x_dach[1]) + R[2, 2] * (Z18 - x_dach[2])
        
        b_dach_x = b_dach[::2]
        b_dach_y = b_dach[1::2]
        #if verzeichnung:
        #    verz_x, verz_y = verzeichnungBerechnen(A1, A2, B1, B2, b_dach_x, b_dach_y)
        #    xs = x0 - c * Zx / N + verz_x
        #    ys = y0 - c * Zy / N + verz_y
        #else:
        #    xs = x0 - c * Zx / N
        #    ys = y0 - c * Zy / N
            
        xs = x0 - c * Zx / N
        ys = y0 - c * Zy / N
        
        f_x_dach[::2] = xs
        f_x_dach[1::2] = ys
    
        probe = b_dach.reshape(-1) - f_x_dach
        
        if iter == 100 or np.max(np.abs(probe)) < 1e-10:
            break
        #print(x_dach)
        x_o = x_dach
    
    sig02 = (v_dach.T @ Pbb @ v_dach) / (n - u)
    
    Qxx = np.linalg.inv(A.T @ Pbb @ A)
    Kxx = sig02 * Qxx
    
    std_xx = np.sqrt(np.diag(Kxx))
    
    Qbb_dach = A @ Qxx @ A.T
    Kbb = sig02 * Qbb_dach
    
    Qvv = Qbb - A @ np.linalg.inv(A.T @ Pbb @ A) @ A.T
    R_r = Qvv @ Pbb
    red = np.diag(R_r)
    r_test = np.sum(red)
    
    plt.figure()
    plt.plot(xs, ys, 'rx')
    plt.title("Bildkoordinaten der Passpunkte nach der Ausgleichung")
    for i in range(len(pixelkoord_pp18)):
        plt.text(xs[i], ys[i], str(pixelkoord_pp18[i, 0]))
    
    plt.show()