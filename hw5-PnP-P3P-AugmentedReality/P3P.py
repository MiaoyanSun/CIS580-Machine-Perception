import numpy as np
import math

def P3P(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_P3P.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    
    # define a,b,c,alpha,beta,gamma
    # step 1: set a, b, c
    P1, P2, P3 = Pw[0], Pw[1], Pw[2]
    P1c, P2c, P3c = Pc[0], Pc[1], Pc[2]

    a = np.linalg.norm(P2 - P3)
    b = np.linalg.norm(P1 - P3)
    c = np.linalg.norm(P1 - P2)

    # step 2: find unit vectors j1 (FOR P1), j2 (FOR P2), j3 (FOR P3)
    focal = [K[0, 0], K[1, 1]]
    f = (focal[0] + focal[1]) / 2
    center = [K[0, 2], K[1, 2]]
    u0, v0 = center[0], center[1]

    u1, v1 = P1c[0], P1c[1]
    u2, v2 = P2c[0], P2c[1]
    u3, v3 = P3c[0], P3c[1]
    u1 = u1 - u0
    u2 = u2 - u0
    u3 = u3 - u0
    v1 = v1 - v0
    v2 = v2 - v0
    v3 = v3 - v0
    j1_vector = np.array([u1, v1, f])
    j2_vector = np.array([u2, v2, f])
    j3_vector = np.array([u3, v3, f])
    j1 = j1_vector / np.linalg.norm(j1_vector)
    j2 = j2_vector / np.linalg.norm(j2_vector)
    j3 = j3_vector / np.linalg.norm(j3_vector)
    #print(np.linalg.norm(j1), np.linalg.norm(j2), np.linalg.norm(j3))
    # step 3: find alpha, beta, gamma from j1, j2, j3
    alpha = np.arccos(np.dot(j2, j3))
    beta = np.arccos(np.dot(j1, j3))
    gamma = np.arccos(np.dot(j1, j2))
    #print(alpha, beta, gamma)


    # define coefficients of the 4th degree polynomial
    # step 4: enter coefficients A0, A1, A2, A3, A4
    A4 = ((a**2 - c**2)/b**2 - 1)**2 - ((4*c**2)/ b**2) * (math.cos(alpha))**2
    A3 = 4 * (((a**2 - c**2)/b**2) * (1 - (a**2 - c**2)/b**2) * math.cos(beta) - (1 - (a**2 + c**2)/b**2)* math.cos(alpha) * math.cos(gamma) + 2*(c**2/b**2)*(math.cos(alpha))**2*math.cos(beta))
    A2 = 2*(((a**2 - c**2)/b**2)**2 - 1 + 2*((a**2 - c**2)/b**2)**2*(math.cos(beta))**2 + 2*((b**2 - c**2)/b**2)* (math.cos(alpha))**2-4*((a**2 + c**2)/b**2)*math.cos(alpha)*math.cos(beta)*math.cos(gamma) + 2*((b**2 - a**2)/b**2)*(math.cos(gamma))**2)
    A1 = 4 *(-((a**2 - c**2)/b**2)*(1+(a**2 - c**2)/b**2)*math.cos(beta) + 2*(a**2/b**2)*(math.cos(gamma))**2*math.cos(beta) - (1-(a**2 + c**2)/b**2)*math.cos(alpha)*math.cos(gamma))
    A0 = (1+(a**2 - c**2)/b**2)**2 - 4*(a**2/b**2)*(math.cos(gamma))**2

    AA = np.array([A4, A3, A2, A1, A0])
    roots = np.roots(AA)
    #print(roots[0], roots[3])
    v_1, v_2 = roots[0].real, roots[3].real


    # calculate real roots u and v
    u_1 = ((-1+(a**2 - c**2)/b**2)*v_1**2-2*((a**2 - c**2)/b**2)*math.cos(beta)*v_1+1+((a**2 - c**2)/b**2))/(2*(math.cos(gamma)-v_1*math.cos(alpha)))
    u_2 = ((-1+(a**2 - c**2)/b**2)*v_2**2-2*((a**2 - c**2)/b**2)*math.cos(beta)*v_2+1+((a**2 - c**2)/b**2))/(2*(math.cos(gamma)-v_2*math.cos(alpha)))
    #print(u_1, u_2)
   
    #u_, v_ = u_2, v_2
    u_, v_ = u_1, v_1
    s1 = (c**2/(1+u_**2-2*u_*math.cos(gamma)))**0.5
    #print("s1", s1)
    s2 = u_ * s1
    s3 = v_ * s1

    # check for valid distances

    # calculate 3D coordinates in Camera frame
    p1c = s1 * j1
    p2c = s2 * j2
    p3c = s3 * j3
    pc = np.zeros([3,3])
    pc[0] = p1c
    pc[1] = p2c
    pc[2] = p3c
    #pc = np.array([p1c, p2c, p3c])

    # Calculate R,t using Procrustes
    #Pw = np.array([Pw[0], Pw[1], Pw[2]])
    pw = np.zeros([3, 3])
    pw[0] = P1
    pw[1] = P2
    pw[2] = P3


    R, t = Procrustes(pc, pw)
    
    ##### STUDENT CODE END #####
    
    return R, t

def Procrustes(X, Y):
    """ 
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate 
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 1x3 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])

    A = Y
    B = X

    A_ = (A[0] + A[1] + A[2]) / 3
    B_ = (B[0] + B[1] + B[2]) / 3

    A = np.array([A[0] - A_, A[1] - A_, A[2] - A_]).transpose()
    B = np.array([B[0] - B_, B[1] - B_, B[2] - B_]).transpose()


    [U, S, V] = np.linalg.svd(B @ np.transpose(A))
    diagonal_matrix = np.eye(3)
    diagonal_matrix[2, 2] = np.linalg.det(np.transpose(V) @ np.transpose(U))
    R = np.transpose(V) @ diagonal_matrix @ np.transpose(U)




    t = A_ - R @ B_

    ##### STUDENT CODE END #####
    
    return R, t
