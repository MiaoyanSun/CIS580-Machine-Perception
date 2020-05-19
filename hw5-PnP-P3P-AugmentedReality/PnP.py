import numpy as np
from est_homography import est_homography

def PnP(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_PnP.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####


    R = np.eye(3)
    t = np.zeros([3])


    # step 1: recover H from "est_homography"
    world_coord = np.zeros([4,2])
    world_coord[:,0] = Pw[:,0]
    world_coord[:,1] = Pw[:,1]
  
    # camera ~ H * world    Y ~ H * X
    H = est_homography(world_coord, Pc)
 

    # step 2: K^-1 H = (R t) = (h1' h2' h3')
    RT_matrix = np.zeros([3,3])
    RT_matrix = np.matmul(np.linalg.inv(K), H)
    
    h1_ = RT_matrix[:,0]
    h2_ = RT_matrix[:,1]
    h3_ = RT_matrix[:,2]

    A = np.zeros([3,3])
    A[:,0] = h1_
    A[:,1] = h2_
    A[:,2] = np.cross(h1_, h2_)
    # step 3: use SVD to find R and T
    [U, S, V] = np.linalg.svd(A)
    diagonal_matrix = np.eye(3)
    diagonal_matrix[2,2] = np.linalg.det(U @ V)

    R = U @ diagonal_matrix @ V

    R = np.transpose(R)

    t = h3_ / np.linalg.norm(h1_)

    t = -R@t


    # change to Rwc and twc


    ##### STUDENT CODE END #####








    return R, t
