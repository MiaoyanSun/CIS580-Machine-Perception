import numpy as np
from est_homography import est_homography

def warp_pts(X, Y, interior_pts):
    """ 
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts. 
        These coordinate describe where a point inside the goal will be warped 
        to inside the penn logo. For this assignment, you can keep these new 
        coordinates as float numbers.
        
    """
    
    # You should Complete est_homography first!
    H = est_homography(X, Y)
    
    ##### STUDENT CODE START #####


    warped_pts = []

    pts_num = len(interior_pts)  # the number of points inside goal

    for i in range(0, pts_num):
        position_vec = np.array([[interior_pts[i, 0]], [interior_pts[i, 1]], [1]])
        logo = H @ position_vec  # the corresponding coordinates inside penn logo, 3X1 vector
        logo_x = logo[0, 0] / logo[2, 0]  # corresponding x inside penn logo, float
        logo_y = logo[1, 0] / logo[2, 0]  # corresponding y inside penn logo, float
        pt = [logo_x, logo_y]  # list that store the x,y coordinates inside penn logo
        warped_pts.append(pt)  # matrix that stored the corresponding coordinates inside penn logo

    warped_pts = np.array(warped_pts)  # NX2 matrix containing
    ##### STUDENT CODE END #####

    
    ##### STUDENT CODE END #####
    
    return warped_pts