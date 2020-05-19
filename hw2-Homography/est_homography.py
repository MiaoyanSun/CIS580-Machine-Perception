import numpy as np

def est_homography(X, Y):
    """ 
    Calculates the homography of two planes, from the plane defined by X 
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X
        
    """
    
    ##### STUDENT CODE START #####

    H = []
    A = []  # a 8X9 matrix for solving H
    for i in range(0, 4):
        x = X.item(i, 0)  # x coordinates of goal corners in video frame
        y = X.item(i, 1)  # y coordinates of goal corneers in video frame

        x_ = Y.item(i, 0)  # x' coordinates of logo corners in penn logo
        y_ = Y.item(i, 1)  # y' coordinates of logo corners in penn logo

        ax = np.array([-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_])  # compute ax
        ay = np.array([0, 0, 0, -x, -y, -1, x * y_, y * y_, y_])  # compute ay
        A.append(ax)  # create A matrix
        A.append(ay)  # create A matrix
    A = np.array(A)  # get the A matrix

    [U, S, V] = np.linalg.svd(A)
    h = V[-1, :]  # the last row of V is the h vector
    H = np.array([h[0:3], h[3:6], h[6:9]])  # the H matrix

    ##### STUDENT CODE END #####
    # A = []
    #
    # for i in range(1, 5):
    #     ax = np.array([-X[i,1], -X[i,2], -1, 0, 0, 0, X[i,1] * Y[i,1], X[i,2] * Y[i,1], Y[i,1]])
    #     ay = np.array([0, 0, 0, -X[i,1], -X[i,2], -1, X[i,1] * Y[i,2], X[i,2] * Y[i,2], Y[i,2]])
    #     A += ax
    #     A += ay



    # A = []
    # for i = 1:4
    # ax = [-video_pts(1, i) - video_pts(2, i) - 1 0 0 0 video_pts(1, i) * logo_pts(1, i) video_pts(2, i) * logo_pts(1, i)
    #       logo_pts(1, i)];
    # ay = [0 0 0 - video_pts(1, i) - video_pts(2, i) - 1 video_pts(1, i) * logo_pts(2, i)
    #       video_pts(2, i) * logo_pts(2, i) logo_pts(2, i)];
    # A = [A; ax; ay];
    # end
    #
    # % ax1 = [-video_pts(1, 1) - video_pts(2, 1) - 1 0 0 0 video_pts(1, 1) * logo_pts(1, 1)
    #          video_pts(2, 1) * logo_pts(1, 1) logo_pts(1, 1)];
    # % ay1 = [0 0 0 - video_pts(1, 1) - video_pts(2, 1) - 1 video_pts(1, 1) * logo_pts(2, 1)
    #          video_pts(2, 1) * logo_pts(2, 1) logo_pts(2, 1)];
    # %
    # % ax2 = [-video_pts(1, 2) - video_pts(2, 2) - 1 0 0 0 video_pts(1, 2) * logo_pts(1, 2)
    #          video_pts(2, 2) * logo_pts(1, 2) logo_pts(1, 2)];
    # % ay2 = [0 0 0 - video_pts(1, 2) - video_pts(2, 2) - 1 video_pts(1, 2) * logo_pts(2, 2)
    #          video_pts(2, 2) * logo_pts(2, 2) logo_pts(2, 2)];
    # %
    # % ax3 = [-video_pts(1, 3) - video_pts(2, 3) - 1 0 0 0 video_pts(1, 3) * logo_pts(1, 3)
    #          video_pts(2, 3) * logo_pts(1, 3) logo_pts(1, 3)];
    # % ay3 = [0 0 0 - video_pts(1, 3) - video_pts(2, 3) - 1 video_pts(1, 3) * logo_pts(2, 3)
    #          video_pts(2, 3) * logo_pts(2, 3) logo_pts(2, 3)];
    # %
    # % ax4 = [-video_pts(1, 4) - video_pts(2, 4) - 1 0 0 0 video_pts(1, 4) * logo_pts(1, 4)
    #          video_pts(2, 4) * logo_pts(1, 4) logo_pts(1, 4)];
    # % ay4 = [0 0 0 - video_pts(1, 4) - video_pts(2, 4) - 1 video_pts(1, 4) * logo_pts(2, 4)
    #          video_pts(2, 4) * logo_pts(2, 4) logo_pts(2, 4)];
    #
    # % A = [ax1;
    # ay1;
    # ax2;
    # ay2;
    # ax3;
    # ay3;
    # ax4;
    # ay4];
    #
    # [U S V] = svd(A);
    # h = V(:, 9);
    # H = reshape(h, 3, 3)
    # '


    ##### STUDENT CODE END #####
    
    return H