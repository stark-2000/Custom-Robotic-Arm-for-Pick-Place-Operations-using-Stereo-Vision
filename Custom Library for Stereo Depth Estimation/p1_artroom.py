#Dependencies: numpy, cv2, matplotlib, random, PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


##class for stereo depth estimation:
    #Fundamental matrix estimation
    #Esential matrix estimation
    #Camera pose estimation
    #Epilines calc & plotting
    #Stereo Rectification
    #Correspondence matching
    #Disparity map generation
    #Depth map generation
    #Heatmap generation
class stereo_depth_estimation:
    def __init__(self):
        #Calibration matrices of left and right cameras:
        self.kL = np.array([
                    [1733.74, 0      , 792.27],
                    [0      , 1733.74, 541.89],
                    [0      , 0      , 1]
                ])
        
        self.kR = np.array([
                    [1733.74, 0      , 792.27], 
                    [0      , 1733.74, 541.89], 
                    [0      , 0      , 1]
                ])
        
        self.baseline = 536.62 #baseline length in mm
        self.focal_length = 1733.74 #focal length in px

        image_path=['../Dataset/artroom/im0.png','../Dataset/artroom/im1.png'] #Path of the images
        self.imgs = []
        for i in range(len(image_path)): #Reading the images
            self.imgs.append(cv2.imread(image_path[i]))


    #Function for feature detection:
        #Input: image
        #Output: keypoints, descriptors
    def FeaDet_sift(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        sift = cv2.SIFT_create() 
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des


    #Function for feature matching:
        #Input: descriptors of two images
        #Output: matches
    def FeaMat_KNN(self, des1, des2):
        #Flann based matcher parameters:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good_matches = [] #List of good matches
        
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.5*n.distance:
                good_matches.append([m]) 
        
        return good_matches
    

    #Function to normalize feature points:
        #Input: matched points of one image
        #Output: normalized points
    def normalize_pts(self, point_set1):
        mean_x = np.mean(point_set1[:,0]) #mean of x coordinates
        mean_y = np.mean(point_set1[:,1]) #mean of y coordinates
        std_x = np.std(point_set1[:,0]) #standard deviation of x coordinates
        std_y = np.std(point_set1[:,1]) #standard deviation of y coordinates
        
        #Scale matrix:
        V_matrix = np.array([
            [1/std_x, 0, 0],
            [0, 1/std_y, 0],
            [0, 0, 1]
        ])

        #Offset matrix:
        m_matrix = np.array([
            [1, 0, -mean_x],
            [0, 1, -mean_y],
            [0, 0, 1]
        ])

        #Transformation matrix:
        T_matrix = np.dot(V_matrix, m_matrix)

        #Normalizing the points by multiplying with the transformation matrix:
        for i in range(len(point_set1)):
            point = np.append(point_set1[i], 1).reshape(3,1)
            point = np.dot(T_matrix, point).reshape(1,3)
            point_set1[i] = point[0][0:2]

        return point_set1, T_matrix
    


    ##Formula: x2^T * F * x1 = 0
    #Function to estimate fundamental matrix:
        #Input: matched points
        #Output: fundamental matrix
    def fund(self, hypo_inliers):
        A = []
        for i in range(hypo_inliers.shape[0]): #for each point
            imgL = np.append(hypo_inliers[i][0:2], 1) #left image points
            imgR = np.append(hypo_inliers[i][2:4], 1) #right image points
            row1 = [imgL[0]*imgR[0], imgL[0]*imgR[1], imgL[0], imgL[1]*imgR[0], imgL[1]*imgR[1], imgL[1], imgR[0], imgR[1], 1]
            A.append(row1)

        A = np.array(A) #A is the matrix of the equations
        U, S, V = np.linalg.svd(A) 
        H = V[-1, :].reshape(3, 3) #last column of V is the solution to the fundgraphy matrix
        return H
    

    #Function to select random points for initial iniliers in RANSAC:
        #Input: matched points, number of points to be chosen
        #Output: random points
    def obtain_random_point(self, mat_pts, n): #n is the number of points to be chosen
        choice = random.sample(range(len(mat_pts)), n) #randomly choose n points
        point = [mat_pts[i] for i in choice ] #obtain the points
        return np.array(point)


    #Function to check if points are inliers:
        #Input: matched points, fundamental matrix, threshold
        #Output: inliers
    def alsoInliers(self, mat_pts, H, threshold): 
        inliers = [] #list of inliers
        for i in range(len(mat_pts)): #for each point
            p1 = np.array([mat_pts[i][0], mat_pts[i][1], 1]).reshape(1,3) #left image points
            p2 = np.array([mat_pts[i][2], mat_pts[i][3], 1]).reshape(3,1) #right image points

            temp = np.dot(p1, H).reshape(1, 3) #temp = x2^T * F
            temp2 = np.dot(temp, p2) #temp2 = x2^T * F * x1

            if temp2 < threshold: #if error is less than threshold
                inliers.append(mat_pts[i]) #append the point to the list of inliers

        return np.array(inliers)
    

    #Function to perform RANSAC:
        #Input: matched points, maximum number of iterations, threshold, probability of obtaining inliers
        #Output: best inliers, best fundamental matrix
    def ransac(self, mat_pts, max_iterations, threshold, fit_prob):
        max_count = 0 #maximum number of inliers obtained
        
        for i in range(0, max_iterations): #for each iteration
            hypo_inliers = self.obtain_random_point(mat_pts, 8) #randomly choose 8 points
            F_matrix = self.fund(hypo_inliers) #obtain the fundgraphy matrix
            obtained_inliers = self.alsoInliers(mat_pts, F_matrix, threshold) #obtain the inliers

            if np.linalg.matrix_rank(F_matrix) < 3: #if the rank of the f_matrix is less than 3
                continue #continue to the next iteration

            if len(obtained_inliers) > max_count: #if the number of inliers obtained is greater than the maximum number of inliers obtained
                best_inliers = obtained_inliers.copy() #update the best inliers
                max_count = len(obtained_inliers) #update the maximum number of inliers obtained
                best_F_matrix = self.fund(best_inliers) #update the best fund matrix
            
            if (max_count/len(mat_pts)) > fit_prob: #if the number of inliers obtained is greater than the probability of fitting
                break 
        
        return best_F_matrix, np.array(best_inliers)
    

    #Function to unnormalize the fundamental matrix:
        #Input: fundamental matrix, transformation matrix of left image, transformation matrix of right image
        #Output: unnormalized fundamental matrix
    def unnormalize_fund_mat(self, F, T1, T2):
        return np.dot(T2.T, np.dot(F, T1)) #F = T2^T * F * T1


    #Function to unnormalize the points:
        #Input: points/inliers, transformation matrix of left image, transformation matrix of right image
    def unnormalize_pts(self, inliers, T1, T2):
        pts1 = np.zeros((len(inliers), 2))
        pts2 = np.zeros((len(inliers), 2))

        for i in range(len(inliers)):
            pts1[i] = inliers[i][0:2]
            pts2[i] = inliers[i][2:4]

            #Formula: x = T^-1 * x'
            pts1[i] = np.dot(np.linalg.inv(T1), np.append(pts1[i], 1).reshape(3,1)).reshape(1,3)[0][0:2] 
            pts2[i] = np.dot(np.linalg.inv(T2), np.append(pts2[i], 1).reshape(3,1)).reshape(1,3)[0][0:2]

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        return pts1, pts2, np.concatenate((pts1, pts2), axis=1) 
    

    #Function to enforce rank 2 constraint:
        #Input: fundamental matrix
        #Output: fundamental matrix with rank 2
    def make_rank_2(self,mat):
        U, S, V = np.linalg.svd(mat) #SVD of fundamental matrix
        S = np.diag(S) #diagonal matrix of singular values

        if (S[0, 0] <= S[1, 1] and S[0, 0] <= S[2, 2]):
            S[0, 0] = 0
        
        elif (S[1, 1] <= S[0, 0] and S[1, 1] <= S[2, 2]): 
            S[1, 1] = 0
        
        else:
            S[2, 2] = 0 #make the smallest singular value zero

        new_mat = np.dot(U, np.dot(S, V)) #new fundamental matrix 
        new_mat = new_mat/new_mat[2, 2] #normalize the new fundamental matrix
        return new_mat
    

    #Function to rectify the images with epipolar lines:
        #Input: left image with epilines, right image with epilines, fundamental matrix, points in left image, points in right image
        #Output: rectified stereo image, rectified left image, rectified right image, transformation matrix of left image, transformation matrix of right image
    def rectify_epi_image(self, left_image, right_image, Fund_mat, pts1, pts2):
        img_size = (left_image.shape[1], left_image.shape[0])

        #Obtain the transformation matrix of left and right image
        _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, Fund_mat, img_size) 

        #Rectify the images
        left_rectified = cv2.warpPerspective(left_image, H1, img_size)
        right_rectified = cv2.warpPerspective(right_image, H2, img_size)
        opencv_rectified = cv2.hconcat([left_rectified, right_rectified])

        return opencv_rectified, left_rectified, right_rectified, H1, H2
    

    #Function to calc essential matrix:
        #Input: fundamental matrix, intrinsic matrix of left image, intrinsic matrix of right image
        #Output: essential matrix
    def essential_mat(self,kl,kr,fund_mat):
        #Formula: E = K1^T * F * K2
        temp = np.dot(kl.T,fund_mat) #temp = K1^T * F
        essen_mat = np.dot(temp, kr) #E = temp * K2
        return essen_mat


    #Function to decompose the essential matrix & obtain the camera pose:
        #Input: essential matrix
        #Output: 4 possible camera pose
    def cam_pose(self,essen_mat):
        U, S, V = np.linalg.svd(essen_mat) #SVD of essential matrix

        #Formula: U * W * V^T
        W = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
        ])

        temp = np.dot(U, W)
        temp1 = np.dot(U, W.T)

        C1 = U[:,2] 
        R1 = np.dot(temp, V) #R1 = U * W * V^T

        C2 = -U[:,2]
        R2 = R1

        C3 = C1
        R3 = np.dot(temp1, V) #R3 = U * W^T * V^T

        C4 = C2
        R4 = np.dot(temp1, V)

        return C1, C2, C3, C4, R1, R2, R3, R4
    


    #Function to perform square sum of difference between two images:
        #Input: left image, right image, kernel/moving window size, maximum offset/pixel disparity
        #Output: disparity map
    def Correspondence_SSD(self, left_img, right_img, kernel, max_offset):
        #Assumptions:
            #1. Both images are of same size
            #2. Both images are in RGBA format 8bit per channel

        left_img = Image.open(left_img).convert('L')
        left = np.asarray(left_img)
        right_img = Image.open(right_img).convert('L')
        right = np.asarray(right_img)    
        w, h = left_img.size #same size used for both images
        
        disparity = np.zeros((w, h), np.uint8) #Empty disparity map to be filled in later
        disparity.shape = h, w
        
        kernel_half = int(kernel / 2) #half of kernel size used to iterate through the image
        offset_adjust = 255 / max_offset  #Map the disparity values to 0-255 range
        
        #Iterate through the image - column wise
        for y in range(kernel_half, h - kernel_half):     
            #Print the progress as percentage         
            print("\rProcessing.. %d%% complete"%(y / (h - kernel_half) * 100), end="", flush=True)        

            #Iterate through the image - row wise
            for x in range(kernel_half, w - kernel_half):
                best_offset = 0 #initialise the best offset to 0
                prev_ssd = 65534 #initialise the previous ssd to a very large number
                
                #Iterate through the image - offset wise
                for offset in range(max_offset):               
                    ssd = 0 #initialise the ssd to 0
                    ssd_temp = 0 #initialise the ssd_temp to 0                      
                    
                    #v and u are the (x,y) of moving window search, as just squared differences of 
                    #two pixels alone is not enough to find the best match
                    for v in range(-kernel_half, kernel_half):
                        for u in range(-kernel_half, kernel_half):
                            #iteratively sum the sum of squared differences value for this block
                            #left[] and right[] are arrays of uint8, so converting them to int saves
                            #potential overflow
                            ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                            ssd += ssd_temp * ssd_temp              
                    
                    #if this value is smaller than the previous ssd at this block
                    #then it is a closer match. Store this value against this block
                    if ssd < prev_ssd:
                        prev_ssd = ssd
                        best_offset = offset
                                
                #set depth output for this x,y location to the best match
                #and convert to 0-255 range
                disparity[y, x] = best_offset * offset_adjust
                                    
        #Convert to PIL lib and save it
        Image.fromarray(disparity).save('output/artroom/disparity_map_gray.png')


    #Function to convert disparity map to depth map:
        #Input: disparity map
        #Output: depth map
    def disparity_to_depth(self, disparity_map):
        #Formula: Z = baseline * focal_length / disparity_map
        rep = 1/disparity_map
        depth_map = self.baseline * self.focal_length * rep 
        return depth_map


    #Function to display images:
        #Input: image, name of the image
        #Output: display image
    def display_images(self, img, name):
        imgs1 = cv2.resize(img,(0 ,0),fx=0.4,fy=0.4) #Resizing the images
        cv2.imshow(name, imgs1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #Function to display matrices & their properties:
        #Input: matrix, name of the matrix
        #Output: display matrix, rank, determinant
    def matrix_prop(self, mat, name):
        print(name + " matrix: \n", mat)
        print("Rank of " + name + " matrix: " + str(np.linalg.matrix_rank(mat)))
        print("Determinant of " + name + " matrix: " + str(round(np.linalg.det(mat), 5)))
        print("\n")


    #Function to draw matches:
        #Input: combined image, matched points
        #Output: image with lines joining matched points
    def plot_matches(self, matches, total_img):
        match_img = total_img.copy()
        offset = total_img.shape[1]/2
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.imshow(np.array(match_img).astype('uint8')) #RGB is integer type
        
        ax.plot(matches[:, 0], matches[:, 1], 'xr')
        ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
        
        ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
                'r', linewidth=0.5)

        plt.show()


    #Function to draw epipolar lines:
        #Input: left image, right image, epipolar lines, left img points, right img points
        #Output: left image with epipolar lines, right image with epipolar lines
    def drawlines(self,img1,img2,lines,pts1,pts2):
        r,c,g = img1.shape
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)

        return img1,img2




def main():
    p1 = stereo_depth_estimation()
    p1.display_images(p1.imgs[0], 'Original Left Image')
    p1.display_images(p1.imgs[1], 'Original Right Image')

    #####################################################################
    ##Part 1: Feature Detection & Matching
    print("\nStarting Part 1: Feature Detection & Matching...")

    #SIFT Feature Detection & Matching:
    mat_pts = []
    kp1, des1 = p1.FeaDet_sift(p1.imgs[0])
    kp2, des2 = p1.FeaDet_sift(p1.imgs[1])
    features = p1.FeaMat_KNN(des1, des2)

    for m in features: 
        mat_pts.append(list(kp1[m[0].queryIdx].pt) + list(kp2[m[0].trainIdx].pt)) 

    mat_pts = np.array(mat_pts)
    pts1 = mat_pts[:, :2]
    pts2 = mat_pts[:, 2:]
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    #Plotting the matches:
    total_img = np.concatenate((p1.imgs[0], p1.imgs[1]), axis=1)
    p1.plot_matches(mat_pts, total_img)


    #####################################################################
    ##Part 2: Fundamental Matrix Estimation
    print("\nStarting Part 2: Fundamental Matrix Estimation...")

    #Normalizing the points:
    pts1_n, T1 = p1.normalize_pts(pts1)
    pts2_n, T2 = p1.normalize_pts(pts2)
    mat_pts_n = np.concatenate((pts1_n, pts2_n), axis=1)

    #RANSAC:
    Fund_matrix_n, best_mat_pts_n = p1.ransac(mat_pts_n, 1000, 0.01, 0.99) #obtain the fundamental matrix and the best inliers
    
    #Unnormalizing the points and the fundamental matrix:
    best_pts1, best_pts2, best_mat_pts = p1.unnormalize_pts(best_mat_pts_n, T1, T2)
    Fund_matrix = p1.unnormalize_fund_mat(Fund_matrix_n, T1, T2)
    
    #Plotting the matches and fundamental matrix:
    p1.plot_matches(best_mat_pts, total_img) 
    p1.matrix_prop(Fund_matrix, "Fundamental")

    #Enforcing rank 2 constraint:
    Fund_matrix = p1.make_rank_2(Fund_matrix)
    p1.matrix_prop(Fund_matrix, "Rank Enforced Fundamental")

    #Computing the epipolar lines and plotting them:
    epi_lines1 = cv2.computeCorrespondEpilines(best_pts2.reshape(-1,1,2), 2,Fund_matrix)
    epi_lines1 = epi_lines1.reshape(-1,3)
    img1,img2 = p1.drawlines(p1.imgs[0],p1.imgs[1],epi_lines1,best_pts1,best_pts2)

    epi_lines2 = cv2.computeCorrespondEpilines(best_pts1.reshape(-1,1,2), 1,Fund_matrix)
    epi_lines2 = epi_lines2.reshape(-1,3)
    img3,img4 = p1.drawlines(p1.imgs[1],p1.imgs[0],epi_lines2,best_pts2,best_pts1)

    plt.subplot(121),plt.imshow(img1)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


    #####################################################################
    ##Part 3: Stereo Rectification
    print("\nStarting Part 3: Stereo Rectification...")

    rec_combined, rec_left, rec_right, fund_left, fund_right  = p1.rectify_epi_image(img1, img3, Fund_matrix, best_pts1, best_pts2)
    cv2.imwrite('output/artroom/rectified_left_image.png', rec_left)
    cv2.imwrite('output/artroom/rectified_right_image.png', rec_right)

    print("\nLeft Image Rectification homo Matrix: \n", fund_left)
    print("\nRight Image Rectification homo Matrix: \n", fund_right)
    plt.subplot(111),plt.imshow(rec_combined)
    plt.show()


    #####################################################################
    ##Part 4: Essential Matrix & Camera Pose Estimation
    print("\nStarting Part 4: Essential Matrix & Camera Pose Estimation...")

    #Essential Matrix:
    Essen_mat = p1.essential_mat(p1.kL, p1.kR, Fund_matrix)
    print("\nEssential Matrix: \n", Essen_mat)

    #Camera Pose:
    C1, C2, C3, C4, R1, R2, R3, R4 = p1.cam_pose(Essen_mat)
    print("\nCamera Pose 1: \n", C1, "\n", R1)
    print("\nCamera Pose 2: \n", C2, "\n", R2)
    print("\nCamera Pose 3: \n", C3, "\n", R3)
    print("\nCamera Pose 4: \n", C4, "\n", R4)

    
    #####################################################################
    ##Part 5: Stereo Correspondence
    print("\nStarting Part 5: Stereo Correspondence using SSD...")

    #Generate Disparity Map - gray and saving:
    print("Generating Disparity Map...")
    p1.Correspondence_SSD("output/artroom/rectified_left_image.png", "output/artroom/rectified_right_image.png", 6, 30)
    
    #Heat Map Conversion - Disparity Map:
    disparity_img_gray = cv2.imread("output/artroom/disparity_map_gray.png")
    disparity_img_heatmap = cv2.applyColorMap(disparity_img_gray, cv2.COLORMAP_JET)

    #Saving the disparity map - color:
    cv2.imwrite('output/artroom/disparity_map_color.png', disparity_img_heatmap)

    #Displaying the disparity map - gray and color:
    disparity_map = np.concatenate((disparity_img_gray, disparity_img_heatmap), axis=1)
    p1.display_images(disparity_map, 'Disparity Map')
    
    #Generate Depth Map - gray and saving:
    print("Generating Depth Map...")
    depth_map_gray = p1.disparity_to_depth(disparity_img_gray)
    depth_map_gray = depth_map_gray*30/255
    depth_map_gray = depth_map_gray.astype(np.uint8)
    cv2.imwrite('output/artroom/depth_map_gray.png', depth_map_gray)

    #Heat Map Conversion - Depth Map:
    depth_map_heatmap = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_JET)

    #Saving the depth map - color:
    cv2.imwrite('output/artroom/depth_map_color.png', depth_map_heatmap)

    #Displaying the depth map - gray and color:
    depth_map = np.concatenate((depth_map_gray, depth_map_heatmap), axis=1)
    p1.display_images(depth_map, 'Depth Map')


    
if __name__ == '__main__':
    main()