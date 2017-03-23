import numpy as np
import math
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

"""
The Camera class has methods to calibrate camera, and undistort an image using the calibrated values.
"""

class Camera(object):
    def __init__(self):
        self.pickle_fname='calib.pickle'
        self.mtx=None
        self.dist=None
        pass

    def get_pickle_file(self,mode):
        try:
            fileobj = open(self.pickle_fname, mode)
            print("The Calibration Pickle file exits.")
            # file exists! just return true
            fileobj.close()
            return True
        except:
            return False

    def calibrate_camera(self,image_path,nx,ny):
        objpoints = []
        imgpoints = []
        gray_shape=None
        sample=None
        mtx=None
        dist=None
        # Check if pickle file exits
        if self.get_pickle_file('rb')==True:
            return True;
        images=glob.glob(image_path)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        for fname in images:
            print(fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if sample is None:
                sample=img
                gray_shape=gray.shape[::-1]
            ret, corners =cv2.findChessboardCorners(gray,(nx,ny),None)
            if ret==True:
                objpoints.append(objp)
                imgpoints.append(corners)
                ret,mtx, dist,rvecs,tvecs =cv2.calibrateCamera(objpoints,imgpoints,gray_shape,None,None)
        fileobj = open(self.pickle_fname, 'wb')
        pickle.dump([mtx,dist], fileobj)
        fileobj.close()
        return True

    def undistort_image(self,img):
        if self.mtx==None:
            try:
                fileobj = open(self.pickle_fname, 'rb')
                self.mtx, self.dist = pickle.load(fileobj)
                fileobj.close()
            except:
                return None
        undistorted=cv2.undistort(img,self.mtx,self.dist,None,self.mtx)
        return undistorted

    def plot_and_save_calibration_images(self,img1,img2):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(cv2.imread(img1))
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(img2)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        f.savefig("output_images/calibration_image.png")

"""
The Image Processor class provides methods to apply gradient thresholds, gradient magnitude threshold and HLS thresholds.
The apply_threshold method return the thresholded binary Image with the lanes lines clearly visible
"""
class ImageProcessor():
    def __init__(self,color_min,color_max,hls_min,hls_max,grad_mag_min,grad_mag_max,sobel_kernel=5):
        self.image=None
        self.hls_min=hls_min
        self.hls_max=hls_max
        self.grad_min=grad_mag_min
        self.grad_max=grad_mag_max
        self.sobel_kernel=sobel_kernel
        self.color_min=color_min
        self.color_max=color_max

    def apply_gradient_threshold(self):
        gray_image=cv2.cvtColor(self.image.copy(),cv2.COLOR_RGB2GRAY)
        sobelx=cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.grad_min) & (scaled_sobel <= self.grad_max)] = 1
        return sxbinary

    def apply_gradient_mag_threshold(self):
        gray_image=cv2.cvtColor(self.image.copy(),cv2.COLOR_RGB2GRAY)
        sobelx=cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        grad_mag=np.sqrt(sobelx**2,sobely**2)
        scale_factor = np.max(grad_mag) / 255
        gradmag = (grad_mag / scale_factor).astype(np.uint8)
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= self.grad_min) & (gradmag <= self.grad_max)] = 1
        return mag_binary

    def apply_hls_threshold(self):
        hls_image= cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2HLS)
        s_channel=hls_image[:,:,2]
        s_binary=np.zeros_like(s_channel)
        s_binary[(s_channel> self.hls_min)& (s_channel<=self.hls_max)]=1
        return s_binary

    def apply_rgb_threshold(self):
        #Use only the R Channel
        r_channel =  self.image[:,:,0]
        s_binary=np.zeros_like(r_channel)

        s_binary[(r_channel >= self.color_min)&(r_channel<=self.color_max)]=1
        return s_binary

    def apply_threshold(self,image):
        self.image=image
        grad_thresh_image=self.apply_gradient_threshold()
        rgb_thresh_image=self.apply_rgb_threshold()
        hls_thresh_image=self.apply_hls_threshold()
        #self.plot_and_save_images(self.image,"original",grad_thresh_image,"magnitude threshold","grad_threshold.png")
        #self.plot_and_save_images(self.image, "original", rgb_thresh_image, "Red threshold", "red_threshold.png")
        #self.plot_and_save_images(self.image, "original", hls_thresh_image, "hls threshold", "mag_threshold.png")
        color_binary=np.dstack((rgb_thresh_image,grad_thresh_image,hls_thresh_image))
        combined_binary = np.zeros_like(grad_thresh_image)
        combined_binary[(grad_thresh_image == 1) | (rgb_thresh_image == 1)|(hls_thresh_image==1)] = 1
        #self.plot_and_save_images(self.image, "original", combined_binary, "Final Thresholded Image", "final_threshold.png")
        return combined_binary

    def plot_and_save_images(self,img1,title1,img2,title2,filename):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img1,cmap='gray')
        ax1.set_title(title1, fontsize=30)
        ax2.imshow(img2,cmap='gray')
        ax2.set_title(title2, fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        f.savefig('output_images/'+filename)

"""
    The Lane class is a utility class that is used to store the lane information
"""
class Lane():
    def __init__(self,frames_to_buffer):
        self.fit_buffer=[]
        self.fit_averaged=[]
        self.fitted_x=None

        self.allx=[]
        self.ally=[]
        self.radius_of_curvature=None
        self.frames_to_buffer=frames_to_buffer



    def populate_lane_object(self, x, ploty, fitx):
        if len(self.fit_buffer) >=self.frames_to_buffer:
            self.fit_buffer.pop(0) # FIFO Queue. Remove the first entry before adding the new entry
            self.allx.pop(0)
            self.ally.pop(0)
        self.fit_buffer.append(fitx)
        self.allx.append(x)
        self.ally.append(ploty)
        self.ploty=ploty
        fit_array = np.asarray(self.fit_buffer)
        self.fit_averaged = np.sum(fit_array, axis=0) / len(fit_array)

    def get_fitted_line(self):
        #ploty = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        self.fitted_x= self.fit_averaged[0] * self.ploty ** 2 + self.fit_averaged[1] * self.ploty + self.fit_averaged[2]
        return self.fitted_x

"""
The LaneFinder class returns the polynomial fit of the left and right lanes and their corresponding x and y coordinates
"""

class LaneFinder():

    def __init__(self, no_of_windows=9,margin=100,minpix=50):
        self.nwindows=no_of_windows
        self.margin=margin
        self.minpix=minpix
        self.left_fit=[]
        self.right_fit=[]
        self.start_frame=True
        self.binary_image=None
        self.Minv=None
        self.prev_left_curve=None
        self.prev_right_curve=None

    def __change_prespective(self):
        src = np.float32([[280, 665], [1035, 665], [520, 500], [780, 500]])
        dst = np.float32([[205, 665], [1075, 665], [200, 470], [1105, 470]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        img_size = (self.binary_image.shape[1], self.binary_image.shape[0])
        self.binary_warped = cv2.warpPerspective(self.binary_image, self.M, img_size, flags=cv2.INTER_LINEAR)
        self.Minv=cv2.getPerspectiveTransform(dst, src)
        #self.plot_and_save_images(self.binary_image,"Binary Image",self.binary_warped,"Binary Warped","Binary_Warped.png")

    def __get_base_pts_from_histogram(self,binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base,rightx_base


    def __run_sanity_check(self,ploty,left_fitx,right_fitx,left_fit,right_fit):
       # Distance Check. Check to see if maximum deviation between the two lines is not more than 10% of the mean
        line_distance=right_fitx-left_fitx
        max_distance= np.max(line_distance)
        min_distance= np.min(line_distance)
        mean_distance=np.mean(line_distance)
        diff=max_distance-min_distance
        if (diff >= mean_distance*0.2):
            print("Horizontal distance validation: The distance betweem max and min is greater than 10% of the mean.")
            print("Mean distance=", mean_distance, " Difference Max and Min Distances =",diff )
            return False

        # Slope test . The average rate of change of the slope of the two lanes should be identical (Secant Lines)
        y_max=np.max(ploty)
        y_min=np.min(ploty)
        y_max_ind=np.argmax(ploty)
        y_min_ind=np.argmin(ploty)
        leftx_at_y_max=left_fitx[y_max_ind]
        leftx_at_y_min=left_fitx[y_min_ind]
        rightx_at_y_max=right_fitx[y_max_ind]
        rightx_at_y_min=right_fitx[y_min_ind]

        # Left Secant Line
        average_left_slope=(leftx_at_y_max-leftx_at_y_min)/(y_max-y_min)
        # Right Secant Line
        average_right_slope = (rightx_at_y_max - rightx_at_y_min)/ (y_max - y_min)
        #print("Right Slope=", average_right_slope, " Left Slope=", average_left_slope)
        left_angle= math.degrees(average_left_slope)
        right_angle=math.degrees(average_right_slope)
        if(abs(right_angle-left_angle))>=20:
            print("Slope Validation: Difference in the slope of sceant line greater than 20 degrees ")
            print("left_angle=",left_angle, " right=", right_angle)
            return False
        return True

    """
    This method return the polynomial fit of the left and right lane and their corresponding cordinates. The method uses
    histogram with sliding windows to identity the lanes
    """

    def __find_lanes_with_sliding_windows(self):
        print("In find by sliding window")
        leftx_base,rightx_base=self.__get_base_pts_from_histogram(self.binary_warped)
        window_height = np.int(self.binary_warped.shape[0] / self.nwindows)
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped)) * 255
        left_lane_inds = []
        right_lane_inds = []
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        for window in range(self.nwindows):
            win_y_low = self.binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = self.binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # CODE TO VISUALIZE LANE
        """

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx,ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        """
        if (self.__run_sanity_check(ploty,left_fitx,right_fitx,left_fit,right_fit) == True):
            return left_fit, right_fit,leftx,rightx
        else:
            return None,None,None,None

    """
    This method return the polynomial fit of the left and right lane and their corresponding cordinates. The method
    extrapolates the starting pointing provided to it an input argument to find the lanes.
    """
    def __find_lanes_by_extrapolation(self,left_fit,right_fit):
        #print("In find by extrapolation", left_fit)
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - self.margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + self.margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - self.margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + self.margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # CODE TO VISUALIZE THE LANES AND SLIDING WINDOWS
        """
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        """
        if (self.__run_sanity_check(ploty,left_fitx,right_fitx,left_fit,right_fit) == True):
            return left_fit, right_fit,leftx,rightx
        else:
            return self.__find_lanes_with_sliding_windows()

    """
    This method returns a tuple containing the polynomial fit of the left lane, right lane and their corresponding X and Y cordinates
    """
    def find_lanes(self,binary_image):
        self.binary_image=binary_image
        self.__change_prespective()
        ploty = np.linspace(0, self.binary_image.shape[0] - 1, self.binary_image.shape[0])
        if(self.start_frame==True):
            left_fit,right_fit,leftx,rightx=self.__find_lanes_with_sliding_windows()
            if(left_fit!=None):
                self.left_fit=left_fit
                self.right_fit=right_fit
                self.leftx=leftx
                self.rightx=rightx
                self.start_frame=False
        else:
            left_fit, right_fit, leftx, rightx = self.__find_lanes_by_extrapolation(self.left_fit,self.right_fit)
            if (left_fit != None):
                self.left_fit = left_fit
                self.right_fit = right_fit
                self.leftx = leftx
                self.rightx = rightx
        return self.left_fit, self.right_fit, self.leftx, self.rightx,ploty

    def plot_and_save_images(self,img1,title1,img2,title2,filename):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img1,cmap='gray')
        ax1.set_title(title1, fontsize=30)
        ax2.imshow(img2,cmap='gray')
        ax2.set_title(title2, fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        f.savefig('output_images/'+filename)


class DrawLanes():

    def __init__(self,camera,frames_to_buffer=5):
        self.frames_to_buffer=frames_to_buffer
        self.right_lane=Lane(frames_to_buffer)
        self.left_lane=Lane(frames_to_buffer)
        self.lane_finder=LaneFinder(9,100,50)
        color_min = 220
        color_max = 250
        hls_min = 180
        hls_max = 255
        grad_min = 20
        grad_max = 100
        sobel_kernel = 3
        self.image_processor=ImageProcessor(color_min=220,color_max=250,hls_min=180,hls_max=255,grad_mag_min=20,grad_mag_max=100,sobel_kernel=5)
        self.camera=camera
        self.ym_per_pix=30/720
        self.xm_per_pix=3.7/806

    def find_radius_of_curvature(self,leftx,rightx,ploty):
        ym_per_pix = self.ym_per_pix  # meters per pixel in y dimension
        xm_per_pix = self.xm_per_pix # meters per pixel in x dimension
        y_eval= np.max(ploty)
        #print("Y Eval=", y_eval)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        return left_curverad,right_curverad

    def render_lanes(self,distorted_frame):
        frame=self.camera.undistort_image(distorted_frame)
        #self.image_processor.plot_and_save_images(distorted_frame, "Original", frame, "Un-Distorted", "Distored_Undistored.png")
        binary_image=self.image_processor.apply_threshold(frame)
        image_center = binary_image.shape[1] / 2
        left_fitx, right_fitx, leftx, rightx, ploty = self.lane_finder.find_lanes(binary_image)
        minv = self.lane_finder.Minv

        if(left_fitx!=None):
            self.right_lane.populate_lane_object(rightx,ploty,right_fitx)
            self.left_lane.populate_lane_object(leftx, ploty, left_fitx)
        else:
            print("Fit is None")

        binary_warped=self.lane_finder.binary_warped
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        avg_fitted_leftx= self.left_lane.get_fitted_line()
        avg_fitted_rightx=self.right_lane.get_fitted_line()
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([avg_fitted_leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([avg_fitted_rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, minv, (binary_warped.shape[1], binary_warped.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
        left_radius,right_radius = self.find_radius_of_curvature(avg_fitted_leftx,avg_fitted_rightx,ploty)
        self.left_lane.radius_of_curvature=left_radius
        self.right_lane.radius_of_curvature=right_radius

        y_max_ind = np.argmax(ploty)
        leftx_at_max = avg_fitted_leftx[y_max_ind]
        rightx_at_max = avg_fitted_rightx[y_max_ind]
        lane_center=(rightx_at_max - leftx_at_max)/2 + leftx_at_max
        offset_in_pix=abs(lane_center-image_center)
        offset_in_mts=offset_in_pix*self.xm_per_pix

        radius_str="Radius="+str(round(left_radius-offset_in_mts,2)) +"(m)"
        offset_str="Vehicle is "+str(round(offset_in_mts,2)) +" left of center"

        cv2.putText(result,radius_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, offset_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return result



if __name__ == '__main__':
    camera = Camera()
    camera.calibrate_camera("camera_cal/calibration*.jpg", 9, 6)
    drawlanes=DrawLanes(camera,frames_to_buffer=12)
    #Read the Project Video File and Write the output to "Project_output.mp4"
    project_output = 'project_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    print("FPS =", clip1.fps)
    out_clip = clip1.fl_image(drawlanes.render_lanes)  # NOTE: this function expects color images!!
    out_clip.write_videofile(project_output, audio=False)




