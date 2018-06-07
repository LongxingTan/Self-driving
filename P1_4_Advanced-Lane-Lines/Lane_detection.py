import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import os
from moviepy.editor import VideoFileClip

if os.path.exists('calibration_pickle.p'):
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
else:
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    camera_cal = glob.glob('./camera_cal/*.jpg')
    for idx, fname in enumerate(camera_cal):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    img = cv2.imread('./camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('calibration_pickle.p', 'wb'))

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('calibration_undistort.jpg', dst)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    #plt.show()


def noise_reduction(image, threshold=4):
    """
    This method is used to reduce the noise of binary images.
    :param image:
        binary image (0 or 1)
    :param threshold:
        min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < threshold] = 0
    return image

def binarize(image,gray_thresh=(120,255),s_thresh=(170, 255),l_thresh=(40, 255),sobel_kernal=3):
    img_test=image.copy()

    hls=cv2.cvtColor(img_test,cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2]

    sobelx=cv2.Sobel(l_channel,cv2.CV_64F,1,0,ksize=sobel_kernal)
    abs_sobelx=np.absolute(sobelx)
    scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sobel_x_binary=np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel>=gray_thresh[0])&(scaled_sobel<=gray_thresh[1])]=1

    s_binary=np.zeros_like(s_channel)
    s_binary[(s_channel>=s_thresh[0])&(s_channel<=s_thresh[1])]=1

    l_binary=np.zeros_like(l_channel)
    l_binary[(l_channel>=l_thresh[0])&(l_channel<=l_thresh[1])]=1

    binary=np.zeros_like(l_channel)
    binary[((l_binary==1)&(s_binary==1)|(sobel_x_binary==1))]=1
    binary=255*np.dstack((binary,binary,binary)).astype('uint8')
    return noise_reduction(binary)


class perspective_transformer:
    def __init__(self):
        self.corners = np.float32([[250, 690], [591, 451], [698, 457], [1052, 685]])
        self.new_top_left = np.array([self.corners[0, 0], 0])
        self.new_top_right = np.array([self.corners[3, 0], 0])
        self.offset = [50, 0]
        self.src = np.float32([self.corners[0], self.corners[1], self.corners[2], self.corners[3]])
        self.dst = np.float32([self.corners[0] + self.offset, self.new_top_left + self.offset, self.new_top_right - self.offset, self.corners[3] - self.offset])
        self.M=cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def transformer(self,img):
        warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def inverse_transform(self,img):
        warped = cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped


def lane_detect(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[1] // 2:, :, 0], axis=0)
    #plt.plot(histogram)
    #plt.show()
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(binary_warped.shape[0] / nwindows)
    margin = 100
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    min_num_pixels = 45

    # save pixel ids in these two lists
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - (window) * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > min_num_pixels:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_num_pixels:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            # Concatenate the ndarrays of indices
    left_lane_array = np.concatenate(left_lane_inds)
    right_lane_array = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_array]
    lefty = nonzeroy[left_lane_array]
    rightx = nonzerox[right_lane_array]
    righty = nonzeroy[right_lane_array]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    binary_warped[nonzeroy[left_lane_array], nonzerox[left_lane_array]] = [255, 0, 0]
    binary_warped[nonzeroy[right_lane_array], nonzerox[right_lane_array]] = [0, 0, 255]
    '''
    plt.imshow(binary_warped)
    plt.plot(fit_leftx, fity, color='yellow')
    plt.plot(fit_rightx, fity, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()'''
    return binary_warped,fit_leftx,fit_rightx,fity

def fill_lane_lines(image, fit_left_x, fit_right_x):
    copy_image = np.zeros_like(image)
    fit_y = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])
    pts_left = np.array([np.transpose(np.vstack([fit_left_x, fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, fit_y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))
    return copy_image

def merge_images(binary_img, src_image):
    copy_binary = np.copy(binary_img)
    copy_src_img = np.copy(src_image)
    #perspective_transformer=perspective_transformer()
    copy_binary_pers = perspective_transformer.inverse_transform(copy_binary)
    result = cv2.addWeighted(copy_src_img, 1, copy_binary_pers, 0.3, 0)
    return result

class line_detect():
    def __init__(self):
        self.MAX_BUFFER_SIZE = 10
        self.buffer_index = 0
        self.iter_counter = 0
        self.buffer_left = np.zeros((self.MAX_BUFFER_SIZE, 720))
        self.buffer_right = np.zeros((self.MAX_BUFFER_SIZE, 720))

    def get_road_info(self,ave_left,ave_right,ploty):
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        left_fit_cr = np.polyfit(ploty * ym_per_pix, ave_left * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, ave_right* xm_per_pix, 2)
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5)/np.absolute(2 * left_fit_cr[0])
        right_curverad=((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5)/np.absolute(2 * right_fit_cr[0])
        return left_curverad,right_curverad


    def process(self,image):
        img = image.copy()
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        # perspective_transformer=perspective_transformer()
        warped_img = perspective_transformer.transformer(undist_img)
        binary_warped_img = binarize(warped_img, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255))
        binary_warped_lane, fit_leftx, fit_rightx, fity = lane_detect(binary_warped_img)

        self.buffer_left[self.buffer_index] = fit_leftx
        self.buffer_right[self.buffer_index] = fit_rightx

        self.buffer_index += 1
        self.buffer_index %= self.MAX_BUFFER_SIZE

        if self.iter_counter < self.MAX_BUFFER_SIZE:
            self.iter_counter += 1
            ave_left = np.sum(self.buffer_left, axis=0) / self.iter_counter
            ave_right = np.median(self.buffer_right,axis=0)
        else:
            ave_left = np.average(self.buffer_left, axis=0)
            ave_right = np.median(self.buffer_right,axis=0)
        left_curvature, right_curvature = self.get_road_info(ave_left,ave_right,fity)
        curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curvature, right_curvature)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, curvature_text, (100, 50), font, 1, (221, 28, 119), 2)
        filled_image = fill_lane_lines(binary_warped_img, ave_left, ave_right)
        merged_image = merge_images(filled_image, img)
        return merged_image




perspective_transformer = perspective_transformer()
line_detect=line_detect()
'''
images = glob.glob('./test_images/*.jpg')
for i, image in enumerate(images):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    merged_image =line_detect.process(image)
    plt.subplot(4, 2, i + 1)
    plt.imshow(merged_image)
    # plt.plot(fit_leftx, fity, color='yellow')
    # plt.plot(fit_rightx, fity, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
plt.show()
'''
output_file = 'processed_project_video.mp4'
input_file = 'project_video.mp4'
clip = VideoFileClip(input_file)
out_clip = clip.fl_image(line_detect.process)
out_clip.write_videofile(output_file, audio=False)









