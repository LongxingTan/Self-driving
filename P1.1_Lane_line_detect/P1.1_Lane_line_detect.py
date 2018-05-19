import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline



import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_lane_x,left_lane_y,right_lane_x,right_lane_y=[],[],[],[]
    left_slope,right_slope=[],[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            if ((y2-y1)/(x2-x1))>0:
                left_lane_x.append(int((x2+x1)/2))
                left_lane_y.append(int((y2+y1)/2))
                left_slope.append((y2-y1)/(x2-x1))
            elif ((y2-y1)/(x2-x1))<0:
                right_lane_x.append(int((x2+x1)/2))
                right_lane_y.append(int((y2+y1)/2))
                right_slope.append((y2-y1)/(x2-x1))
    left_slope_avg = sum(left_slope) / len(left_slope)
    right_slope_avg = sum(right_slope) / len(right_slope)
    cv2.line(img, (int(left_lane_x[1] - (left_lane_y[1] - img.shape[0]) * (1 / left_slope_avg)), img.shape[0]),
             (int(left_lane_x[1] - (left_lane_y[1] - 320) * (1 / left_slope_avg)), 320), color, thickness=5)
    cv2.line(img, (int(right_lane_x[0] - (right_lane_y[0] - img.shape[0]) * (1 / right_slope_avg)), img.shape[0]),
             (int(right_lane_x[0] - (right_lane_y[0] - 320) * (1 / right_slope_avg)),320), color, thickness=5)
    '''
    for i in range(len(left_lane_x)-1):
        cv2.line(img, (left_lane_x[i], left_lane_y[i]), (left_lane_x[i+1], left_lane_y[i+1]), color, thickness)
    for i in range(len(right_lane_x)-1):
        cv2.line(img, (right_lane_x[i], right_lane_y[i]), (right_lane_x[i+1], right_lane_y[i+1]), color, thickness)
    '''





def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """ `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.  """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.c
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


import os
os.listdir("test_images/")

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    image_gray=grayscale(image)
    image_blur=gaussian_blur(image_gray, kernel_size=5)
    image_canny=canny(image_blur, low_threshold=50, high_threshold=150)
    imshape=image_canny.shape
    image_region=region_of_interest(image_canny, vertices=np.array([[(0,imshape[0]),(455,320),(485,320),(imshape[1],imshape[0])]],dtype=np.int32))
    image_hough=hough_lines(image_region, rho=2, theta=np.pi/180, threshold=35, min_line_len=40, max_line_gap=100)
    result=weighted_img(image_hough, image, α=0.8, β=1., γ=0.)
    return result



image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
image=process_image(image)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray'
plt.show()


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds

#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip1.reader.close()

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
