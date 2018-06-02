import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from skimage.feature import hog
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import metrics
import pickle
import os
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

def load_data():
    # load the sample of cars and noncars to train a classifier
    cars, notcars = [], []
    label_cars, label_notcars = [], []
    images_cars = glob.glob('./vehicles/*/*.png')
    for img in images_cars:
        cars.append(img)
        label_cars.append(1)
    image_notcars = glob.glob('./non-vehicles/*/*.png')
    for img in image_notcars:
        notcars.append(img)
        label_notcars.append(0)
    example = np.random.randint(min(len(cars), len(notcars)))
    plt.subplot(121)
    plt.imshow(mpimg.imread(cars[example]))
    plt.title('Car example')
    plt.subplot(122)
    plt.imshow(mpimg.imread(notcars[example]))
    plt.title('Not Car example')
    # plt.show()
    print('Cars shape', np.array(cars).shape, ',not cars shape', np.array(notcars).shape)
    print('Picture shape:',mpimg.imread(cars[example]).shape)
    return cars,notcars,label_cars,label_notcars


#feature extraction by HOG
def get_hog_features(image,orient,pix_per_cell,cell_per_block,vis=False,feature_vec=True):
    if vis:
        features,hog_image=hog(image,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
                               cells_per_block=(cell_per_block,cell_per_block), block_norm= 'L2-Hys',
                               transform_sqrt=True,visualise=vis,feature_vector=feature_vec)
        return features,hog_image
    else:
        features=hog(image,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
                     cells_per_block=(cell_per_block,cell_per_block), block_norm= 'L2-Hys',
                     transform_sqrt=True,visualise=vis,feature_vector=feature_vec)
        return features


def extract_hog_features(images,orient=9,pix_per_cell=9,cell_per_block=2,hog_channel=0,vis=False):
    features=[]
    for image in images:
        image_feature=image.copy()
        if hog_channel=='ALL':
            hog_features=[]
            for channel in range(image_feature.shape[2]):
                if vis:
                    hogfeatures, hog_image=get_hog_features(image_feature[:,:,channel],orient,pix_per_cell,cell_per_block,vis=vis,feature_vec=True)
                    hog_features.append(list(hogfeatures))
                    plt.subplot(121)
                    plt.imshow(image)
                    plt.title('Original')
                    plt.subplot(122)
                    plt.imshow(hog_image)
                    plt.title('HOG')
                    plt.show()
                else:
                    hogfeatures=get_hog_features(image_feature[:,:,channel],orient,pix_per_cell,cell_per_block,vis=vis,feature_vec=True)
                    hog_features.append(hogfeatures)
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(image_feature[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                            vis=vis,feature_vec=True)
        features.append(hog_features)
    return features


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_svm_features(image,conv,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins):
    image_features = image.copy()
    image_features = image_features.astype(np.float32)/ 255
    image_svm_features = convert_color(image_features, conv=conv)
    hog_feat1 = get_hog_features(image_svm_features[:,:,0],orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,feature_vec=False).ravel()
    hog_feat2 = get_hog_features(image_svm_features[:,:,1],orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,feature_vec=False).ravel()
    hog_feat3 =get_hog_features(image_svm_features[:,:,2],orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,feature_vec=False).ravel()
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    spatial_features = bin_spatial(image, size=spatial_size)
    hist_features = color_hist(image, nbins=hist_bins)
    return spatial_features, hist_features, hog_features

def hog_svm(x,y,conv,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins):
    spatial_features,hist_features,hog_features=[],[],[]
    for image in x:
        image=mpimg.imread(image)
        spatial_feature, hist_feature, hog_feature=extract_svm_features(image,conv=conv,orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,spatial_size=spatial_size,hist_bins=hist_bins)
        spatial_features.append(spatial_feature)
        hist_features.append(hist_feature)
        hog_features.append(hog_feature)
    X_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(len(x), -1)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X_features), np.array(y), test_size=0.2)
    print(X_train.shape,y_train.shape)
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    svc = LinearSVC(C=1)
    svc.fit(X_train, y_train)
    y_test_pred = svc.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_test_pred)
    print("SVM accuracy:",acc_score)
    print('Train Accuracy of SVC=', round(svc.score(X_train, y_train), 4))
    print('Test Accuracy of SVC=', round(svc.score(X_test, y_test), 4))
    with open('svm.pickle', 'wb') as fw:
        pickle.dump(svc, fw)
    with open('scaler.pickle','wb') as sc:
        pickle.dump(X_scaler,sc)
    return X_scaler,svc

def hog_parameter(cars,notcars):
    for orient in range(7, 17, 2):
        for pix_per_cell in range(7, 17, 2):
            for cell_per_block in range(2, 4):
                acc = hog_svm(cars, notcars, orient, pix_per_cell, cell_per_block)
                print('orient:', orient, 'pix_per_cell:', pix_per_cell, 'cell_per_block:', cell_per_block, 'accuracy:',
                      acc)


def sliding_window(img,x_start_stop=[None,None],y_start_stop=[None,None],xy_window=(64,64),xy_overlap=(0.5,0.5)):
    if x_start_stop[0]==None:
        x_start_stop[0]=0
    if x_start_stop[1]==None:
        x_start_stop[1]=img.shape[1]
    if y_start_stop[0]==None:
        y_start_stop[0]=0
    if y_start_stop[1]==None:
        y_start_stop[1]=img.shape[0]
    xpan=x_start_stop[1]-x_start_stop[0]
    ypan=y_start_stop[1]-y_start_stop[0]
    nx_pix_per_step=int(xy_window[0]*(1-xy_overlap[0]))
    ny_pix_per_step=int(xy_window[1]*(1-xy_overlap[1]))
    nx_buffer=int(xy_window[0]*xy_overlap[0])
    ny_buffer=int(xy_window[1]*xy_overlap[1])
    nx_windows=int((xpan-nx_buffer)/nx_pix_per_step)
    ny_windows=int((ypan-ny_buffer)/ny_pix_per_step)
    window_list=[]
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx=xs*nx_pix_per_step+x_start_stop[0]
            endx=startx+xy_window[0]
            starty=ys*ny_pix_per_step+y_start_stop[0]
            endy=starty+xy_window[1]
            window_list.append(((startx,starty),(endx,endy)))
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def get_heat_map(image, bbox_list,threshold):
    """Computes heat map of hot windows. Puts all specified
    hot windows on top of each other, so every pixel of returned image will
    contain how many hot windows covers this pixel
    Args:
        image (numpy.array): image
    Returns:
        heatmap (numpy.array) grayscale image of the same size as input image
    """
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img



def process_image(image):
    if os.path.exists('svm.pickle'):
        with open('svm.pickle', 'rb') as fr:
            svc = pickle.load(fr)
        with open('scaler.pickle','rb') as sc:
            X_scaler=pickle.load(sc)
    else:
        cars, notcars, label_cars, label_notcars = load_data()
        x = cars + notcars
        y = label_cars + label_notcars
        x, y = shuffle(x, y)
        X_scaler, svc=hog_svm(x, y, conv='RGB2YCrCb', orient=8, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32),
                hist_bins=32)

    image_test = image.copy()
    windowlist1 = sliding_window(image_test, x_start_stop=[700, None], y_start_stop=[390, 500], xy_window=(110,110),
                                 xy_overlap=(0.75, 0.75))

    windowlist2 = sliding_window(image_test, x_start_stop=[700, None], y_start_stop=[380, 550], xy_window=(170, 170),
                                 xy_overlap=(0.75, 0.75))

    windowlist3 = sliding_window(image_test, x_start_stop=[700, None], y_start_stop=[390,500], xy_window=(64, 64),
                                 xy_overlap=(0.75, 0.75))
    windowlist = windowlist1 +windowlist2+ windowlist3
    on_windows = []
    for window in windowlist:
        test_img = cv2.resize(image_test[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        spatial_feature, hist_feature, hog_feature = extract_svm_features(test_img, conv='RGB2YCrCb', orient=8,
                                                                          pix_per_cell=8, cell_per_block=2,
                                                                          spatial_size=(32, 32),
                                                                          hist_bins=32)
        features_ = X_scaler.transform(np.hstack((spatial_feature, hist_feature, hog_feature)).reshape(1, -1))
        prediction = svc.predict(features_)
        if prediction == 1:
            on_windows.append(window)
    box_img = draw_boxes(image_test, on_windows, color=(0, 0, 255), thick=3)
    heatmap = get_heat_map(image_test, on_windows, threshold=3)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image_test), labels)
    return draw_img


def main():

    images_cars = glob.glob('./test_images/*.jpg')
    for i,img in enumerate(images_cars):
        img=mpimg.imread(img)
        draw_img = process_image(img)
        plt.subplot(3,2,i+1)
        plt.imshow(draw_img)
        #plt.subplot(6,2,2*i+1)
        #plt.imshow(box)
        #plt.subplot(6,2,2*i+2)
        #plt.imshow(draw_img)
        #plt.subplot(3,2,i+1)
        #plt.imshow(labels[0], cmap='gray')
    plt.savefig("result.png",dpi=1000)
    plt.show()

    out_dir = './output_images/'
    inpfile = 'test_video.mp4'
    outfile = out_dir + 'processed_' + inpfile
    clip = VideoFileClip(inpfile)
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(outfile, audio=False)

if __name__ == '__main__':
    main()


