import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Test functions________________________________________________________________________________________________________
def undistorted_test():
    """Test function for undistortion of image"""
    fname = './test_images/test2.jpg'
    # fname = './camera_cal/calibration1.jpg'
    img = cv2.imread(fname)

    undistorted_image = undistorted(image=img)

    cv2.imshow('dist', img)
    cv2.waitKey(10000)
    cv2.imshow('undistorted', undistorted_image)
    cv2.waitKey(10000)


def warp_unwarp_test():
    """Test function for image_warper"""
    images = glob.glob('./test_images/*.jpg')

    for i, fname in enumerate(images):
        image = cv2.imread(fname)
        src_points = origin_image_pts()
        dst_points = destination_image_points(image)

        undist_image = undistorted(image)

        warped_image = image_warper(undist_image)

        src_points_img = add_points(image, src_points)
        src_points_img = add_lines(src_points_img, src_points)
        dst_points_warped = add_points(warped_image, dst_points)
        dst_points_warped = add_lines(dst_points_warped, dst_points)
        cv2.imshow('org', src_points_img)
        cv2.imshow('warped', dst_points_warped)
        cv2.waitKey(10000)


def canny_test():
    """Test function for canny"""
    image = cv2.imread('./test_images/test2.jpg')
    warped_image = image_warper(image=image)
    canny_image = canny_edge(warped_image)
    cv2.imshow('canny image', canny_image)
    cv2.imshow('original', image)
    cv2.waitKey(10000)


def binary_combiner_test():
    """Test function for binary_combiner"""
    img = cv2.imread('./test_images/test1.jpg')

    combined_binary = binary_combiner(image=img, sobel_flag=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_binary, cmap='gray')
    plt.waitforbuttonpress()


def line_fit_calculator_test():
    images = glob.glob('./test_images/test*.jpg')

    # Loop through each image and send it to the pipeline
    for fname in images:
        img = cv2.imread(fname)
        img_og = np.copy(img)

        # Run this through our binary pipeline
        combined_binary = binary_combiner(img_og, sobel_flag=True, clean_flag=True)

        # Run the warped, binary image from the pipeline through the fitter
        left_fit, right_fit, out_img = line_fit_calculator(combined_binary)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image with Source Points')

        ax2.imshow(combined_binary, cmap='gray')
        ax2.set_title('Warped Perspective')

        # Generate x and y values for plotting
        plot_y = np.linspace(0, combined_binary.shape[0] - 1, combined_binary.shape[0])
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        ax3.imshow(out_img)
        ax3.plot(left_fit_x, plot_y, color='yellow')
        ax3.plot(right_fit_x, plot_y, color='yellow')

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.figure(figsize=(9, 9))
        plt.imshow(out_img)
        plt.plot(left_fit_x, plot_y, color='yellow')
        plt.plot(right_fit_x, plot_y, color='yellow')
        plt.waitforbuttonpress()


# UN-DISTORTION_________________________________________________________________________________________________________
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate():
    """Calibrates the camera and returns camera matrix and distance coefficients"""
    global gray
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('./camera_cal/calibration*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    cv2.destroyAllWindows()

    ret, cam_mtx, dist_coeff, rot_vectors, trans_vectors = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                               None, None)
    return cam_mtx, dist_coeff


def undistorted(image):
    """returns the undistorted image un_dist after calibrating the camera"""
    cam_mtx, dist_coeff = calibrate()
    un_dist = cv2.undistort(image, cam_mtx, dist_coeff, None, cam_mtx)
    return un_dist


# WARP-UNWARP___________________________________________________________________________________________________________
def add_points(image, src_image_points):
    """Draw points on the image using src coordinates"""
    point_image = np.copy(image)
    color = [255, 0, 0]  # Red
    thickness = -1
    radius = 15
    x0, y0 = src_image_points[0]
    x1, y1 = src_image_points[1]
    x2, y2 = src_image_points[2]
    x3, y3 = src_image_points[3]
    cv2.circle(point_image, (x0, y0), radius, color, thickness)
    cv2.circle(point_image, (x1, y1), radius, color, thickness)
    cv2.circle(point_image, (x2, y2), radius, color, thickness)
    cv2.circle(point_image, (x3, y3), radius, color, thickness)
    return point_image


def add_lines(image, src_image_points):
    """Draw lines on the image using src coordinates"""
    line_image = np.copy(image)
    color = [255, 0, 0]  # Red
    thickness = 2
    x0, y0 = src_image_points[0]
    x1, y1 = src_image_points[1]
    x2, y2 = src_image_points[2]
    x3, y3 = src_image_points[3]
    cv2.line(line_image, (x0, y0), (x1, y1), color, thickness)
    cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    cv2.line(line_image, (x2, y2), (x3, y3), color, thickness)
    cv2.line(line_image, (x3, y3), (x0, y0), color, thickness)
    return line_image


def origin_image_pts():
    """Returns points for the original image"""
    org_image_pts = np.float32([
        [210, 700],
        [570, 460],
        [705, 460],
        [1075, 700]
    ])
    return org_image_pts


def destination_image_points(image):
    """Returns points for the warped image"""
    h, w = image.shape[:2]
    dst_img_points = np.float32([
        [400, 720],
        [400, 0],
        [w - 400, 0],
        [w - 400, 720]
    ])
    return dst_img_points


def image_warper(image):
    """Returns warped version of image"""
    h, w = image.shape[:2]
    dst_points = np.float32([
        [400, 720],
        [400, 0],
        [w - 400, 0],
        [w - 400, 720]
    ])
    src_points = np.float32([
        [210, 700],
        [570, 460],
        [705, 460],
        [1075, 700]
    ])
    image_size = (image.shape[1], image.shape[0])
    transform_matrix = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
    warped_image = cv2.warpPerspective(image, transform_matrix, image_size, flags=cv2.INTER_NEAREST)

    return warped_image


def image_unwarper(image):
    """Returns unwarped version of image"""
    h, w = image.shape[:2]
    dst_points = np.float32([
        [400, 720],
        [400, 0],
        [w - 400, 0],
        [w - 400, 720]
    ])
    src_points = np.float32([
        [210, 700],
        [570, 460],
        [705, 460],
        [1075, 700]
    ])
    image_size = (image.shape[1], image.shape[0])
    transform_matrix = cv2.getPerspectiveTransform(src=dst_points, dst=src_points)
    unwarped_image = cv2.warpPerspective(image, transform_matrix, image_size, flags=cv2.INTER_NEAREST)

    return unwarped_image


# EDGE-DETECTION________________________________________________________________________________________________________
def sobel_edge(image, sob_x=False, sob_y=False, threshold_low=25, threshold_high=200):
    """Returns sobel binary image after applying sobel x and/or y"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)

    if sob_x:
        sobel_absolute = np.absolute(sobel_x)
        sobel_scaled = np.uint8(255 * sobel_absolute / np.max(sobel_absolute))
    elif sob_y:
        sobel_absolute = np.absolute(sobel_y)
        sobel_scaled = np.uint8(255 * sobel_absolute / np.max(sobel_absolute))
    else:
        sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        sobel_scaled = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

    sobel_binary = np.zeros_like(sobel_scaled)
    sobel_binary[(sobel_scaled >= threshold_low) & (sobel_scaled <= threshold_high)] = 1

    return sobel_binary


def grayscale(image):
    """convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=5):
    """apply gaussian blur on the image using a kernel of size = (kernel_size, kernel_size)"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, threshold_low, threshold_high):
    """use canny edge detection with thresholds on the image"""
    return cv2.Canny(image, threshold_low, threshold_high)


def canny_edge(image, kernel_size=5, threshold_low=50, threshold_high=150):
    """Returns the canny image of the input image after thresholding"""
    gray_image = grayscale(image)
    gaussian_blur_image = gaussian_blur(gray_image, kernel_size=kernel_size)
    canny_image = canny(gaussian_blur_image, threshold_low=threshold_low, threshold_high=threshold_high)
    return canny_image


# COLOR SEPARATION______________________________________________________________________________________________________
def apply_threshold(channel, threshold_low, threshold_high):
    """Returns a binary """
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= threshold_low) & (channel <= threshold_high)] = 1
    return binary_output


def bgr_r_threshold(bgr_image, threshold_low=125, threshold_high=255):
    """Thresholds the R channel of the image and returns the binary output"""
    channel = bgr_image[:, :, 2]  # For a BGR format image
    return apply_threshold(channel=channel, threshold_low=threshold_low, threshold_high=threshold_high)


def hls_s_threshold(image, threshold_low=125, threshold_high=255):
    """Thresholds the S channel of the HSL converted image and returns the binary output"""
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    channel = hls_image[:, :, 2]
    return apply_threshold(channel=channel, threshold_low=threshold_low, threshold_high=threshold_high)


def lab_b_threshold(image, threshold_low=125, threshold_high=255):
    """Thresholds the B channel of the LAB converted image and returns the binary output"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channel = lab_image[:, :, 2]
    return apply_threshold(channel=channel, threshold_low=threshold_low, threshold_high=threshold_high)


def luv_l_threshold(image, threshold_low=125, threshold_high=255):
    """Thresholds the L channel of the LUV converted image and returns the binary output"""
    luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    channel = luv_image[:, :, 0]
    return apply_threshold(channel=channel, threshold_low=threshold_low, threshold_high=threshold_high)


def noise_remover(image):
    """Returns the image without noise"""
    kernel = np.ones((5, 5), np.uint8)
    erosion_image = cv2.erode(image * 255, kernel, iterations=1)
    clean_image = cv2.dilate(erosion_image, kernel, iterations=1)
    return clean_image


def binary_combiner(image, sobel_flag=False, canny_flag=False, clean_flag=False,
                    sobel_threshold_low=35, sobel_threshold_high=50,
                    canny_threshold_low=50, canny_threshold_high=150,
                    r_threshold_low=225, r_threshold_high=255,
                    s_threshold_low=220, s_threshold_high=250,
                    b_threshold_low=175, b_threshold_high=255,
                    l_threshold_low=215, l_threshold_high=255):
    """Combines the binary images and returns combined_binary_image"""
    global edge_image
    working_image = np.copy(image)

    undistorted_image = undistorted(working_image)
    warped_image = image_warper(undistorted_image)
    r_binary = bgr_r_threshold(warped_image, r_threshold_low, r_threshold_high)
    s_binary = hls_s_threshold(warped_image, s_threshold_low, s_threshold_high)
    b_binary = lab_b_threshold(warped_image, b_threshold_low, b_threshold_high)
    l_binary = luv_l_threshold(warped_image, l_threshold_low, l_threshold_high)

    if canny_flag:
        edge_image = canny_edge(image=warped_image,
                                threshold_low=canny_threshold_low,
                                threshold_high=canny_threshold_high)
    if sobel_flag:
        edge_image = sobel_edge(image=warped_image,
                                threshold_low=sobel_threshold_low,
                                threshold_high=sobel_threshold_high)

    combined_binary = np.zeros_like(r_binary)
    combined_binary[(r_binary == 1) | (s_binary == 1) | (b_binary == 1) | (l_binary == 1) | (edge_image == 1)] = 1

    if clean_flag:
        clean_image = noise_remover(combined_binary * 255)
        return clean_image
    return combined_binary


# FINDING LANES_________________________________________________________________________________________________________
def line_fit_calculator(image):
    """Calculates and returns the line fits for the left and the right lane lines and the output image"""
    n_windows = 9
    margin = 100
    min_pix = 50

    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)

    output_image = np.dstack((image, image, image)) * 255

    mid_point = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:mid_point])
    right_base = np.argmax(histogram[mid_point:]) + mid_point

    window_height = int(image.shape[0] / n_windows)

    non_zero = image.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])

    left_x_current = left_base
    right_x_current = right_base

    left_lane_indices = []
    right_lane_indices = []

    for window in range(n_windows):
        window_y_low = image.shape[0] - (window + 1) * window_height
        window_y_high = image.shape[0] - window * window_height

        window_x_left_l = left_x_current - margin
        window_x_left_r = left_x_current + margin
        window_x_right_l = right_x_current - margin
        window_x_right_r = right_x_current + margin

        cv2.rectangle(output_image, (window_x_left_l, window_y_low), (window_x_left_r, window_y_high), (0, 255, 0), 2)
        cv2.rectangle(output_image, (window_x_right_l, window_y_low), (window_x_right_r, window_y_high), (0, 255, 0), 2)

        valid_left_indices = ((non_zero_y >= window_y_low) & (non_zero_y <= window_y_high) &
                              (non_zero_x >= window_x_left_l) & (non_zero_x <= window_x_left_r)).nonzero()[0]
        valid_right_indices = ((non_zero_y >= window_y_low) & (non_zero_y <= window_y_high) &
                               (non_zero_x >= window_x_right_l) & (non_zero_x <= window_x_right_r)).nonzero()[0]

        left_lane_indices.append(valid_left_indices)
        right_lane_indices.append(valid_right_indices)

        if len(valid_left_indices) > min_pix:
            left_x_current = int(np.mean(non_zero_x[valid_left_indices]))
        if len(valid_right_indices) > min_pix:
            right_x_current = int(np.mean(non_zero_x[valid_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    left_x = non_zero_x[left_lane_indices]
    left_y = non_zero_y[left_lane_indices]
    right_x = non_zero_x[right_lane_indices]
    right_y = non_zero_y[right_lane_indices]

    left_line_fit = np.polyfit(left_y, left_x, 2)
    right_line_fit = np.polyfit(right_y, right_x, 2)

    output_image[non_zero_y[left_lane_indices], non_zero_x[left_lane_indices]] = [255, 0, 0]
    output_image[non_zero_y[right_lane_indices], non_zero_x[right_lane_indices]] = [0, 0, 255]

    return left_line_fit, right_line_fit, output_image


class Line:
    """Class for keeping track of a few previous lane line curves"""

    def __init__(self):
        """Initialize variables"""
        self.detected_in_prev = False
        self.best_fit_px = None
        self.latest_fit_px = None
        self.previous_fits_px = []
        self.coeff_diff = np.array([0, 0, 0], dtype='float')

    def calc_best_fit(self):
        """Averages the latest 5 line fits"""
        self.previous_fits_px.append(self.latest_fit_px)

        if len(self.previous_fits_px) > 5:
            self.previous_fits_px = self.previous_fits_px[1:]

        self.best_fit_px = np.average(self.previous_fits_px, axis=0)
        return

    def check_diff(self):
        """Checks whether the difference between new and latest fit px"""
        if self.coeff_diff[0] > 0.001:
            return True
        if self.coeff_diff[1] > 0.25:
            return True
        if self.coeff_diff[2] > 1000.:
            return True
        return False

    def add_new_fit(self, new_fit_px):
        """Adds a new fit to the class"""
        if np.array(self.latest_fit_px).all() is None and self.previous_fits_px == []:
            self.detected_in_prev = True
            self.latest_fit_px = new_fit_px
            self.calc_best_fit()
        else:
            self.coeff_diff = np.abs(new_fit_px - self.latest_fit_px)
            if self.check_diff():
                self.detected_in_prev = False
                return
            self.detected_in_prev = True
            self.latest_fit_px = new_fit_px
            self.calc_best_fit()
            return


def from_prev_line_fit_calculator(image, left_line=Line(), right_line=Line()):
    """Returns the left and right lane line fit along with the weighted image output"""

    left_fit = left_line.best_fit_px
    right_fit = right_line.best_fit_px

    margin = 100

    non_zero = image.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])

    left_low = (left_fit[0]*(non_zero_y**2)) + (left_fit[1]*non_zero_y) + left_fit[2] - margin
    left_high = (left_fit[0]*(non_zero_y**2)) + (left_fit[1]*non_zero_y) + left_fit[2] + margin
    right_low = (right_fit[0]*(non_zero_y**2)) + (right_fit[1]*non_zero_y) + right_fit[2] - margin
    right_high = (right_fit[0]*(non_zero_y**2)) + (right_fit[1]*non_zero_y) + right_fit[2] + margin
    left_lane_indices = ((non_zero_x > left_low) & (non_zero_x < left_high))
    right_lane_indices = ((non_zero_x > right_low) & (non_zero_x < right_high))

    left_x = non_zero_x[left_lane_indices]
    left_y = non_zero_y[left_lane_indices]
    right_x = non_zero_x[right_lane_indices]
    right_y = non_zero_y[right_lane_indices]

    left_line_fit = np.polyfit(left_y, left_x, 2)
    right_line_fit = np.polyfit(right_y, right_x, 2)
    plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fit_x = left_line_fit[0] * plot_y ** 2 + left_line_fit[1] * plot_y + left_line_fit[2]
    right_fit_x = right_line_fit[0] * plot_y ** 2 + right_line_fit[1] * plot_y + right_line_fit[2]

    output_image = np.dstack((image, image, image)) * 255
    window_image = np.zeros_like(output_image)

    output_image[non_zero_y[left_lane_indices], non_zero_x[left_lane_indices]] = [255, 0, 0]
    output_image[non_zero_y[right_lane_indices], non_zero_x[right_lane_indices]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_image, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_image, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(output_image, 1, window_image, 0.3, 0)

    return left_fit, right_fit, result
