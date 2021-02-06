import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpImg
import matplotlib.path as mpPath
from moviepy.editor import VideoFileClip


# functions to be used _________________________________________________________________________________________________

def grayscale(image):
    """convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny(image, low_threshold, high_threshold):
    """use canny edge detection with thresholds on the image"""
    return cv2.Canny(image, low_threshold, high_threshold)


def gaussian_blur(image, kernel_size):
    """apply gaussian blur on the image using a kernel of size = (kernel_size, kernel_size)"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def region_of_interest(image, vertices):
    """define roi of the image"""
    mask = np.zeros_like(image)
    if len(image.shape) > 2:  # more than two channels
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=1):
    """draw lines on the image"""
    leftLane = [0, 0, 0, 0, 0]  # [slopeSum, count, yIntersectSum, x, y]
    rightLane = [0, 0, 0, 0, 0]  # [slopeSum, count, yIntersectSum, x, y]

    # Image dimensions being used (y, x): 540, 960

    # used to find if a point is within the polygon
    # https://stackoverflow.com/questions/39660851/deciding-if-a-point-is-inside-a-polygon-python
    vertices = np.array([[100, 540], [420, 330], [520, 330], [900, 540]])
    bbPath = mpPath.Path(vertices)

    minX = 100
    maxX = 900

    minY = 330
    maxY = 540

    # Go through each line and keep track of the slope and y-intersect of each line
    # we will average these out later to obtain the lane marking.

    for line in lines:
        for x1, y1, x2, y2 in line:

            # Only calculate slope if it's within our bounded region
            if bbPath.contains_point([x1, y1]) is not True:
                continue

            m = (y2 - y1) / (x2 - x1)
            # b = y - mx
            b = y1 - (m * x1)
            if m > 0:
                rightLane[0] = rightLane[0] + m
                rightLane[1] += 1
                rightLane[2] += b
                rightLane[3] = x1
                rightLane[4] = y1
            else:
                leftLane[0] = leftLane[0] + m
                leftLane[1] += 1
                leftLane[2] += b
                leftLane[3] = x1
                leftLane[4] = y1

            # cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 2)

    left_lane_slope = 0
    left_lane_y_intercept = 0
    right_lane_slope = 0
    right_lane_y_intercept = 0

    global last_slope_left
    global last_slope_right

    if leftLane[1] != 0:
        left_lane_slope = leftLane[0] / leftLane[1]
        left_lane_y_intercept = leftLane[2] / leftLane[1]
        last_slope_left = left_lane_slope
    else:
        left_lane_slope = last_slope_left

    if rightLane[1] != 0:
        right_lane_slope = rightLane[0] / rightLane[1]
        right_lane_y_intercept = rightLane[2] / rightLane[1]
        last_slope_right = right_lane_slope
    else:
        right_lane_slope = last_slope_right

    # x = (y - b) / m
    # y = mx + b
    x_bottom_left_lane = (maxY - left_lane_y_intercept) / left_lane_slope
    y_bottom_left_lane = left_lane_slope * x_bottom_left_lane + left_lane_y_intercept

    x_top_left_lane = (minY - left_lane_y_intercept) / left_lane_slope
    y_top_left_lane = left_lane_slope * x_top_left_lane + left_lane_y_intercept

    cv2.line(img, (int(round(x_bottom_left_lane)), int(round(y_bottom_left_lane))),
             (int(round(x_top_left_lane)), int(round(y_top_left_lane))), [255, 0, 0], thickness)

    minX_right = 100
    maxX_right = 900

    minY_right = 330
    maxY_right = 540

    x_bottom_right_lane = (maxY_right - right_lane_y_intercept) / right_lane_slope
    y_bottom_right_lane = right_lane_slope * x_bottom_right_lane + right_lane_y_intercept

    x_top_right_lane = (minY - right_lane_y_intercept) / right_lane_slope
    y_top_right_lane = right_lane_slope * x_top_right_lane + right_lane_y_intercept

    cv2.line(img, (int(round(x_bottom_right_lane)), int(round(y_bottom_right_lane))),
             (int(round(x_top_right_lane)), int(round(y_top_right_lane))), [0, 255, 0], thickness)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(img=line_image, lines=lines, color=[0, 255, 0], thickness=15)
    return line_image


def weighted_image(image, initial_image, alpha=0.8, beta=1.0, gamma=0.0):
    return cv2.addWeighted(initial_image, alpha, image, beta, gamma)


def image_processor(image):
    gray_image = grayscale(image)
    gaussian_image = gaussian_blur(gray_image, 5)
    canny_image = canny(gaussian_image, 50, 150)
    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_length = 30
    max_line_gap = 3

    hough_image = hough_lines(canny_image, rho, theta, threshold=threshold, min_line_len=min_line_length,
                              max_line_gap=max_line_gap)

    image_shape = hough_image.shape
    vertices = np.array([[(100, image_shape[0]),
                          (420, 330),
                          (520, 330),
                          (image_shape[1] - 60, image_shape[0])]], dtype=np.int32)
    region_of_interest_image = region_of_interest(hough_image, vertices=vertices)
    final_image = weighted_image(region_of_interest_image, image)

    return final_image
# end of functions _____________________________________________________________________________________________________


white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,3)
white_clip = clip1.fl_image(image_processor) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
cap = cv2.VideoCapture('./test_videos_output/solidWhiteRight.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()