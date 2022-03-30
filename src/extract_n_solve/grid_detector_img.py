import cv2
import numpy as np

from src.settings import *
from src.solving_objects.MyHoughPLines import *


def preprocess_im(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
    blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                    block_size_big, mean_sub_big)
    thresh_not = cv2.bitwise_not(thresh)
    kernel_close = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)
    dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)

    return dilate


def find_corners(contour):
    top_left = [10000, 10000]
    top_right = [0, 10000]
    bottom_right = [0, 0]
    bottom_left = [10000, 0]
    mean_x = np.mean(contour[:, :, 0])
    mean_y = np.mean(contour[:, :, 1])

    for j in range(len(contour)):
        x, y = contour[j][0]
        if x > mean_x:
            if y > mean_y:
                bottom_right = [x, y]
            else:
                top_right = [x, y]
        else:
            if y > mean_y:
                bottom_left = [x, y]
            else:
                top_left = [x, y]
    return [top_left, top_right, bottom_right, bottom_left]


def look_for_corners(img_lines):
    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contours = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    biggest_area = cv2.contourArea(contours[0])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < smallest_area_allow:
            break
        if area > biggest_area / ratio_lim:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, approx_poly_coef * peri, True)
            if len(approx) == 4:
                best_contours.append(approx)

    if not best_contours:
        return None
    corners = []
    for best_contour in best_contours:
        corners.append(find_corners(best_contour))

    return corners


def get_lines_and_corners(img, edges):
    my_lines = []
    img_lines = np.zeros((img.shape[:2]), np.uint8)
    lines_raw = cv2.HoughLinesP(edges,
                                rho=hough_rho, theta=hough_theta,
                                threshold=thresh_hough_p,
                                minLineLength=minLineLength_h_p, maxLineGap=maxLineGap_h_p)

    for line in lines_raw:
        my_lines.append(MyHoughPLines(line))

    for line in my_lines:
        x1, y1, x2, y2 = line.get_limits()
        cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2)

    return look_for_corners(img_lines)


def undistorted_grids(frame, points_grids, ratio):
    undistorted = []
    true_points_grids = []
    transfo_matrix = []
    for points_grid in points_grids:
        points_grid = np.array(points_grid, dtype=np.float32) * ratio
        final_pts = np.array(
            [[0, 0], [target_w_grid - 1, 0],
             [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
            dtype=np.float32)
        M = cv2.getPerspectiveTransform(points_grid, final_pts)
        undistorted.append(cv2.warpPerspective(frame, M, (target_w_grid, target_h_grid)))
        true_points_grids.append(points_grid)
        transfo_matrix.append(np.linalg.inv(M))
    return undistorted, true_points_grids, transfo_matrix


def main_grid_detector_img(frame):
    prepro_im_edges = preprocess_im(frame)
    extreme_points_biased = get_lines_and_corners(frame.copy(), prepro_im_edges)

    if extreme_points_biased is None:
        return None, None, None
    
    grids_final, points_grids, transfo_matrix = undistorted_grids(frame, extreme_points_biased, 1)
    return grids_final, points_grids, transfo_matrix