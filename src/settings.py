import os

from numpy import pi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----RESIZE----#
param_resize_height = 900
param_resize_width = 1600
output_width = 1365 #853
output_height = 768 # 480

# ----PREPRO BIG IMAGE----#
block_size_big = 41
mean_sub_big = 15

# ----GRID COUNTOURS----#
ratio_lim = 2
smallest_area_allow = 75000
approx_poly_coef = 0.1

# ----GRID UPDATE AND SIMILARITY----#
lim_apparition_not_solved = 12
lim_apparition_solved = 60
same_grid_dist_ratio = 0.05
target_h_grid, target_w_grid = 450, 450

# ----HOUGH----#
thresh_hough = 500
thresh_hough_p = 100
minLineLength_h_p = 5
maxLineGap_h_p = 5
hough_rho = 3
hough_theta = 3 * pi / 180

# ----PREPRO IMAGE DIGIT----#
block_size_grid = 29  # 43
mean_sub_grid = 25

# ----DIGITS EXTRACTION----#
thresh_conf_cnn = 0.98
thresh_conf_cnn_high = 0.99
digits_2_check = 12

lim_bord = 10
thresh_h_low = 15
thresh_h_high = 50
thresh_area_low = 210
thresh_area_high = 900
l_case = 45
l_border = 1
offset_y = 2
min_digits_extracted = 13
