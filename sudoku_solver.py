import os
import sys

from flask import Flask, request, Response, render_template
from tensorflow.keras.models import load_model
from src.settings import *
from src.extract_n_solve.extract_digits import process_extract_digits
from src.extract_n_solve.grid_detector_img import main_grid_detector_img
from src.extract_n_solve.grid_solver import main_solve_grids
from src.extract_n_solve.new_img_generator import *
from src.useful_functions import my_resize

app = Flask(__name__)
image_format = {"image/jpg", "image/jpeg", "image/png"}
input_file_path = 'static/images/input.jpg'


def main_process(im_path):
    frame = cv2.imread(im_path)
    if frame.shape[0] > 1000 or frame.shape[0] < 800:
        frame = my_resize(frame, width=param_resize_width, height=param_resize_height)
    im_grids_final, points_grids, list_transform_matrix = main_grid_detector_img(frame)
    if im_grids_final is None:
        print('Error')
        sys.exit(3)
    grids_matrix = process_extract_digits(im_grids_final, load_model('model/my_model.h5'))
    if all(elem is None for elem in grids_matrix):
        print('Error')
        sys.exit(3)
    grids_solved = main_solve_grids(grids_matrix)

    if grids_solved is None:
        print('Failed to solve')
        sys.exit(3)

    ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
    im_final = recreate_img_filled(frame, ims_filled_grid, points_grids, list_transform_matrix)
    im_final = cv2.resize(im_final, (480, 480))

    cv2.imwrite('static/images/' + os.path.splitext(os.path.basename(im_path))[0] + "_solved.jpg", im_final)


@app.route('/solution', methods=['POST'])
def solution():
    try:
        input_file = request.files['file']
    except:
        return Response('No file selected', status=400)
 
    input_file.save(os.path.join(app.root_path, input_file_path))
    main_process(input_file_path)

    return render_template('index.html', result=True)


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
