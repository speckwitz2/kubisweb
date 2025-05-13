from flask import Blueprint, Response
from route_functions import index, count, inference, history
import time

gen_route_bp = Blueprint('general_bp', __name__)
model_bp = Blueprint('model_bp', __name__)

gen_route_bp.route('/')(index)
gen_route_bp.route('/count')(count)
gen_route_bp.route('/history')(history)

# to do inference or count
model_bp.route('/model/inference', methods=['POST'])(inference)

# to upload new model
@model_bp.route('/model/upload')
def upload():
    return 0