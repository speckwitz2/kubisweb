from flask import Blueprint, Response
from route_functions import index, count, inference, history ,kentang, hasilkentang , pdfkentang
import time

gen_route_bp = Blueprint('general_bp', __name__)
model_bp = Blueprint('model_bp', __name__)
kentang_bp = Blueprint('kentang_bp', __name__)

gen_route_bp.route('/')(index)
gen_route_bp.route('/count')(count)
gen_route_bp.route('/history')(history)
kentang_bp.route('/kentang', methods=['GET', 'POST'])(kentang)
kentang_bp.route('/hasilkentang')(hasilkentang)
kentang_bp.route('/pdfkentang')(pdfkentang)

# to do inference or count
model_bp.route('/model/inference', methods=['POST'])(inference)

# to upload new model
@model_bp.route('/model/upload')
def upload():
    return 0