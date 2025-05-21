from flask import Blueprint, session, render_template
import os
import csv
from datetime import datetime

kentang_bp = Blueprint('kentang_bp', __name__, template_folder='templates')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULT_DIR = os.path.join(BASE_DIR, 'static', 'uploads', 'ndre_result')

@kentang_bp.route('/hasilkentang')
def hasilkentang():
    result_image_url = session.get('result_image_url')
    mask_image_url = session.get('mask_image_url')
    final_overlay_url = session.get('final_overlay_url')
    original_preview = session.get('original_preview')
    nir_preview = session.get('nir_preview')
    red_preview = session.get('red_preview')
    rededge_preview = session.get('rededge_preview')
    coordinates = session.get('coordinates')
    overlay_result_url = session.get('overlay_result_url')
    ndre_image_url = session.get('ndre_image_url')
    overlay_before_labels_url = session.get('overlay_before_labels_url')

    classification_summary = None
    csv_path = os.path.join(RESULT_DIR, 'classification_summary.csv')
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            classification_summary = [row for row in reader]

    return render_template('hasilkentang.html',
        result_image_url=result_image_url,
        mask_image_url=mask_image_url,
        final_overlay_url=final_overlay_url,
        original_preview=original_preview,
        nir_preview=nir_preview,
        red_preview=red_preview,
        rededge_preview=rededge_preview,
        coordinates=coordinates,
        overlay_result_url=overlay_result_url,
        ndre_image_url=ndre_image_url,
        overlay_before_labels_url=overlay_before_labels_url,
        classification_summary=classification_summary,
        current_year=datetime.now().year
    )
