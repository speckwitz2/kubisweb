from flask import Blueprint, render_template, request, redirect, url_for, session ,make_response
import os
import uuid
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import rasterio
import cv2
import exifread
import matplotlib.colors as mcolors
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import csv
import base64
import datetime
from xhtml2pdf import pisa
from io import BytesIO

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Path model (gunakan raw string jika di Windows)
MODEL_PATH = r"D:\kubisweb-master\model\Kentang\final_vgg16_plant_health100.h5"

# Load model TensorFlow
model = tf.keras.models.load_model(MODEL_PATH)

# Blueprint Flask
kentang_bp = Blueprint('kentang_bp', __name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Semua media disimpan di dalam D:\kubisweb-master\static
STATIC_ROOT = r"D:\kubisweb-master\static"

# Upload folder di dalam static
UPLOAD_FOLDER = os.path.join(STATIC_ROOT, 'uploads')

# URL prefix untuk akses file static
MEDIA_URL = '/static/uploads/'

# Folder hasil klasifikasi NDRE di dalam static/uploads/ndre_result
RESULT_DIR = os.path.join(UPLOAD_FOLDER, 'ndre_result')

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def image_to_base64(image_url):
    """Convert media URL (MEDIA_URL + filename) to base64."""
    # Ambil bagian path relatif dari URL
    relative_path = image_url.replace(MEDIA_URL, '')
    image_path = os.path.join(RESULT_DIR, relative_path)

    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
@kentang_bp.route('/pdfkentang')
def pdfkentang():
    coordinates = session.get('coordinates')  # tetap bisa pakai koordinat kalau ada

    # Langsung pakai nama file yang fix
    filename = "Overlay_Before_Labels.png"
    final_overlay_url = f"/static/uploads/ndre_result/{filename}"
    image_path = os.path.abspath(os.path.join(BASE_DIR, '..', final_overlay_url.lstrip('/')))

    if not os.path.exists(image_path):
        return f"Gambar tidak ditemukan: {image_path}", 404

    context = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'coordinates': coordinates,
        'image_path': final_overlay_url  # tetap URL, agar bisa ditangani link_callback()
    }

    html = render_template("pdfkentang.html", **context)

    pdf_stream = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_stream, link_callback=link_callback)

    if pisa_status.err:
        return "Gagal membuat PDF", 500

    response = make_response(pdf_stream.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename="ndre_report.pdf"'

    return response


def link_callback(uri, rel):
    if uri.startswith('/static/'):
        path = os.path.abspath(os.path.join(BASE_DIR, '..', uri.lstrip('/')))
        if os.path.isfile(path):
            return path
    return uri