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

@kentang_bp.route('/kentang', methods=['GET', 'POST'])
def kentang():
    if request.method == 'POST' and all(field in request.files for field in ['nir', 'red', 'red_edge', 'original']):
        uid = str(uuid.uuid4())
        original = request.files['original']
        nir = request.files['nir']
        red = request.files['red']
        red_edge = request.files['red_edge']

        # Simpan nama file dengan UUID
        orig_name = f'{uid}_original.png'
        nir_name = f'{uid}_nir.TIF'
        red_name = f'{uid}_red.TIF'
        red_edge_name = f'{uid}_red_edge.TIF'

        original_path = os.path.join(UPLOAD_FOLDER, orig_name)
        nir_path = os.path.join(UPLOAD_FOLDER, nir_name)
        red_path = os.path.join(UPLOAD_FOLDER, red_name)
        red_edge_path = os.path.join(UPLOAD_FOLDER, red_edge_name)

        # Simpan file upload
        original.save(original_path)
        nir.save(nir_path)
        red.save(red_path)
        red_edge.save(red_edge_path)

        # Ambil koordinat GPS dari metadata
        coordinates = read_gps_metadata(original_path)

        # Panggil fungsi utama NDRE + klasifikasi, kirim 'model' sebagai argumen
        result_filename, prediction_result, mask_filename, final_overlay_path, ndre_filename, temp_overlay_path = process_ndre_and_classify(
            nir_path, red_path, red_edge_path, original_path, model
        )

        # Simpan ke session
        session['result_image_url'] = MEDIA_URL + 'ndre_result/' + result_filename
        session['mask_image_url'] = MEDIA_URL + mask_filename if mask_filename else None
        session['original_preview'] = MEDIA_URL + orig_name
        session['nir_preview'] = MEDIA_URL + nir_name
        session['red_preview'] = MEDIA_URL + red_name
        session['rededge_preview'] = MEDIA_URL + red_edge_name
        session['final_overlay_url'] = MEDIA_URL + 'ndre_result/' + os.path.basename(final_overlay_path) if final_overlay_path else None
        session['coordinates'] = coordinates
        session['ndre_image_url'] = MEDIA_URL + ndre_filename

        # Cek jika ada hasil overlay CNN sebelum pelabelan
        overlay_result_url = None
        if "MaskUnhealthy" in result_filename:
            overlay_filename = result_filename.replace("MaskUnhealthy", "OverlayResult")
            overlay_path_full = os.path.join(RESULT_DIR, overlay_filename)
            if os.path.exists(overlay_path_full):
                overlay_result_url = MEDIA_URL + 'ndre_result/' + overlay_filename
        session['overlay_result_url'] = overlay_result_url

        # Cek hasil overlay sebelum label CNN
        overlay_before_labels_path = os.path.join(RESULT_DIR, 'Overlay_Before_Labels.png')
        overlay_before_labels_url = MEDIA_URL + 'ndre_result/Overlay_Before_Labels.png' if os.path.exists(overlay_before_labels_path) else None
        session['overlay_before_labels_url'] = overlay_before_labels_url

        # Arahkan ke halaman hasil
        return redirect(url_for('kentang_bp.hasilkentang'))

    return render_template('kentang.html')

def read_gps_metadata(file_path):
    """Read GPS metadata from JPEG/PNG/TIFF files."""
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f)

        gps_info = {
            'Latitude': tags.get('GPS GPSLatitude'),
            'Longitude': tags.get('GPS GPSLongitude'),
            'Altitude': tags.get('GPS GPSAltitude'),
            'DateTime': tags.get('EXIF DateTimeOriginal')
        }

        # Print out metadata for debugging
        print("📸 Metadata from file (JPEG/PNG/TIFF):")
        for key, value in gps_info.items():
            print(f"   {key}: {value}")

        if gps_info['Latitude'] and gps_info['Longitude'] and gps_info['Altitude']:
            lat = dms_to_decimal(gps_info['Latitude'].values)
            lon = dms_to_decimal(gps_info['Longitude'].values)
            alt = float(eval(str(gps_info['Altitude'].values[0])))

            formatted_lat, formatted_lon, formatted_alt = format_coordinates(lat, lon, alt)
            formatted_datetime = str(gps_info['DateTime']) if gps_info['DateTime'] else "Date/time not available"

            print(f"   Latitude: {formatted_lat}")
            print(f"   Longitude: {formatted_lon}")
            print(f"   Altitude: {formatted_alt}")
            print(f"   DateTime: {formatted_datetime}")

            return {
                'Latitude': formatted_lat,
                'Longitude': formatted_lon,
                'Altitude': formatted_alt,
                'DateTime': formatted_datetime
            }
        else:
            return None
        

def dms_to_decimal(dms):
    """Convert from [deg, min, sec] format to decimal"""
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(eval(str(dms[2])))  # 'numerator/denominator'
    return degrees + (minutes / 60.0) + (seconds / 3600.0)

def format_coordinates(lat, lon, alt):
    """Format coordinates and altitude for better readability"""
    lat_direction = "N" if lat >= 0 else "S"
    lon_direction = "E" if lon >= 0 else "W"
    lat = abs(lat)
    lon = abs(lon)

    formatted_lat = f"{lat:.2f}° {lat_direction}"
    formatted_lon = f"{lon:.2f}° {lon_direction}"
    formatted_alt = f"{alt:.2f} meters" if alt else "Altitude not available"

    return formatted_lat, formatted_lon, formatted_alt

def process_ndre_and_classify(nir_path, red_path, red_edge_path, original_rgb_path, model):
    print("\n[INFO] Membaca dan menormalkan citra multispektral...")
    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(np.float32)
    with rasterio.open(red_path) as src:
        red = src.read(1).astype(np.float32)
    with rasterio.open(red_edge_path) as src:
        red_edge = src.read(1).astype(np.float32)

    nir /= np.max(nir)
    red /= np.max(red)
    red_edge /= np.max(red_edge)

    print("[INFO] Menghitung NDVI dan membuat mask vegetasi...")
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)
    ndvi_thresh = threshold_otsu(ndvi)
    vegetation_mask = ndvi > ndvi_thresh

    print("[INFO] Menghitung NDRE di area vegetasi...")
    ndre = np.full_like(nir, -1.0, dtype=np.float32)
    ndre[vegetation_mask] = (nir[vegetation_mask] - red_edge[vegetation_mask]) / (
        nir[vegetation_mask] + red_edge[vegetation_mask] + epsilon)
    ndre = np.nan_to_num(ndre, nan=-1.0)
    ndre = gaussian_filter(ndre, sigma=1)

    print("[INFO] Membuat citra NDRE berwarna...")
    ndre_classes = [
        (-1.0, -0.40, "#404040"), (-0.40, -0.10, "#8B4513"),
        (-0.10, 0.15, "#FF0000"), (0.15, 0.35, "#FFFF00"),
        (0.35, 0.80, "#00FF00")
    ]
    colors = np.array([mcolors.hex2color(c[2]) for c in ndre_classes]) * 255
    ndre_colored = np.zeros((*ndre.shape, 3), dtype=np.uint8)

    for i, (low, high, _) in enumerate(ndre_classes):
        mask = (ndre >= low) & (ndre < high)
        for j in range(3):
            ndre_colored[:, :, j][mask] = int(colors[i][j])

    filename = os.path.basename(nir_path).replace(".TIF", "_NDRE.jpg")
    ndre_image_path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(ndre_image_path, cv2.cvtColor(ndre_colored, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Citra NDRE tersimpan di: {ndre_image_path}")

    print("[INFO] Melakukan prediksi CNN pada citra NDRE...")
    prediction = predict_image(ndre_image_path)

    print("[INFO] Membuat mask vegetasi dari NIR + RED...")
    mask_filename = generate_nir_mask(nir_path, red_path)
    mask_path_full = os.path.join(RESULT_DIR, mask_filename)
    if not os.path.exists(mask_path_full):
        raise FileNotFoundError(f"❌ Mask file tidak ditemukan: {mask_path_full}")
    print(f"[FOUND] Mask vegetasi ditemukan: {mask_path_full}")

    print("[INFO] Melakukan klasifikasi zona berdasarkan NDRE dan mask vegetasi...")
    overlay_filename = classify_zones_with_mask(ndre_image_path, mask_path_full, model)
    overlay_path = os.path.join(RESULT_DIR, overlay_filename)
    print(f"[SAVED] Mask klasifikasi zona disimpan di: {overlay_path}")

    print("[INFO] Membuat overlay transparan dengan klasifikasi...")
    final_overlay_filename = "Final_Overlay_Transparan_Kentang.png"
    final_overlay_path = os.path.join(RESULT_DIR, final_overlay_filename)

    try:
        generate_overlay_using_classified_mask(
            nir_path=nir_path,
            original_path=original_rgb_path,
            classification_mask_path=overlay_path,
            output_dir=RESULT_DIR
        )
        if os.path.exists(final_overlay_path):
            print(f"[SUCCESS] Overlay transparan berhasil disimpan di: {final_overlay_path}")
        else:
            print(f"⚠️ Gagal menyimpan overlay transparan. File tidak ditemukan di: {final_overlay_path}")
            final_overlay_path = None
    except Exception as e:
        print("❌ Gagal membuat overlay transparan:", e)
        final_overlay_path = None

    relative_ndre_path = os.path.join('static', 'ndre_result', filename)

    print("\n=== RINGKASAN ===")
    print("Overlay klasifikasi mask:", overlay_filename)
    print("Prediksi CNN:", prediction)
    print("Mask vegetasi:", mask_filename)
    print("Final overlay transparan:", final_overlay_path)
    print("Relative NDRE path:", relative_ndre_path)
    print("Full overlay klasifikasi path:", overlay_path)
    print("==================\n")

    return overlay_filename, prediction, mask_filename, final_overlay_path, relative_ndre_path, overlay_path


def generate_nir_mask(nir_path, red_path):
    # Baca band NIR dan Red
    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(np.float32)

    with rasterio.open(red_path) as src:
        red = src.read(1).astype(np.float32)

    # Normalisasi ke 0-1
    nir /= np.max(nir)
    red /= np.max(red)

    # Hitung NDVI
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)

    # Thresholding Otsu untuk deteksi vegetasi
    ndvi_thresh = threshold_otsu(ndvi)
    adjusted_thresh = max(ndvi_thresh - 0.05, 0)  # Koreksi threshold
    vegetation_mask = (ndvi > adjusted_thresh).astype(np.uint8) * 255

    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=4)

    # KMeans Clustering
    masked_ndvi = np.copy(ndvi)
    masked_ndvi[mask_cleaned == 0] = np.nan
    ndvi_values = masked_ndvi[~np.isnan(masked_ndvi)].reshape(-1, 1)

    if len(ndvi_values) < 10:
        raise ValueError("NDVI values terlalu sedikit untuk clustering.")

    kmeans = KMeans(n_clusters=2, random_state=42).fit(ndvi_values)
    labels = np.full(masked_ndvi.shape, -1, dtype=np.int32)
    labels[~np.isnan(masked_ndvi)] = kmeans.labels_

    cluster_means = [np.mean(ndvi_values[kmeans.labels_ == i]) for i in range(2)]
    kentang_cluster = np.argmax(cluster_means)

    kentang_mask = (labels == kentang_cluster).astype(np.uint8) * 255

    # Filter kontur berdasarkan area
    contours, _ = cv2.findContours(kentang_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(kentang_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 200000:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    os.makedirs(RESULT_DIR, exist_ok=True)

    filename = os.path.splitext(os.path.basename(nir_path))[0] + "_Mask.png"
    result_path = os.path.join(RESULT_DIR, filename)

    print(f"[DEBUG] Menyimpan mask ke: {result_path}")
    cv2.imwrite(result_path, filtered_mask)

    # Return path relatif yang bisa diakses via URL Flask: /static/ndre_result/...
    return filename



def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")

    try:
        img = Image.open(img_path).convert("RGB").resize((256, 256))  # konversi ke RGB
    except Exception as e:
        raise ValueError(f"Gagal membuka gambar {img_path}: {e}")

    img_array = np.array(img) / 255.0

    if img_array.shape != (256, 256, 3):
        raise ValueError(f"Ukuran gambar tidak sesuai: {img_array.shape}, harus (256, 256, 3)")

    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)
    except Exception as e:
        raise RuntimeError(f"Model gagal memprediksi: {e}")

    return 'Sehat' if prediction[0][0] > 0.5 else 'Tidak Sehat'



def classify_zones_with_mask(image_path, mask_path, model):
    # Load citra dan ubah ke RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load dan resize mask ke ukuran image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Bersihkan mask dengan morfologi
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    classification_overlay = np.zeros_like(image, dtype=np.uint8)

    confidences = []
    predictions = []
    sehat_counter = 1
    sakit_counter = 1

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h != 0 else 0

        if area > 1000 and 0.3 < aspect_ratio < 3.5:
            roi = image[y:y+h, x:x+w]
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour - [x, y]
            cv2.drawContours(contour_mask, [shifted_contour], -1, 255, -1)

            masked_roi = cv2.bitwise_and(roi, roi, mask=contour_mask)
            white_bg = np.ones_like(masked_roi, dtype=np.uint8) * 255
            final_patch = np.where(masked_roi == 0, white_bg, masked_roi)

            patch_resized = cv2.resize(final_patch, (256, 256)) / 255.0
            patch_resized = np.expand_dims(patch_resized, axis=0)

            prediction = model.predict(patch_resized, verbose=0)[0]
            class_idx = np.argmax(prediction)

            confidences.append(np.max(prediction))
            predictions.append(class_idx)

            if class_idx == 1:  # Zona Sakit
                color = (255, 0, 0)
                cv2.drawContours(binary_mask[y:y+h, x:x+w], [shifted_contour], -1, 1, thickness=cv2.FILLED)
                cv2.drawContours(classification_overlay[y:y+h, x:x+w], [shifted_contour], -1, color, thickness=cv2.FILLED)
                cv2.drawContours(image, [contour], -1, color, thickness=2)
                sakit_counter += 1
            else:  # Zona Sehat
                color = (0, 255, 0)
                cv2.drawContours(classification_overlay[y:y+h, x:x+w], [shifted_contour], -1, color, thickness=cv2.FILLED)
                sehat_counter += 1

    # Overlay hasil klasifikasi di atas gambar asli
    overlay_result = cv2.addWeighted(image, 0.6, classification_overlay, 0.4, 0)

    # Simpan mask biner zona sakit (0 dan 1)
    mask_filename = os.path.basename(image_path).replace("_NDRE.jpg", "_MaskUnhealthy.jpg")
    mask_full_path = os.path.join(RESULT_DIR, mask_filename)
    cv2.imwrite(mask_full_path, binary_mask * 255)

    # Simpan gambar overlay hasil klasifikasi
    overlay_filename = os.path.basename(image_path).replace("_NDRE.jpg", "_OverlayResult.jpg")
    overlay_full_path = os.path.join(RESULT_DIR, overlay_filename)
    cv2.imwrite(overlay_full_path, cv2.cvtColor(overlay_result, cv2.COLOR_RGB2BGR))

    # Hitung statistik area
    total_area = image.shape[0] * image.shape[1]
    unhealthy_area = np.sum(binary_mask == 1)
    healthy_area = total_area - unhealthy_area
    num_total = len(confidences)
    num_sakit = sum(1 for c in zip(confidences, predictions) if c[1] == 1)
    num_sehat = num_total - num_sakit

    # Persentase area sehat dan sakit
    healthy_percentage = (healthy_area / total_area) * 100
    unhealthy_percentage = (unhealthy_area / total_area) * 100

    # Simpan ringkasan ke CSV (tulis ulang setiap kali)
    csv_path = os.path.join(RESULT_DIR, 'classification_summary.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'zona_sehat (%)', 'zona_tidak_sehat (%)', 'mean_confidence', 'num_zones', 'sehat_zones', 'sakit_zones'])
        writer.writerow([
            os.path.basename(image_path),
            round(healthy_percentage, 2),
            round(unhealthy_percentage, 2),
            round(np.mean(confidences), 4) if confidences else 0.0,
            num_total,
            num_sehat,
            num_sakit
        ])

    return mask_filename


def generate_overlay_using_classified_mask(nir_path, original_path, classification_mask_path, output_dir):
    # === 1. Load mask area sakit dari CNN (putih = sakit, hitam = sehat) === #
    klasifikasi_mask = cv2.imread(classification_mask_path, cv2.IMREAD_GRAYSCALE)
    sakit_mask = np.uint8(klasifikasi_mask == 255) * 255

    # Dilasi biar area sakit kelihatan lebih "menggoda" 😉
    kernel = np.ones((5, 5), np.uint8)
    sakit_mask = cv2.dilate(sakit_mask, kernel, iterations=2)

    # === 2. Load citra NIR dan ubah ke RGB === #
    with rasterio.open(nir_path) as src:
        nir_img = src.read(1).astype(np.float32)
        nir_img = (nir_img - nir_img.min()) / (nir_img.max() - nir_img.min()) * 255
        nir_img = nir_img.astype(np.uint8)
    nir_rgb = cv2.cvtColor(nir_img, cv2.COLOR_GRAY2RGB)

    # Resize mask ke ukuran NIR
    sakit_mask_resized = cv2.resize(sakit_mask, (nir_img.shape[1], nir_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # === 3. Buat overlay sementara untuk registrasi === #
    heatmap_red_temp = np.zeros_like(nir_rgb)
    heatmap_red_temp[:, :, 2] = sakit_mask_resized
    overlay_temp = cv2.addWeighted(nir_rgb, 1, heatmap_red_temp, 0.2, 0)

    temp_overlay_path = os.path.join(output_dir, "Overlay_FullRed.png")
    cv2.imwrite(temp_overlay_path, overlay_temp)

    # === 4. Registrasi ke citra RGB asli pakai ORB === #
    image_rgb = cv2.imread(original_path)
    image_overlay = cv2.imread(temp_overlay_path)

    gray1 = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image_overlay, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    if len(good_matches) < 10:
        print("💔 Tidak cukup fitur untuk transformasi.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is None:
        print("💔 Affine transform gagal dihitung.")
        return None

    h, w = image_overlay.shape[:2]
    aligned_rgb = cv2.warpAffine(image_rgb, M, (w, h))

    # Resize mask ke ukuran align
    if sakit_mask_resized.shape[:2] != (h, w):
        sakit_mask_resized = cv2.resize(sakit_mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)

    # === 5. Buat heatmap merah & overlay ke RGB === #
    heatmap_red = np.zeros_like(aligned_rgb)
    heatmap_red[:, :, 2] = sakit_mask_resized
    heatmap_red = cv2.multiply(heatmap_red, np.array([1, 1, 4], dtype=np.float32))
    heatmap_red = np.clip(heatmap_red, 0, 255).astype(np.uint8)
    heatmap_red = cv2.GaussianBlur(heatmap_red, (11, 11), 0)

    alpha = 0.4
    final_overlay_before_labels = cv2.addWeighted(aligned_rgb, 1, heatmap_red, alpha, 0)

    # Simpan hasil sebelum label
    output_before_labels = os.path.join(output_dir, "Overlay_Before_Labels.png")
    cv2.imwrite(output_before_labels, final_overlay_before_labels)

    # === 6. Menambahkan label zona sakit === #
    contours, _ = cv2.findContours(sakit_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sakit_counter = 1
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Hanya zona yang cukup besar yang diberi label
            x, y, w, h = cv2.boundingRect(contour)

            # Menambahkan label zona sakit
            zona_label = f"Zona Sakit {sakit_counter}"
            sakit_counter += 1
            label_position = (x, max(20, y - 10))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.7, min(1.2, (w * h) / (image_rgb.shape[0] * image_rgb.shape[1] * 0.004)))
            font_thickness = 2

            # Ukuran teks untuk latar
            (text_width, text_height), _ = cv2.getTextSize(zona_label, font, font_scale, font_thickness)
            text_bg_topleft = (label_position[0] - 5, label_position[1] - text_height - 5)
            text_bg_bottomright = (label_position[0] + text_width + 5, label_position[1] + 5)

            # Pastikan tetap di dalam gambar
            text_bg_topleft = (max(0, text_bg_topleft[0]), max(0, text_bg_topleft[1]))
            text_bg_bottomright = (
                min(image_rgb.shape[1], text_bg_bottomright[0]),
                min(image_rgb.shape[0], text_bg_bottomright[1])
            )

            # Rectangle latar belakang teks (hitam pekat semi-transparan)
            overlay_tmp = final_overlay_before_labels.copy()
            cv2.rectangle(overlay_tmp, text_bg_topleft, text_bg_bottomright, (0, 0, 0), thickness=-1)
            alpha = 0.75
            final_overlay_before_labels = cv2.addWeighted(overlay_tmp, alpha, final_overlay_before_labels, 1 - alpha, 0)

            # Tulisan zona dengan outline tiga lapis
            cv2.putText(final_overlay_before_labels, zona_label, label_position, font, font_scale, (255, 255, 255), thickness=5, lineType=cv2.LINE_AA)  # luar putih
            cv2.putText(final_overlay_before_labels, zona_label, label_position, font, font_scale, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)        # tengah hitam
            cv2.putText(final_overlay_before_labels, zona_label, label_position, font, font_scale, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)            # isi merah

    # Simpan hasil akhir overlay setelah label
    output_final = os.path.join(output_dir, "Final_Overlay_Transparan_Kentang.png")
    cv2.imwrite(output_final, final_overlay_before_labels)

    print(f"🔥 Overlay transparan disimpan di: {output_final}")
    return output_before_labels, output_final

