import os
import base64
from flask import request, jsonify
from model.inference import process_ndre_and_classify
from model.config import TEMP_DIR
import sqlite3
from PIL import Image
import piexif
from datetime import datetime as dt


def inference_kentang():
    cg_id = request.form.get("counting_group_id")
    saved_paths = {}

    try:
        required_files = ['nir', 'red', 'red_edge', 'rgb']

        # Simpan semua file sementara
        for fkey in required_files:
            if fkey not in request.files:
                return jsonify({"error": f"File '{fkey}' tidak ditemukan"}), 400

            f = request.files[fkey]
            if f.filename == '':
                return jsonify({"error": f"File '{fkey}' kosong namanya"}), 400

            save_path = os.path.join(TEMP_DIR, f.filename)
            f.save(save_path)
            saved_paths[fkey] = save_path

        # Panggil inferensi
        result = process_ndre_and_classify(
            nir_path=saved_paths['nir'],
            red_path=saved_paths['red'],
            red_edge_path=saved_paths['red_edge'],
            original_rgb_path=saved_paths['rgb'],
        )

        if not result:
            return jsonify({"error": "Inferensi model gagal"}), 500

        # Ambil data dari hasil
        encoded_original_rgb = result['images']['original_rgb']
        encoded_overlay_result = result['images']['overlay_result']
        encoded_overlay_before = result['images']['overlay_before']
        encoded_overlay_after = result['images']['overlay_after']

        metadata = result['metadata']
        gps = metadata.get('gps', {})
        datetime_str = metadata.get('datetime')
        summary = metadata.get('summary', {})

        # Simpan ke database
        try:
            conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO counting_kentang (
                    group_id, date, conf_average, count, 
                    lat, lat_ref, long, long_ref, alt, image_name, 
                    zona_sehat_percent, zona_tidak_sehat_percent, 
                    sehat_zones, sakit_zones,
                    img_original_rgb, img_overlay_result, img_overlay_before, img_overlay_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(cg_id),
                datetime_str or dt.now().isoformat(),
                summary.get('mean_confidence', 0),
                summary.get('num_zones', 0),
                gps.get('lat', 0.0),
                gps.get('lat_ref', ''),
                gps.get('long', 0.0),
                gps.get('long_ref', ''),
                gps.get('alt', 0.0),
                os.path.basename(saved_paths['rgb']),
                summary.get('zona_sehat_percent', 0),
                summary.get('zona_tidak_sehat_percent', 0),
                str(summary.get('sehat_zones', [])),
                str(summary.get('sakit_zones', [])),
                encoded_original_rgb,
                encoded_overlay_result,
                encoded_overlay_before,
                encoded_overlay_after
            ))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print("[DEBUG] Gagal menyimpan ke database:", db_err)

        # Hapus semua file sementara
        for path in saved_paths.values():
            try:
                os.remove(path)
            except Exception as rm_err:
                print(f"[DEBUG] Gagal menghapus file: {path} | {rm_err}")

        return jsonify({
            'images': {
                'original_rgb': encoded_original_rgb,
                'overlay_result': encoded_overlay_result,
                'overlay_before': encoded_overlay_before,
                'overlay_after': encoded_overlay_after
            },
            'metadata': {
                'gps': gps,
                'datetime': datetime_str,
                'summary': { **summary, 'group_id': int(cg_id) }
            }
        }), 200

    except Exception as e:
        # Pastikan file dibersihkan jika ada error juga
        for path in saved_paths.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        return jsonify({"error": str(e)}), 500
