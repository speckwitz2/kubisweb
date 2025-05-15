from flask import request, jsonify
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import io
import base64
from model import doDetection
import sqlite3
import os

DESIRED_TAGS = {
    'GPSLatitude',
    'GPSLongitude',
    'GPSAltitude',
    'DateTimeOriginal',  # Pillow uses 'DateTimeOriginal' for the original creation timestamp
}

def inference():
    try:        
        im = request.files.get("picture")
        im_arr = np.array(Image.open(im))

        raw_exif = Image.open(im).getexif()


        tag_name_to_id = {name: id for id, name in TAGS.items()}  # ExifTags.TAGS maps IDs → names :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
        metadata = {}
        for tag_name in DESIRED_TAGS:
            tag_id = tag_name_to_id.get(tag_name)
            if tag_id is not None:
                value = raw_exif.get(tag_id)
                if value is not None:
                    metadata[tag_name] = value  # only include if present :contentReference[oaicite:4]{index=4}


        # an example
        # labeled_im = Image.fromarray(im_arr)
        labeled_im_array, count, average_confidence = doDetection(im_arr)
        labeled_im = Image.fromarray(labeled_im_array)
        
        img_io = io.BytesIO()
        labeled_im.save(img_io, 'JPEG')
        img_io.seek(0)

        # insert into db
        conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (date, conf_average, count)
            VALUES (datetime("now"), ?, ?)
        ''', (str('%.2f' % average_confidence), count))
        conn.commit()
        conn.close()

        return jsonify({
            'count' : count,
            'average_confidence' : f"{average_confidence:.2f}", 
            'image' : base64.b64encode(img_io.getvalue()).decode('utf-8'),
            'metadata' : metadata
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500