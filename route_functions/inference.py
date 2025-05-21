from flask import request, jsonify
from PIL import Image
from PIL import ExifTags
import numpy as np, piexif
import io, json
import base64
from model import doDetection
import sqlite3
import os
from datetime import datetime as dttm

def inference():
    try:        
        lat_rational, alt_rational = ("", "")
        im = request.files.get("picture")
        im_arr = np.array(Image.open(im))

        img_file = Image.open(im)
        exif_bytes = img_file.info.get('exif')
        if exif_bytes:
            exif_dict = piexif.load(exif_bytes)
            gps = exif_dict["GPS"]
            lat_rational = gps.get(piexif.GPSIFD.GPSLatitude)
            lat_rational_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef).decode('ascii')
            long_rational = gps.get(piexif.GPSIFD.GPSLongitude)
            long_rational_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef).decode('ascii')
            alt_rational = gps.get(piexif.GPSIFD.GPSAltitude)
            datetime = exif_dict["Exif"].get(piexif.ExifIFD.DateTimeOriginal).decode('ascii')


        labeled_im_array, count, average_confidence = doDetection(im_arr)
        labeled_im = Image.fromarray(labeled_im_array)
        
        img_io = io.BytesIO() 
        labeled_im.save(img_io, 'JPEG')
        img_io.seek(0)

        # adding file
        img_filename = f"detection_{dttm.now()}.jpeg"
        with open(os.path.join(os.path.dirname(__file__), "../static/result", img_filename), "wb") as file:
            file.write(img_io.getvalue())

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
            'image' : f"/static/result/{img_filename}",
            'metadata' : {
                'gps' : {
                    'lat' : lat_rational or None,
                    'lat_ref' : lat_rational_ref or None,
                    'long' : long_rational or None,
                    'long_ref' : long_rational_ref or None,
                    'alt' : alt_rational or None
                },
                'datetime' : datetime
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500