import os
import sqlite3
import base64
import json
from flask import Flask, jsonify
from PIL.TiffImagePlugin import IFDRational
from flask.json.provider import DefaultJSONProvider as BaseProvider

# Blueprint imports (pastikan pathnya sesuai)
from route import gen_route_bp, model_bp, kentang_bp

app = Flask(__name__)

# *** Tambahkan secret key di sini ***
app.secret_key = 'rahasia-super-rahasia-unik-1234567890'

# Daftarkan blueprint
app.register_blueprint(gen_route_bp)
app.register_blueprint(model_bp)
app.register_blueprint(kentang_bp)

# Setup database (pastikan folder database ada)
db_path = os.path.join(os.path.dirname(__file__), 'database', 'logs.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table jika belum ada
cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        conf_average REAL NOT NULL,
        count INTEGER NOT NULL
    )
''')
conn.commit()
conn.close()

# Custom JSON encoder agar bisa handle IFDRational dan bytes
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, IFDRational):
            return float(obj)
        if isinstance(obj, (bytes, bytearray)):
            return base64.b64encode(obj).decode("ascii")
        return super().default(obj)

class CustomJSONProvider(BaseProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)

app.json = CustomJSONProvider(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
