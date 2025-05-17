from flask import Flask
from route import gen_route_bp, model_bp
import sqlite3, base64, os, json

from flask import Flask, jsonify
from PIL.TiffImagePlugin import IFDRational
from flask.json.provider import DefaultJSONProvider as BaseProvider

app = Flask(__name__)
app.register_blueprint(gen_route_bp)
app.register_blueprint(model_bp)

conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'database', 'logs.db'))
cursor = conn.cursor()

# create table if doesnot exists (for first time)
cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            conf_average REAL NOT NULL,
            count INTEGER NOT NULL
        )
''')

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # IFDRational → float
        if isinstance(obj, IFDRational):
            return float(obj)
        # bytes/bytearray → base64 str
        if isinstance(obj, (bytes, bytearray)):
            return base64.b64encode(obj).decode("ascii")
        # fallback to raise TypeError for anything else
        return super().default(obj)

class CustomJSONProvider(BaseProvider):
    def dumps(self, obj, **kwargs):
        # ensure jsonify() uses our encoder
        return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)

app.json = CustomJSONProvider(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)