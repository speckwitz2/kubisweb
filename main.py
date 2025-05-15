from flask import Flask
from route import gen_route_bp, model_bp
import sqlite3
import os

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)