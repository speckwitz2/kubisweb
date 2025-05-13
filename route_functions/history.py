import sqlite3
from flask import render_template
import os

def history():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, count, conf_average
        FROM history
        ORDER BY id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    return render_template('history.html', rows=rows)