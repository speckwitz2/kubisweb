from flask import Flask, render_template, redirect
import sqlite3, os

def index():
    return render_template("index.html")

def count():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, date FROM counting_group
    ''')
    rows = cursor.fetchall()
    rows = list(rows)
    print(rows)
    cursor.close()

    return render_template("count.html", count_group=rows)

