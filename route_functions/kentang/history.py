import sqlite3
from flask import render_template, request, jsonify, redirect
import os

def history_delete(id):
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
        cursor = conn.cursor()

        cursor.execute(
            'DELETE FROM counting_group WHERE id = ?',
            (id,)
        )

        conn.commit()
        conn.close()
        return redirect("/kubis/history")

    except Exception as e:
        return jsonify({
            "error" : e
        }), 500
    

def history_detail(id):
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
    cursor = conn.cursor()

    cursor.execute(
        'SELECT * FROM counting c WHERE c.group_id = ?',
        (id,)
    )
    rows = cursor.fetchall()

    cursor.execute(
        '''
        SELECT cg.id as id, cg.date as date, AVG(c.conf_average) as group_conf_average, SUM(c.count) as group_count_total FROM counting_group as cg
        INNER JOIN counting as c on cg.id = c.group_id WHERE cg.id = ? GROUP BY cg.id
        ''',
        (id,)
    )

    info = next(iter(cursor.fetchall()))
    cursor.close()

    print(info)

    return render_template('history_detail.html', his=rows, info=info)

def history():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db'))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT cg.id as id, cg.date as date, AVG(c.conf_average) as group_conf_average, SUM(c.count) as group_count_total FROM counting_group as cg
        LEFT JOIN counting as c on cg.id = c.group_id GROUP BY cg.id
    ''')
    rows = cursor.fetchall()

    print(rows)
    conn.close()

    return render_template('history.html', his=rows)

def add_counting_group():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('counting_group_name')
        dt = data.get('date')

        if not name or not dt:
            return jsonify({
                "error" : "all form are required",
                "value" : [name, dt]
            }), 400

        with sqlite3.connect(os.path.join(os.path.dirname(__file__), '../database', 'logs.db')) as conn:
            c = conn.cursor()
            c.execute(
                'INSERT INTO counting_group (name, date) VALUES (?, ?)',
                (name, dt)
            )
            conn.commit()

        conn.close()
        return jsonify({
            "status" : "successful",
            "input" : {
                "name" : name,
                "date" : dt
            }
        }), 200

    return