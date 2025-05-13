from flask import Flask, render_template, redirect

def index():
    return redirect("/count")
    # return render_template("index.html")

def count():
    return render_template("count.html")

