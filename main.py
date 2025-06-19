from flask import Flask, redirect, session, url_for, render_template
from flask.json.provider import DefaultJSONProvider as BaseProvider

import sqlite3, base64, os, json
import route_functions as rf
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from PIL.TiffImagePlugin import IFDRational

app = Flask(__name__)
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app.secret_key = env.get("APP_SECRET_KEY")

path = os.path.join(os.path.dirname(__file__), 'database', 'logs.db')
# with open(path, 'w') as f:
#     f.truncate(0)

conn = sqlite3.connect(path)
cursor = conn.cursor()

# create table if doesnot exists (for first time)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS counting_group (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date DATETIME NOT NULL);
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS counting (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                date DATETIME NOT NULL,
                conf_average REAL NOT NULL,
                count INTEGER NOT NULL,
                lat REAL NOT NULL,
                lat_ref TEXT NOT NULL,
                long REAL NOT NULL,
                long_ref TEXT NOT NULL,
                alt REAL NOT NULL,
                image_name TEXT NOT NULL,
                FOREIGN KEY (group_id) REFERENCES counting_group(id)
    );
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

# custom oAuth
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
        "audience" : "https://localhost:5000/"
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

# authentication related route
@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri = url_for("callback", _external=True)
    )

# Only available if these true
## an existing user let you to signup (thus, auth token required) or no user available
@app.route('/signup')
def signup():
    return render_template('signup.html')

# A function to do the heavy lifting in signup (e.g : send data to auth0)
@app.route('/signup/post', methods=["POST"])
def adduser():
    return ""

@app.route('/signup/with_google')
def signup_w_google():
        return oauth.auth0.authorize_redirect(
        redirect_uri = url_for("callback", _external=True),
        connection = 'google-oauth2'
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN") + "/v2/logout?" + urlencode(
            {
                "returnTo": url_for("index", _external=True),
                "client_id" : env.get("AUTH0_CLIENT_ID")
            },
            quote_via=quote_plus,
        )
    )

@app.route('/check')
def check():
    if 'token' not in session:
        return 'no user'
    else:
        return 'user'

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["token"] = token
    return redirect("/")


app.route('/')(rf.index)

app.route('/kubis/count')(rf.requires_auth(rf.count))
app.route('/kubis/history')(rf.requires_auth(rf.history))
app.route('/kubis/history/detail/<int:id>')(rf.requires_auth(rf.history_detail))
app.route('/kubis/history/delete/<int:id>')(rf.requires_auth(rf.history_delete))

app.route('/kubis/action/add_counting_group', methods=['POST'])(rf.add_counting_group)
app.route('/kubis/action/inference', methods=['POST'])(rf.inference)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# TODO: 
