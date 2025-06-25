from flask import request, session, redirect
from functools import wraps
from os import environ as env

import requests, jwt
from authlib.jose import JsonWebKey, JsonWebToken

class AuthError(Exception):
    def __init__(self, error, status_code):
        self.err = error
        self.status_code = status_code



def validate_token(token):
    headers = {
        "Authorization" : f"Bearer {token}"
    }

    response = requests.get(f"https://{env.get('AUTH0_DOMAIN')}/userinfo", headers=headers)
    if response.status_code == 200:
        userinfo = response.json()
        print(userinfo)
    else:
        raise Exception(f"Error: {response.status_code} - {response.status_code}")
    
    
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'token' not in session:
            return redirect('/login')
        else:
            try:
                validate_token(session['token']['access_token'])
            except Exception as e:
                return redirect('/login')

        return f(*args, **kwargs)
    return decorated