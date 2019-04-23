#!/usr/bin/env python
from flask import Flask, redirect, render_template, request

from io import BytesIO
import urllib.request

from fastai import *
from fastai.vision import *

model_file_url = "https://drive.google.com/uc?export=download&id=1OD6aNXAUJ8jVSpxrYqMHgxMNR64u68iO"
model_file_name = 'melanoma_model.pth'
classes = ['Melanoma', 'NotMelanoma']
#path = Path(__file__).parent
path = Path("models/")

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}


def download_file(url, dest):
    if dest.exists():
        return

    req = urllib.request.Request(url, headers=hdr)
    response = urllib.request.urlopen(req)
    data = response.read()

    with open(dest, 'wb') as f:
        f.write(data)


def setup_learner():
    learn = load_learner(path, model_file_name)
    return learn

# Only allow certain extensions


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a function for if something goes wrong


def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code


# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    """Show a form where you can upload an image of a mole"""

    # Return the template
    return render_template("index.html")


@app.route("/url", methods=["GET"])
def url():
    """Processes and predicts melanoma classification via an url."""

    # If someone requested something
    if request.method == "GET":

        url = request.args.get('url', '')

        # Ensure an url is given
        if not url:
            return apology("Must provide an url", 400)

        else:
            req = urllib.request.Request(url, headers=hdr)
            img = open_image(BytesIO(urllib.request.urlopen(req).read()))
            learn = setup_learner()
            result = learn.predict(img)
            show = []
            show.append(result[0])
            show.append(result[2][result[1]].item())
            return render_template("results.html", data=show)

    # If no request has been made
    else:

        return render_template("index.html")


@app.route("/image", methods=["POST"])
def image():
    """Processes and predicts melanoma classification via an image."""

    # If someone request via POST
    if request.method == "POST":
        # Get the file
        file = request.files['image']

        if not file:
            return apology("Must provide an url", 400)
        elif not allowed_file(file.filename):
            return apology("Not an allowed file extension", 400)
        else:
            img = open_image(BytesIO(file.read()))
            learn = setup_learner()
            result = learn.predict(img)
            show = []
            show.append(result[0])
            show.append(result[2][result[1]].item())
            return render_template("results.html", data=show)

    # If no request has been made return the homepage
    else:
        return render_template("index.html")
