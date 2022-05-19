"""Runs the server."""

from flask import Flask, render_template, flash, session, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from uniter_interface import UNITERInterface
import os
from PIL import Image, ImageDraw
import io
import base64
import spacy
from spacy import displacy
from parser_spacy import extract_all_noun_phrases, separate_pick_and_place
import wordninja
import random
from IPython import embed
import scenario_manager 
import numpy as np

app = Flask(__name__)
SECRET_KEY = os.environ.get('SECRET_KEY')
app.config['SECRET_KEY'] = SECRET_KEY
uniter_interface = UNITERInterface()
nlp = spacy.load('en_core_web_trf')


def get_image_and_bbox():
    all_images = os.listdir("../bottom-up-attention.pytorch/outdir/")
    str_to_return = random.choice(all_images).split(".")[0]

    bboxes = np.load(f"../bottom-up-attention.pytorch/outdir/{str_to_return}.npz")['bbox']

    return str_to_return, bboxes

def get_image():
    return "test_bananastand"

class REForm(FlaskForm):
    expression = StringField('Request', validators=[DataRequired()])
    submit = SubmitField("Go")

@app.route("/", methods=['POST', 'GET'])
def render_form():
    form = REForm()
    if 'mode' not in session:
        session['mode'] = "start"
    
    #if form.validate_on_submit():
    #    session['mode'] = "infer"
    if session['mode'] == "start":
        which_image, which_bbox = get_image_and_bbox()
        session['cur_image'] = which_image
        image_in = Image.open(f"../bottom-up-attention.pytorch/images/{which_image}.jpg")
        prev_image = image_in
        pick_object = random.randint(0, which_bbox.shape[0]-1)
        place_object = random.randint(0, which_bbox.shape[0]-1)
        while place_object == pick_object:
            place_object = random.randint(0, which_bbox.shape[0]-1)

        pick_crop = image_in.crop(which_bbox[pick_object, :])
        place_crop = image_in.crop(which_bbox[place_object, :])
    
        output = io.BytesIO()

        pick_crop.save(output, "JPEG")
        encoded_pick_data = base64.b64encode(output.getvalue())
       
        output.seek(0)
        output.truncate(0)
        place_crop.save(output, "JPEG")
        encoded_place_data = base64.b64encode(output.getvalue())

        output.seek(0)
        output.truncate(0)
        
        image_in.save(output, "JPEG")
        encoded_img_data = base64.b64encode(output.getvalue())
        output.seek(0)
        output.truncate(0)
        session['place_bbox'] = which_bbox[place_object, :].tobytes()
        session['pick_bbox'] = which_bbox[pick_object, :].tobytes()
        #session['place_crop'] = encoded_place_data
        #session['base_image'] = encoded_img_data

        session['mode'] = "infer"
        
        return render_template('main.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'), pick_img=encoded_pick_data.decode('utf-8'), place_img=encoded_place_data.decode('utf-8'))
    
    if session['mode'] == "infer":
        if not form.validate_on_submit():
            session['mode'] = "start"
            return redirect("/")
        print(session.keys())
        image_in = Image.open(f"../bottom-up-attention.pytorch/images/{session['cur_image']}.jpg")
        phrase = " ".join(wordninja.split(form.expression.data))
        doc = nlp(phrase)
        noun_phrases = extract_all_noun_phrases(doc)
        pick_dict = separate_pick_and_place(doc, noun_phrases)
        pick_phrase = pick_dict['pick']
        pick_bbox = uniter_interface.forward(pick_phrase, session['cur_image'])

        place_phrase = pick_dict['place']
        place_bbox = uniter_interface.forward(place_phrase, session['cur_image'])
        flash(f'{pick_phrase}->{place_phrase}')
        pick_crop = image_in.crop(pick_bbox)
        place_crop = image_in.crop(place_bbox)
        output = io.BytesIO()

        pick_crop.save(output, "JPEG")
        encoded_pick_data = base64.b64encode(output.getvalue())
       
        output.seek(0)
        output.truncate(0)
        place_crop.save(output, "JPEG")
        encoded_place_data = base64.b64encode(output.getvalue())
        
        output.seek(0)
        output.truncate(0)
        image_in.save(output, "JPEG")
        encoded_img_data = base64.b64encode(output.getvalue())
       
        pick_target_crop = image_in.crop(np.fromstring(session['pick_bbox'], dtype=np.float32))
        output.seek(0)
        output.truncate(0)
        pick_target_crop.save(output, "JPEG")
        pick_target_data = base64.b64encode(output.getvalue())

        place_target_crop = image_in.crop(np.fromstring(session['place_bbox'], dtype=np.float32))
        output.seek(0)
        output.truncate(0)
        place_target_crop.save(output, "JPEG")
        place_target_data = base64.b64encode(output.getvalue())

        output.seek(0)
        output.truncate(0)
        session['mode'] = "start"
        return render_template('inference.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'), pick_img=encoded_pick_data.decode('utf-8'), place_img=encoded_place_data.decode('utf-8'), target_pick_img=pick_target_data.decode('utf-8'), target_place_img=place_target_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True, port=5151, host="0.0.0.0")
