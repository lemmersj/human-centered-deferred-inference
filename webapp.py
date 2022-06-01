"""

Runs the server."""

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
from parser_spacy import extract_all_noun_phrases, separate_pick_and_place, make_grammar_worse
import wordninja
import random
from IPython import embed
from scenario_manager import ScenarioManager
import numpy as np
import sys

scenario_category = sys.argv[1]

app = Flask(__name__)
SECRET_KEY = os.environ.get('SECRET_KEY')
app.config['SECRET_KEY'] = SECRET_KEY
uniter_interface = UNITERInterface(scenario_category)
nlp = spacy.load('en_core_web_trf')
scenario_manager = ScenarioManager(scenario_category)

class REForm(FlaskForm):
    expression = StringField('Request', validators=[DataRequired()])
    submit = SubmitField("Go")

@app.route("/", methods=['POST', 'GET'])
def render_form():
    """Loads the main page.

    Args:
        None

    Returns:
        A rendered webpage.
    """
    form = REForm()

    # If we don't have a user id, get one and start a new session.
    if 'user_id' not in session:
        session['user_id'] = scenario_manager.get_new_user_id()
        session['state'] = "start"

    
    #if form.validate_on_submit():
    #    session['mode'] = "infer"
    image_loc, pick_goal_bbox, place_goal_bbox = scenario_manager.get_targets(session['user_id'])

    if image_loc == -1:
        return "All scenarios have been completed. Thank you."

    image_in = Image.open(f"../bottom-up-attention.pytorch/images/{scenario_category}/{image_loc}.jpg")
    if session['state'] == "start":
        pick_crop = image_in.crop(pick_goal_bbox)
        place_crop = image_in.crop(place_goal_bbox)
    
        output = io.BytesIO()

        pick_crop.save(output, "PNG")
        encoded_pick_data = base64.b64encode(output.getvalue())
       
        output.seek(0)
        output.truncate(0)
        place_crop.save(output, "PNG")
        encoded_place_data = base64.b64encode(output.getvalue())

        output.seek(0)
        output.truncate(0)
        
        image_in.save(output, "PNG")
        encoded_img_data = base64.b64encode(output.getvalue())
        output.seek(0)
        output.truncate(0)

        session['state'] = "infer"
        
        return render_template('main.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'), pick_img=encoded_pick_data.decode('utf-8'), place_img=encoded_place_data.decode('utf-8'))
    
    if session['state'] == "infer":
        if not form.validate_on_submit():
            session['state'] = "start"
            return redirect("/")
        # TODO: Decide whether it's worth it to have wordninja running.
        # while it can unmush mushed-together words, it also fails sometimes
        # (rarely) notably on the phrase "sit the white dog on the brown couch"
        # phrase = " ".join(wordninja.split(form.expression.data))
        phrase = form.expression.data
        print(phrase)
        doc = nlp(phrase)
        noun_phrases = extract_all_noun_phrases(doc)
        pick_dict = separate_pick_and_place(doc, noun_phrases)
        pick_phrase = pick_dict['pick']
        scores_pick, pick_infer_bbox = uniter_interface.forward(pick_phrase, image_loc, dropout=True)
        place_phrase = pick_dict['place']
        scores_place, place_infer_bbox = uniter_interface.forward(place_phrase, image_loc, dropout=True)
        print(scores_pick)
        print(scores_place)
        flash(f'{pick_phrase}->{place_phrase}')
        pick_infer_crop = image_in.crop(pick_infer_bbox)
        place_infer_crop = image_in.crop(place_infer_bbox)
        output = io.BytesIO()

        pick_infer_crop.save(output, "PNG")
        encoded_pick_data = base64.b64encode(output.getvalue())
       
        output.seek(0)
        output.truncate(0)
        place_infer_crop.save(output, "PNG")
        encoded_place_data = base64.b64encode(output.getvalue())
        
        output.seek(0)
        output.truncate(0)
        image_in.save(output, "PNG")
        encoded_img_data = base64.b64encode(output.getvalue())
       
        pick_target_crop = image_in.crop(pick_goal_bbox)
        output.seek(0)
        output.truncate(0)
        pick_target_crop.save(output, "PNG")
        pick_target_data = base64.b64encode(output.getvalue())

        place_target_crop = image_in.crop(place_goal_bbox)
        output.seek(0)
        output.truncate(0)
        place_target_crop.save(output, "PNG")
        place_target_data = base64.b64encode(output.getvalue())

        output.seek(0)
        output.truncate(0)

        scenario_manager.add_inference(session['user_id'],
                                      form.expression.data,
                                      pick_dict['pick'],
                                      pick_dict['place'],
                                      scores_pick,
                                      scores_pick,
                                      scores_place,
                                      scores_place)

        has_more = scenario_manager.step(session['user_id'])
        #if not has_more:
        #    return "Complete! Thank you!"
        session['state'] = "start"
        return render_template('inference.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'), pick_img=encoded_pick_data.decode('utf-8'), place_img=encoded_place_data.decode('utf-8'), target_pick_img=pick_target_data.decode('utf-8'), target_place_img=place_target_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True, port=5151, host="0.0.0.0")
