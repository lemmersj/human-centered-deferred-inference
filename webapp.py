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
import pdb
from parser_spacy import extract_all_noun_phrases, separate_pick_and_place, make_grammar_worse
import wordninja
import random
from IPython import embed
from scenario_manager import ScenarioManager
import numpy as np
import sys
from requery import RandomRequery, EntropyRequery, AcceptFirstRequery
import torch

scenario_category = sys.argv[1]

app = Flask(__name__)
SECRET_KEY = os.environ.get('SECRET_KEY')
app.config['SECRET_KEY'] = SECRET_KEY
uniter_interface = UNITERInterface(scenario_category)
nlp = spacy.load('en_core_web_trf')
scenario_manager = ScenarioManager(scenario_category)

if sys.argv[2].lower() == "random":
    requery_fn = RandomRequery(0.8) 
if sys.argv[2].lower() == "entropy":
    requery_fn = EntropyRequery(0.2)
if sys.argv[2].lower() == "first":
    requery_fn = AcceptFirstRequery(0)

class REForm(FlaskForm):
    """ The flask form."""
    #expression = StringField('request', validators=[DataRequired()])
    expression_pick = StringField('request_pick', validators=[DataRequired()])
    expression_place = StringField('request_place', validators=[DataRequired()])
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
        session['state'] = "awaiting_text"
        session['pick_belief'] = None
        session['place_belief'] = None
        session['rq_depth'] = 0

    # Get the targets from the scenario manager.
    # This contains all of the information that is necessary to render the
    # webpage.
    image_loc, pick_goal_bbox, place_goal_bbox = scenario_manager.get_targets(
        session['user_id'])

    # the scenario manager returns -1 if there are no more scenarios.
    if image_loc == -1:
        return "All scenarios have been completed. Thank you."

    # Load the image. This is used in all instances.
    image_in = Image.open(
        f"../bottom-up-attention.pytorch/images/{scenario_category}/{image_loc}.jpg")

    # Are we in the "infer" state? If so, get the output and determine whether
    # to re-query.
    if session['state'] == "infer":
        # Make sure the form is valid
        if not form.validate_on_submit():
            session['state'] = "start"
            return redirect("/")
    
        # Get the phrase and extract the two referring expressions
        #phrase = form.expression.data
        #print(phrase)
        #doc = nlp(phrase)
        #noun_phrases = extract_all_noun_phrases(doc)
        #pick_dict = separate_pick_and_place(doc, noun_phrases)

        # Perform inference.
        pick_phrase = form.expression_pick.data #pick_dict['pick']
        place_phrase = form.expression_place.data #pick_dict['place']
        flash(f'{pick_phrase}->{place_phrase}')
        with torch.no_grad():
            scores_pick, pick_bboxes = uniter_interface.forward(pick_phrase, image_loc, dropout=True, return_all_boxes=True, return_raw_scores=False)
            #scores_pick = (scores_pick/5.1).softmax(dim=0).cpu()
            scores_place, place_bboxes = uniter_interface.forward(place_phrase, image_loc, dropout=True, return_all_boxes=True, return_raw_scores=False)
            #scores_place = (scores_place/5.1).softmax(dim=0).cpu()
           

        if session['pick_belief'] is None:
            session['pick_belief'] = scores_pick.cpu().numpy().tolist()
            session['place_belief'] = scores_place.cpu().numpy().tolist()
        else:
            #entropy_prev = torch.tensor(session['pick_belief'])
            #entropy_prev = (-entropy_prev*torch.log(entropy_prev)).sum()
            #entropy_cur = (-scores_pick*torch.log(scores_pick)).sum()
            #if entropy_prev > entropy_cur:
            #    session['pick_belief'] = scores_pick
            
            #entropy_prev = torch.tensor(session['place_belief'])
            #entropy_prev = (-entropy_prev*torch.log(entropy_prev)).sum()
            #entropy_cur = (-scores_place*torch.log(scores_place)).sum()
            #if entropy_prev > entropy_cur:
            #    session['place_belief'] = scores_place
            tmp_pick_belief = np.array(session['pick_belief'])*scores_pick.cpu().numpy()
            session['pick_belief'] = (tmp_pick_belief/tmp_pick_belief.sum()).tolist()

            tmp_place_belief = np.array(session['place_belief'])*scores_place.cpu().numpy()
            session['place_belief'] = (tmp_place_belief/tmp_place_belief.sum()).tolist()
        flash(scores_pick)
        flash(session['pick_belief'])
        flash((-torch.tensor(session['pick_belief'])*torch.log(torch.tensor(session['pick_belief']))).sum())
        flash(scores_place)
        flash(session['place_belief'])
        flash((-torch.tensor(session['place_belief'])*torch.log(torch.tensor(session['place_belief']))).sum())
        pick_infer_bbox = pick_bboxes[torch.tensor(session['pick_belief']).argmax()]
        place_infer_bbox = place_bboxes[torch.tensor(session['place_belief']).argmax()]
        # Determine whether or not to requery
        requery_pick = requery_fn.should_requery(torch.tensor(session['pick_belief']))
        requery_place = requery_fn.should_requery(torch.tensor(session['place_belief']))

        #print(scores_pick, session['pick_belief'], ((-np.array(session['pick_belief'])*np.log(np.array(session['pick_belief'])))).sum())
        #print(scores_place, session['place_belief'], ((-np.array(session['place_belief'])*np.log(np.array(session['place_belief'])))).sum())
        # If we want to requery, set the state.
        if (requery_pick or requery_place) and session['rq_depth'] < 3:
            session['state'] = "awaiting_text_requery"
            session['rq_depth'] += 1
        else:
            session['state'] = "display_result"

    # If we are awaiting text for either the first time or the re-query
    if "awaiting_text" in session['state']:
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

        if "requery" in session['state']:
            robot_img = "robot-requery.png"
        else:
            robot_img = "robot-query.png"

        session['state'] = "infer"
        return render_template('main.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'), pick_img=encoded_pick_data.decode('utf-8'), place_img=encoded_place_data.decode('utf-8'), robot_img=robot_img)
    
    if session['state'] == "display_result":
        # TODO: Decide whether it's worth it to have wordninja running.
        # while it can unmush mushed-together words, it also fails sometimes
        # (rarely) notably on the phrase "sit the white dog on the brown couch"
        # phrase = " ".join(wordninja.split(form.expression.data))
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
                                      f"place the {pick_phrase} in the {place_phrase}",
                                      pick_phrase,
                                      place_phrase,
                                      scores_pick,
                                      scores_pick,
                                      scores_place,
                                      scores_place)

        has_more = scenario_manager.step(session['user_id'])
        session['pick_belief'] = None
        session['place_belief'] = None
        session['rq_depth'] = 0
        #if not has_more:
        #    return "Complete! Thank you!"
        session['state'] = "awaiting_text_initial"
        return render_template('inference.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'), pick_img=encoded_pick_data.decode('utf-8'), place_img=encoded_place_data.decode('utf-8'), target_pick_img=pick_target_data.decode('utf-8'), target_place_img=place_target_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True, port=5151, host="0.0.0.0")
