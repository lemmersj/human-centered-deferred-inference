"""
Runs the server.

Typical usage:
    python webapp.py --scenario_category iui_2023_scenario --consent_form regular --rqd_constraint 1

"""
import io
import base64
import argparse
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from flask import Flask, render_template, session, redirect, request
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, SubmitField, RadioField, SelectField
from wtforms.validators import DataRequired
from uniter_interface import UNITERInterface
from scenario_manager import ScenarioManager
from util import calculation_utils

csrf = CSRFProtect()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scenario_category", type=str, required=True)
parser.add_argument("--consent_form", type=str, required=True, choices=["mturk", "regular"])
parser.add_argument("--rqd_constraint", type=int, required=True)
args = parser.parse_args()

# Initialize flask
app = Flask(__name__)

SECRET_KEY = "test_secret_key" #os.environ.get('SECRET_KEY')
app.config['SECRET_KEY'] = SECRET_KEY
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize the interface to UNITER
uniter_interface = UNITERInterface(args.scenario_category)

# Initialize the scenario manager
# this returns images and targets.
scenario_manager = ScenarioManager(args.scenario_category)

class REForm(FlaskForm):
    """
    The flask form for a crop request.
    """
    expression = StringField(
        'request', validators=[DataRequired()],
        default="", render_kw={'autofocus': True})
    submit = SubmitField("Go")

class SurveyForm(FlaskForm):
    """
    The flask form for the post-setting survey.
    """
    accuracy = RadioField(
        'accuracy', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4'),
                 ('5','5'),('6','6'),('7','7')])

    rqr_satisfaction = RadioField(
        'rqr_satisfaction', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4'),
                 ('5','5'),('6','6'),('7','7')])

    submit = SubmitField("Submit Response", render_kw={'disabled': 'disabled'})

class PreSurveyForm(FlaskForm):
    """
    The flask form for the pre-experiment survey.
    """
    age_choices = [*range(18,100)]
    age_choices = list(zip(age_choices, age_choices))
    age_choices = [(None, '')] + age_choices + [("np", "Prefer not to state")]
    age_select = SelectField('age', choices=age_choices,
                             validators=[DataRequired()], default='')
    gender_select = SelectField('gender', choices=[(None,''), ("male","Male"),
                                                   ("female","Female"),
                                                   ("nb","Non-Binary/Other"),
                                                   ("np","Prefer not to state")],
                                validators=[DataRequired()], default='')
    tech_competence = RadioField(
        'tech_competence', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4'),
                 ('5','5'),('6','6'),('7','7')])

    cva_competence = RadioField(
        'cva_competence', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4'),
                 ('5','5'),('6','6'),('7','7')])

    alexa_use = RadioField(
        'alexa_use', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4')])

    siri_use = RadioField(
        'siri_use', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4')])

    google_use = RadioField(
        'google_use', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4')])

    cortana_use = RadioField(
        'cortana_use', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4')])

    bixby_use = RadioField(
        'bixby_use', validators=[DataRequired()],
        choices=[('1','1'),('2','2'),('3','3'),('4','4')])

    submit = SubmitField("Submit Response",render_kw={'disabled': 'disabled'})

@app.route("/", methods=["GET"])
def start_experiment():
    """Decide where to go"""
    # If the user is done, don't let anything else happen.
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")

    # If the user hasn't started, create a tracker.
    if 'user_id' not in session:
        # Don't let mutiple sessions run simultaneously
        if scenario_manager.current_rqr_idx + scenario_manager.current_rqr_idx_count > 0:
            return "Another user is currently in the system. Please talk to the test administrator."
        session['user_id'] = scenario_manager.get_new_user_id()
        session['state'] = "consent_form"
        session['belief'] = None
        session['rq_depth'] = 0

    # If the user has had a cookie set, but isn't on the server, load
    # their tracker from the drive.
    elif 'user_id' in session and scenario_manager.user_id is None:
        print("Loading user")
        scenario_manager.load_user(session['user_id'])

    # Redirect based on the user's state.
    if session['state'] == "consent_form":
        return redirect("consent_form")
    if session['state'] == "instructions":
        return redirect("instructions")
    if session['state'] == "pre_survey":
        return redirect("pre_survey")

    return redirect("interface")

@app.route("/consent_form", methods=['POST', 'GET'])
def render_consent():
    """Loads the consent form. First step in the study.

    args:
        none

    returns:
        rendered consent form. mturk form if mturk, consent_regular if not.
    """
    # If the user finds their way onto this page at the wrong time,
    # redirect them.
    if 'user_id' not in session:
        print("Redirecting to root")
        return redirect("/")
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")

    # Otherwise, load the consent form.
    # The UM IRB requires a different consent form if users are solicited
    # via mechanical turk.
    session['state'] = "consent_form"
    if args.consent_form == "mturk":
        return render_template('consent_mturk.html')
    return render_template('consent_regular.html')

@app.route("/setting")
def render_setting():
    """Tells the user the current setting of the study."""
    if 'user_id' not in session:
        print("Redirecting to root")
        return redirect("/")
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")
    session['state'] = "setting"
    return render_template('setting.html',
                           setting_num=scenario_manager.current_rqr_idx+1)

@app.route("/instructions")
def render_instructions():
    """Draws the instructions. Second step after consent.

    args:
        none

    returns:
        rendered instructions.
    """
    if 'user_id' not in session:
        print("Redirecting to root")
        return redirect("/")
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")
    session['state'] = "instructions"
    return render_template('instructions.html')

@app.route("/validate_mid_survey", methods=['POST'])
@csrf.exempt
def validate_midsurvey():
    """Performs a callback to ensure that the setting survey is filled out before
    enabling the submit button."""
    if len(request.json.keys()) < 3:
        return json.dumps({'valid': False}), 200, {'ContentType':'application/json'}
    return json.dumps({'valid':True}), 200, {'ContentType':'application/json'}

@app.route("/validate_pre_survey", methods=['POST'])
@csrf.exempt
def validate_presurvey():
    """performs a callback to ensure that the pre-survey is filled out before
    enabling teh submit button."""
    if len(request.json.keys()) < 10:
        return json.dumps({'valid': False}), 200, {'ContentType':'application/json'}
    if request.json['age_select'] == "None" or request.json['gender_select'] == "None":
        return json.dumps({'valid': False}), 200, {'ContentType':'application/json'}

    return json.dumps({'valid':True}), 200, {'ContentType':'application/json'}

@app.route("/pre_survey", methods=['POST','GET'])
def render_presurvey():
    """Renders, validates, processes the pre-survey"""

    # Make sure we're at the right spot.
    if session['state'] == "instructions" or session['state'] == "pre_survey":
        session['state'] = "pre_survey"
    else:
        return redirect("/")
    if 'user_id' not in session:
        print("Redirecting to root")
        return redirect("/")
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")
    form = PreSurveyForm()

    # If the form is valid, save the data.
    if form.validate_on_submit():
        to_log = {}
        to_log['age'] = form.age_select.raw_data[0]
        to_log['gender'] = form.gender_select.raw_data[0]
        to_log['tech_competence'] = int(form.tech_competence.raw_data[0])
        to_log['cva_competence'] = int(form.cva_competence.raw_data[0])
        to_log['alexa_use'] = form.alexa_use.raw_data[0]
        to_log['siri_use'] = form.siri_use.raw_data[0]
        to_log['google_use'] = form.google_use.raw_data[0]
        to_log['cortana_use'] = form.cortana_use.raw_data[0]
        to_log['bixby_use'] = form.bixby_use.raw_data[0]
        scenario_manager.log_initial_survey(to_log)

        # And next we tell the user their setting.
        session['state'] = "setting"
        return redirect("setting")
    return render_template("pre_survey.html", form=form)

@app.route("/survey", methods=['POST','GET'])
def render_survey():
    """Renders, processes, saves, the post-setting survey"""
    global scenario_manager

    # Check if we should be accessing this url.
    if 'user_id' not in session:
        print("Redirecting to root")
        return redirect("/")
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")
    form = SurveyForm()

    # If valid, save the data.
    if form.validate_on_submit():
        to_log = {}
        to_log['acc_satisfaction'] = int(form.accuracy.raw_data[0])
        to_log['rqr_satisfaction'] = int(form.rqr_satisfaction.raw_data[0])
        scenario_manager.log_survey(to_log)

        # if we just finished the last setting, exit. Otherwise, update setting.
        if session['state'] == "last_survey":
            session['state'] = "complete"
            scenario_manager = ScenarioManager(args.scenario_category)
            return render_template("complete.html")
        session['state'] = "setting"
        return redirect("setting")

    return render_template("survey.html", form=form,
                           setting=scenario_manager.current_rqr_idx)

@app.route("/interface", methods=['POST', 'GET'])
def render_form():
    """The main interface.

    This abstracts a state machine, with states awaiting_text, awaiting_text_requery,
    infer, and display_text.

    Args:
        None

    Returns:
        A rendered webpage.
    """
    if 'user_id' not in session:
        print("Redirecting to root")
        return redirect("/")
    if 'user_id' in session and session['state'] == "complete":
        return render_template("complete.html")
    form = REForm()

    # If we don't have a user id, get one and start a new session.
    if scenario_manager.user_id is None:
        return redirect("/")

    if session['state'] == "setting":
        session['state'] = "awaiting_text_initial"

    # Get the targets from the scenario manager.
    # This contains all of the information that is necessary to render the
    # webpage.
    image_loc, goal_bbox = scenario_manager.get_targets(
        session['user_id'])
    if image_loc == "COMPLETE":
        session['state'] = "last_survey"
        return redirect("survey")
    if image_loc == "NEW_RQR":
        return redirect("survey")

    # Load the image. This is used in all instances.
    image_in = Image.open(
        f"scenarios/{args.scenario_category}/images/{image_loc}.jpg")

    # Are we in the "infer" state? If so, get the output and determine whether
    # to re-query.
    if session['state'] == "infer":
        # Make sure the form is valid
        if not form.validate_on_submit():
            session['state'] = "awaiting_text"
            return redirect("/interface")

        # Perform the forward pass
        phrase = form.expression.data
        with torch.no_grad():
            scores, bboxes = uniter_interface.forward(
                phrase, image_loc, dropout=True, return_all_boxes=True,
                return_raw_scores=False)

        # If it's the initial query, update the belief.
        if session['belief'] is None:
            session['belief'] = scores.cpu().numpy().tolist()
        else:
            # If it's the deferral response, multiply the two outputs.
            tmp_belief = np.array(session['belief'])*scores.cpu().numpy()
            session['belief'] = (tmp_belief/tmp_belief.sum()).tolist()

        infer_bbox = bboxes[torch.tensor(session['belief']).argmax()]
        # Determine whether or not to defer
        if session['rq_depth'] >= args.rqd_constraint:
            requery = False
        else:
            requery = scenario_manager.requery_fn.should_requery(
                torch.tensor(session['belief']))

        if calculation_utils.computeIoU(goal_bbox, infer_bbox) >= 0.95:
            correct = True
        else:
            correct = False

        # Set the state based on whether or not we are deferring.
        if requery:
            session['state'] = "awaiting_text_requery"
            session['rq_depth'] += 1
        else:
            session['state'] = "display_result"
            scenario_manager.total_inferences += 1
            scenario_manager.correct_inferences += int(correct)
            print(scenario_manager.total_inferences,
                  scenario_manager.correct_inferences)

        # Save the data.
        scenario_manager.log({'img':image_loc, 'target':goal_bbox,
                              'scores': scores.cpu().numpy(),
                              'belief': session['belief'],
                              'depth': session['rq_depth']-float(requery),
                              'phrase':phrase,
                              'rqd_constraint':args.rqd_constraint,
                              'inference_correct':correct})

    # If we are awaiting text for either the first time or the deferral response...
    if "awaiting_text" in session['state']:
        # Draw the target rectangle
        draw = ImageDraw.Draw(image_in)
        draw.rectangle(goal_bbox, outline="#00ff00", width=4)

        output = io.BytesIO()
        image_in.save(output, "PNG")
        encoded_img_data = base64.b64encode(output.getvalue())
        output.seek(0)
        output.truncate(0)

        # Update the prompt based on initial query or deferral response
        if "requery" in session['state']:
            prompt_text = f"I didn't understand \"{phrase}\". Could you try again?"
        else:
            prompt_text = "What would you like to crop?"

        session['state'] = "infer"
        form.expression.data = ""
        correct_pct = 0
        if scenario_manager.total_inferences > 0:
            correct_pct = int(
                scenario_manager.correct_inferences/\
                scenario_manager.total_inferences*1000)/10.

        return render_template(
            'main.html', form=form, img_data=encoded_img_data.decode('utf-8'),
            prompt_text=prompt_text,
            correct=scenario_manager.correct_inferences,
            total=scenario_manager.total_inferences, pct=correct_pct,
            setting=scenario_manager.current_rqr_idx+1,
            length=scenario_manager.targets_per_rqr)

    # If we're displaying a result...
    if session['state'] == "display_result":
        # first draw the target object
        draw = ImageDraw.Draw(image_in)
        draw.rectangle(goal_bbox, outline="#00ff00", width=4)

        # Then do the shading overlay.
        x_grid = torch.arange(image_in.size[0]).unsqueeze(1).repeat(
            1, image_in.size[1])
        y_grid = torch.arange(image_in.size[1]).unsqueeze(0).repeat(
            image_in.size[0], 1)

        overlay = (x_grid > infer_bbox[0]) * (x_grid < infer_bbox[2]) * (
            y_grid > infer_bbox[1]) * (y_grid < infer_bbox[3])
        overlay = 1-overlay.float()
        new_image_array = np.zeros((
            overlay.shape[0], overlay.shape[1], 4))
        new_image_array[:, :, 3] = overlay*200
        new_image = Image.fromarray(
            np.uint8(new_image_array).transpose(1,0,2))
        image_in.paste(new_image, (0, 0), new_image)

        # Encode the output to pass to the browser
        output = io.BytesIO()
        image_in.save(output, "PNG")
        encoded_img_data = base64.b64encode(output.getvalue())

        output.seek(0)
        output.truncate(0)

        has_more = scenario_manager.step(session['user_id'])
        session['belief'] = None
        session['rq_depth'] = 0
        if not has_more:
            return "Complete! Thank you!"
        session['state'] = "awaiting_text_initial"

        # If it's not the first inference, calculate the correct percentage
        if scenario_manager.total_inferences > 0:
            correct_pct = int(scenario_manager.correct_inferences/\
                              scenario_manager.total_inferences*1000)/10.

        # Set message and color based on whether or not inference was correct.
        color = "red"
        correctmessage = "Crop performed incorrectly."
        if correct:
            color = "green"
            correctmessage = "Crop performed correctly."

        # Render the page.
        return render_template('inference.html', form=form,
                               img_data=encoded_img_data.decode('utf-8'),
                               bgcolor=color,
                               correct=scenario_manager.correct_inferences,
                               total=scenario_manager.total_inferences,
                               pct=correct_pct,
                               correctmessage=correctmessage,
                               setting=scenario_manager.current_rqr_idx+1,
                               length=scenario_manager.targets_per_rqr)
    print(session)

if __name__ == '__main__':
    app.run(debug=False,
            ssl_context=('/etc/ssl/certs/lens.cert','/etc/ssl/private/key.pem'),
            port=443, host="0.0.0.0")
