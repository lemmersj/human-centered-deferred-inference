"""Runs the server."""

from flask import Flask, render_template, flash
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
from IPython import embed

app = Flask(__name__)
SECRET_KEY = os.environ.get('SECRET_KEY')
app.config['SECRET_KEY'] = SECRET_KEY
nlp = spacy.load('en_core_web_trf')
#uniter_interface = UNITERInterface()

def find_imperatives(phrase):
    imperatives = []
    doc = nlp(phrase)
    for i in range(len(doc)):
        if doc[i].pos_ == 'VERB' and ("VerbForm=Inf" in doc[i].morph or "VerbForm=Fin" in doc[i].morph):
            imperatives.append(i)

    return(imperatives)

def find_dependent_word(doc, loc):
    dependent_word = None
    for i in range(len(doc)):
        if (doc[i].dep_ == "dobj" and doc[i].head.i == loc) or (
            doc[i].dep_ == "nsubj" and doc[i].head.i == loc):
            if dependent_word is not None:
                # If there's more than one dependent object, check for a dative.
                # If there is a dative, it's the first D.O. If not, second.
                dative_exists = False
                for i in range(len(doc)):
                    if doc[i].dep_ == 'dative':
                        dative_exists = True
                if not dative_exists:
                    dependent_word = doc[i]
            else:
                dependent_word = doc[i]
    return dependent_word

def find_unused_nouns(doc, used_indices):
    unused_nouns = []
    for i in range(len(doc)):
        if doc[i].pos_ == "NOUN" and i not in used_indices:
            unused_nouns.append(i)

    return unused_nouns

def refexp_one_imperative(phrase):
    doc = nlp(phrase)

    # Only run if there's one imperative, so we can index here.
    loc = find_imperatives(phrase)[0]
    
    # The object to be given is the dependent object of the imperative.
    dependent_word = find_dependent_word(doc, loc)

    if dependent_word is None:
        embed()
    # Find any words related to this word.
    relevant_indices = [dependent_word.i]
    for i in range(len(doc)):
        if doc[i].has_head():
            if doc[i].head == dependent_word:
                relevant_indices.append(i)

    # Add in noun chunks
    for chunk in doc.noun_chunks:
        for index in relevant_indices:
            if index >= chunk.start and index < chunk.end:
                # cheating a little here since we are maxing later.
                if chunk.start not in relevant_indices:
                    relevant_indices.append(chunk.start)
                if chunk.end-1 not in relevant_indices:
                    relevant_indices.append(chunk.end-1)

    # assume contiguous
    object_to_give_string = str(doc[min(relevant_indices):max(relevant_indices)+1])

    # Collect unused nouns
    unused_nouns = find_unused_nouns(doc, [*range(min(relevant_indices),max(relevant_indices)+1)])
    # Find any words related to this word.
    relevant_indices = unused_nouns
    for i in range(len(doc)):
        if doc[i].has_head():
            if doc[i].head.i in relevant_indices:
                relevant_indices.append(i)

    # Add in noun chunks
    for chunk in doc.noun_chunks:
        for index in relevant_indices:
            if index >= chunk.start and index < chunk.end:
                # cheating a little here since we are maxing later.
                if chunk.start not in relevant_indices:
                    relevant_indices.append(chunk.start)
                if chunk.end-1 not in relevant_indices:
                    relevant_indices.append(chunk.end-1)

    # assume contiguous
    try:
        place_to_put_string = str(doc[min(relevant_indices):max(relevant_indices)+1])
    except:
        embed()
    return object_to_give_string, place_to_put_string
    # Remove the object to give string, as a first step for the receiving object.   
    new_string = str(doc[:min(relevant_indices)]) + " " + str(
        doc[max(relevant_indices)+1:])
    new_string = new_string.strip()
    
    # and clean this up.
    doc = nlp(new_string)

    # is there an imperative verb?
    imperatives = find_imperatives(new_string)
    embed()
    receiving_object_list = []
    for i in range(len(doc)):
        if (doc[i].dep_ == "ROOT" or doc[i].dep_ == "dative") and (doc[i].pos_ == "ADP" or doc[i].pos_ == "VERB"):
            continue
        receiving_object_list.append(str(doc[i]))

    receiving_object_string = " ".join(receiving_object_list)

    return object_to_give_string, receiving_object_string

def extract_refexp(phrase):
    """Extracts the referring expressions from a phrase.

    args:
        phrase: the unstructured text.

    returns:
        a dict containing object (object to be retrieved) and target (where
        it is to be placed).
    """
    # First we remove some common extra words.
    remove_list = ["please", "thanks", "thank you", "thankyou"]

    # remove courtesy words
    for word in remove_list:
        phrase = phrase.replace(word, "")

    # tidy text input
    phrase = phrase.strip()

    # First step is to find the imperatives. This forms the basis of our
    # inference.
    imperatives = find_imperatives(phrase)

    # If there's only one imperative, processing should be straightforward.
    if len(imperatives) == 1:
        to_give_string, receiving_object_string  = refexp_one_imperative(phrase)
        return (to_give_string, receiving_object_string)
    return ("", )
    doc = nlp(phrase)
    noun_phrase_starts = []
    noun_phrase_ends = []

    first_phrase_range = 0
    second_phrase_range = len(doc)

    direct_object_count = 0
    direct_object_pronoun_index = None
    direct_object_not_pronoun_index = None
    for i in range(len(doc)):
        if doc[i].dep_ == "dobj":
            direct_object_count += 1
            if doc[i].pos_ == "PRON":
                direct_object_pronoun_index = i
            else:
                direct_object_not_pronoun_index = i
    if direct_object_count == 2 and direct_object_pronoun_index is not None:
        # Substitute the noun phrase for the pronoun
        for chunk in doc.noun_chunks:
            if direct_object_not_pronoun_index > chunk.start and direct_object_not_pronoun_index < chunk.end:
                to_replace_range = [chunk.start, chunk.end]
                break
        new_phrase = f"{str(doc[:direct_object_pronoun_index])} {str(doc[to_replace_range[0]:to_replace_range[1]])} {str(doc[direct_object_pronoun_index + 1:])}"
        doc = nlp(new_phrase)

    # if there is a dative, the target location is second.
    # and any direct objects not related to the dative should be removed.
    dative_loc = None
    for i in range(len(doc)):
        if doc[i].dep_ == "dative":
            dative_loc = i
            break
    if dative_loc is not None:
        new_str = str(doc[doc[dative_loc].head.i:])
        doc = nlp(new_str)
    embed()
    for i in range(len(doc)):
        if doc[i].dep_ == "dobj":
            first_spot = i
            second_spot = doc[i].head.i
            if doc[i].head.pos_ != "VERB":
                print("Something went wrong!")
            put_at_beginning_range = sorted([first_spot, second_spot])
            if put_at_beginning_range[0] != 0:
                new_str = str(doc[put_at_beginning_range[0]:put_at_beginning_range[1]+1]) + " " + str(doc[:put_at_beginning_range[0]])
                doc = nlp(new_str)
            break
    embed()
    # Strip interjections from beginning
    #return displacy.render(doc, jupyter=False, options={'fine_grained': True, 'add_lemma': True, 'collapse_phrases':False})
    for i in range(len(doc)):
        #if not ((doc[i].dep_ == "ROOT" and doc[i].pos_ == "VERB") or doc[i].dep_ == "intj" or doc[i].dep_ == "prt"):
        if not doc[i].dep_ == "intj" or doc[i].dep_ == "prt":
            break
        first_phrase_range += 1

    # Strip interjections from end
    for i in range(len(doc)-1, -1, -1):
        if not (doc[i].dep_ == "intj" or doc[i].dep_=='dep' or doc[i].pos_=="INTJ"):
            break
        second_phrase_range -= 1
    doc = nlp(str(doc[first_phrase_range:second_phrase_range]))
    embed()
    split_start = None
    # See if there's a dative to split on.
    for i in range(len(doc)):
        if doc[i].dep_ == "dative" and doc[i].head.pos_ == "VERB":
            split_start = i
            split_end = i+1
            break
    embed()
    # if not, try splitting on prep->verb/ROOT 
    if split_start is None:
        for i in range(len(doc)):
            if doc[i].dep_ == "prep" and doc[i].head.pos_ == "VERB" and doc[i].head.dep_ == "ROOT":
                split_start = i
                split_end = i+1
                break
    embed()
    # If there's no dative, it will be the noun chunk corresponding to
    # the last direct object.
    #for i in range(len(doc)-1, -1, -1):
    #    if doc[i].dep_ == 'dobj':
    #        for chunk in doc.noun_chunks:
    #            if doc[i].i > chunk.start and doc[i].i < chunk.end:
    #                split_start = chunk.start-1
    #                split_end = chunk.start + 1
    #                break

    # Now remove initial verbs and particles
    sentence_start = 0
    while (doc[sentence_start].pos_ == "VERB" and doc[sentence_start].dep_ == "ROOT") or doc[sentence_start].dep_ == "prt":
        sentence_start += 1
        if split_start is not None:
            # Need to shift the split indices over.
            split_start = split_start - 1
            split_end = split_end - 1
    doc = nlp(str(doc[sentence_start:]))
    # Find the split point: a conjunction that does not link to the preceding
    # word.
    print("---")
    if split_start is None:
        embed()
        print("Split Start")
        split_start = None
        split_end = None
        for i in range(1, len(doc)):
            if doc[i].dep_ == "prep":
                if doc[i].head != doc[i-1]:
                    split_start = i
                    split_end = i+1
                    break
        if split_start == None:
            for i in range(0, len(doc)-1):
                if doc[i].dep_ == "prep":
                    if doc[i].head != doc[i+1]:
                        split_start = i
                        split_end = i+1
                        break

    #return displacy.render(doc, jupyter=False, options={'fine_grained': True, 'add_lemma': True, 'collapse_phrases':False})
    # Remove extra verbage from the center.
    for i in range(split_start):
        if (doc[i].dep_ == 'ROOT' and doc[i].pos=="VERB") or doc[i].dep_ == "cc":
            split_start = i
            break
    for i in range(len(doc)-1, split_end-1, -1):
        if doc[i].dep_ == 'ROOT' or doc[i].dep_ == "cc":
            split_end = i
            break

    #return displacy.render(doc, jupyter=False, options={'fine_grained': True, 'add_lemma': True, 'collapse_phrases':False})
    try:
        return str(doc[:split_start]), str(doc[split_end:])
    except:
        return displacy.render(doc, jupyter=False, options={'fine_grained': True, 'add_lemma': True, 'collapse_phrases':False})


def get_image():
    return "test_bananastand"

class REForm(FlaskForm):
    expression = StringField('Request', validators=[DataRequired()])
    submit = SubmitField("Go")

@app.route("/submit", methods=['POST', 'GET'])
def handle_submitted():
    form = REForm()

    #result = uniter_interface.forward(form.expression.data, get_image())
    return str(result)

@app.route("/", methods=['POST', 'GET'])
def render_form():
    form = REForm()
    image_in = Image.open(f"../bottom-up-attention.pytorch/images/{get_image()}.jpg")
    if form.validate_on_submit():
        #draw = ImageDraw.Draw(image_in)
        #result = uniter_interface.forward(form.expression.data, get_image())
        #draw.rectangle(result)
        flash(f'{extract_refexp(form.expression.data)}')
    output = io.BytesIO()
    image_in.save(output, "JPEG")
    encoded_img_data = base64.b64encode(output.getvalue())
    return render_template('main.html', title="Meet your robot", form=form, img_data=encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True, port=5151, host="0.0.0.0")
