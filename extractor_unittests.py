#from webapp import extract_refexp, find_imperatives
#from extractor_nltk import extract_refexp
#from webapp import find_imperatives
#from IPython import embed

import spacy
from spacy import displacy
from parser_spacy import get_pick_object, get_place_object, extract_all_noun_phrases, separate_pick_and_place

phrases = ["give the green bananas to the man in glasses",
           "to the lady in the yellow scarf give the green bananas",
           "pick up the oranges and give them to the lady in the yellow scarf",
           "put the sandwich in the refrigerator",
           "take the wine glass with red wine and give it to the tall man",
           "take the wine glass with red wine and give it to the man wearing the blue suit",
           "into the wine glass put the oranges",
           "give the man in blue the orange",
           "give the man the orange",
           "the wine goes in the box",
           "give the man wearing blue the orange",
           "see the man in yellow? Give him the banana.",
           "see the banana? give it to the man in yellow.",
           "the man in yellow wants the banana",
           "take the green banana off the table and give it to the lady in grey",
           "give the motorcycle to the woman nearest the camera",
           # "the fuzzy hand into the white bowl", WONTFIX
           # This one might be a parse failure.
           "put the mug on the left on the orange in front",
           "the man in the suit wants to meet the man in the bucket hat.",
           "give the mug on the left to the orange in front"
           #"oranges, lady in yellow" TODO (Maybe)
          ]
pick_objects = ["the green bananas", "the green bananas", "the oranges", "the sandwich", "the wine glass with red wine", "the wine glass with red wine", "the oranges", "the orange", "the orange", "the wine", "the orange", "the banana", "the banana", "the banana", "the green banana off the table", "the motorcycle", #"the fuzzy hand", 
                "the mug on the left", "the man in the suit", "the mug on the left"]

place_objects = ["the man in glasses",
                 "the lady in the yellow scarf",
                 "the lady in the yellow scarf",
                 "the refrigerator",
                 "the tall man",
                 "the man wearing the blue suit",
                 "the wine glass",
                 "the man in blue",
                 "the man",
                 "the box",
                 "the man wearing blue",
                 "the man in yellow",
                 "the man in yellow",
                 "the man in yellow",
                 "the lady in grey",
                 "the woman nearest the camera",
                 #"the white bowl",
                 "the orange in front",
                 "the man in the bucket hat",
                 "the orange in front",
                ]
nlp = spacy.load('en_core_web_trf')
failures = 0
total = 0
for i in range(len(phrases)):
    total += 1
    try:
        phrase_doc = nlp(phrases[i])
        noun_phrases = extract_all_noun_phrases(phrase_doc)
        pick_dict = separate_pick_and_place(phrase_doc, noun_phrases) 
        print(phrase_doc)
        print(f"pick: {phrase_doc[pick_dict['pick'][0]:pick_dict['pick'][1]]}")
        print(f"place: {phrase_doc[pick_dict['place'][0]:pick_dict['place'][1]]}")
        for noun_phrase in noun_phrases:
            print(phrase_doc[noun_phrase[0]:noun_phrase[1]])
        print("---")
        svg = displacy.render(phrase_doc, style="dep")
        with open(f"out_parse/{i}.svg", "w") as out_file:
            out_file.write(svg)
        continue
        picked_object = get_pick_object(phrase_doc)
        assert(picked_object == pick_objects[i])
    except AssertionError:
        failures += 1
        print(f"Failed on {phrases[i]}. Returned {picked_object} instead of {pick_objects[i]}")

print(f"Failed in getting {failures} of {total} pick objects ({failures*100/total})")

failures = 0
total = 0
for i in range(len(phrases)):
    total += 1
    try:
        phrase_doc = nlp(phrases[i])
        placed_object = get_place_object(phrase_doc)
        assert(placed_object == place_objects[i])
    except AssertionError:
        failures += 1
        print(f"Failed on {phrases[i]}. Returned {placed_object} instead of {place_objects[i]}")

print(f"Failed in getting {failures} of {total} target objects ({failures*100/total})")
import sys
sys.exit()
