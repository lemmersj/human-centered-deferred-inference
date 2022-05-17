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
           "give the mug on the left to the orange in front",
           "give the woman on the bottom right to the third woman from right in the back row",
           "put the mug on the bottom left on the adjacent knife",
           #"oranges, lady in yellow" TODO (Maybe)
          ]
pick_objects = ["the green bananas", 
                "the green bananas", 
                "the oranges",
                "the sandwich",
                "the wine glass with red wine",
                "the wine glass with red wine",
                "the oranges",
                "the orange",
                "the orange",
                "the wine",
                "the orange", 
                "the banana",
                "the banana",
                "the banana",
                "the green banana off the table",
                "the motorcycle",
                "the mug on the left",
                "the mug on the left",
                "the woman on the bottom right",
               "the mug on the bottom left"] #"the fuzzy hand", 

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
                 "the orange in front",
                 "the third woman from right in the back row",
                 "the adjacent knife"
                ]
nlp = spacy.load('en_core_web_trf')
pick_failures = 0
place_failures = 0
total = 0
for i in range(len(phrases)):
    print(phrases[i])
    total += 1
    phrase_doc = nlp(phrases[i])
    noun_phrases = extract_all_noun_phrases(phrase_doc)
    pick_dict = separate_pick_and_place(phrase_doc, noun_phrases) 
    #print(phrase_doc)
    #print(f"pick: {phrase_doc[pick_dict['pick'][0]:pick_dict['pick'][1]]}")
    #print(f"place: {phrase_doc[pick_dict['place'][0]:pick_dict['place'][1]]}")
    #for noun_phrase in noun_phrases:
    #    print(phrase_doc[noun_phrase[0]:noun_phrase[1]])
    #print("---")
    #svg = displacy.render(phrase_doc, style="dep")
    #with open(f"out_parse/{i}.svg", "w") as out_file:
    #    out_file.write(svg)
    picked_object = pick_dict['pick']
    placed_object = pick_dict['place']
    try:
        assert(picked_object == pick_objects[i])
    except AssertionError:
        pick_failures += 1
        print(f"Failed on {phrases[i]}. Returned {picked_object} instead of {pick_objects[i]}")
    try:
        assert(placed_object == place_objects[i])
    except AssertionError:
        place_failures += 1
        print(f"Failed on {phrases[i]}. Returned {placed_object} instead of {place_objects[i]}")


print(f"Failed in getting {pick_failures} of {total} pick objects ({pick_failures*100/total})")

print(f"Failed in getting {place_failures} of {total} place objects ({place_failures*100/total})")
