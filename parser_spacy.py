import spacy
from spacy import displacy
from IPython import embed
nlp = spacy.load('en_core_web_trf')

def in_range(i, search_range):
    """checks if i is in search range (inclusive bottom, exclusive top)

    args:
        i: the value we're finding
        search_range: the search range.

    returns:
        bool
    """
    return (i >= search_range[0]) and (i < search_range[1])

def find_noun_indices(doc):
    """Finds indices of nouns.

    args:
        doc: the spacy document

    returns:
        list of ints.
    """
    noun_list = []
    for i in range(len(doc)):
        if doc[i].pos_ == "NOUN" or doc[i].pos_ == "PROPN":
            # Sometimes spacy misclassifies nouns as propositions.
            # keep an eye on this. Hopefully it doesn't break anything
            # else.
            noun_list.append(i)
    return noun_list

def propagate_to_dependents(doc, start_idx):
    """Propagates from a phrase to all its dependents.

    args:
        doc: the spacy document
        start_idx: where we start.

    returns:
        list of all dependent words.
    """
    all_dependents_list = [start_idx]
    updated = True
    while updated:
        updated = False
        for i in range(len(doc)):
            if i in all_dependents_list:
                continue
            if doc[i].head.i in all_dependents_list:
                all_dependents_list.append(i)
                updated = True
    return all_dependents_list
        
def extract_all_noun_phrases(doc):
    """Extracts extended noun chunks.

    This is done by initializing on every noun, and then collecting all words
    dependent on that noun, all words dependent on that word, etc.

    args:
        doc: the spacy document.

    returns:
        list of noun clause indices.
    """
    directional_words = ["left", "right", "top", "bottom", "from right"]
    nouns = find_noun_indices(doc)
    index_ranges = []
    for noun in nouns:
        index_list = propagate_to_dependents(doc, noun)

        # need to add one to the max value to accomodate for indexing.
        index_range = [min(index_list), max(index_list)+1]
        index_ranges.append(index_range)

    # Remove any phrases embedded in other phrases
    to_remove_indices = []
    i = 0
    for idx_range_1 in index_ranges:
        j = 0
        for idx_range_2 in index_ranges:
            if idx_range_1[0] >= idx_range_2[0] and idx_range_1[1] <= idx_range_2[1] and i != j:
                to_remove_indices.append(i)
            j += 1
        i += 1
    
    new_idx_range_list = []
    for i in range(len(index_ranges)):
        if i not in to_remove_indices:
            new_idx_range_list.append(index_ranges[i])

    idx_range_list = new_idx_range_list

    # blend adjacent direction words
    new_idx_range_list = []
    skip = False
    for i in range(len(idx_range_list)-1):
        if skip:
            skip = False
            continue
        for directional_word in directional_words:
            if directional_word in str(doc[idx_range_list[i][1]:idx_range_list[i+1][0]+1]):
                new_idx_range_list.append((
                    idx_range_list[i][0], idx_range_list[i+1][1]))
                skip = True
                break
        if not skip:
            new_idx_range_list.append(idx_range_list[i])
    if not skip:
        new_idx_range_list.append(idx_range_list[-1])
    idx_range_list = new_idx_range_list
    # The length should be two. If not, let's figure it out.
    if len(idx_range_list) > 2:
        # If greater than two, there is probably an adposition between
        # two noun clauses. Most often occurs where "on" occurs twice.
        # e.g., put the mug on the left on the bowl.

        # so find adpositions between two noun clauses.
        # See which noun clauses are separated by one word, and combine if
        # that word is an adpositon.
        one_word_sep = []
        one_word_sep_flat = []
        for i in range(len(idx_range_list)):
            for j in range(len(idx_range_list)):
                if idx_range_list[i][0] - idx_range_list[j][1] == 1:
                    one_word_sep.append([i, j])
                    one_word_sep_flat.append(i)
                    one_word_sep_flat.append(j)

        # keep all elements that are not in one_word_sep
        idx_range_list_new = []
        for i in range(len(idx_range_list)):
            if i not in one_word_sep_flat:
                idx_range_list_new.append(idx_range_list[i])

        # if there's only one element that meets our criteria, we can combine
        # but if there's more than one, things can get weird.
        if len(one_word_sep) == 1:
            for pair in one_word_sep:
                if doc[idx_range_list[pair[1]][1]].pos_ == "ADP":
                    combined_range = idx_range_list[pair[0]] + idx_range_list[pair[1]]
                    idx_range_list_new.append((min(combined_range), max(combined_range)))
            idx_range_list = idx_range_list_new
        else:
            # we have some ambiguity now. If there's a directional word, the
            # phrase probably goes with the one before.
            
            join_indices = []
            join_indices_flat = []
            for pair in one_word_sep:
                for word in directional_words:
                    idx_range = idx_range_list[max(pair)]
                    if word in str(doc[idx_range[0]:idx_range[1]]):
                        join_indices.append(pair)
                        join_indices_flat += pair

            for pair in join_indices:
                combined_range = idx_range_list[pair[0]]+idx_range_list[pair[1]]
                idx_range_list_new.append([min(combined_range), max(combined_range)])
            for i in range(len(idx_range_list)):
                if i not in join_indices_flat:
                    idx_range_list_new.append(idx_range_list[i])

            idx_range_list = idx_range_list_new

    return idx_range_list

def separate_pick_and_place(doc, indices):
    """Choose which noun phrase is the pick, and which is the place.

    args:
        doc: a spacy document.
        indices: the start and end indices of noun phrases. (2x2) list/list/int

    returns:
        dict containing keys pick_range and place_range
    """
    # First make sure that there are two indices
    assert len(indices) == 2

    returning_default = True
    
    # sort the indices
    if indices[0][0] < indices[1][0]:
        first_object = indices[0]
        second_object = indices[1]
    else:
        first_object = indices[1]
        second_object = indices[0]

    # if none of our conditions are met, just return the first one as pick
    # and the second as place.
    pick_object = first_object
    place_object = second_object
    
    # First test: if only one direct object, that is the pick object.
    direct_objects = get_direct_object_indices(doc)
    if len(direct_objects) == 1:
        returning_default = False
        if direct_objects[0] > indices[0][0] and direct_objects[0] < indices[0][1]:
            pick_object = indices[0]
            place_object = indices[1]
        else:
            pick_object = indices[1]
            place_object = indices[0]
    elif len(direct_objects) == 0:
        # set the nominative subject as the pick object.
        for i in range(len(doc)):
            if doc[i].dep_ == 'nsubj':
                pick_object = first_object if in_range(i, first_object) else second_object
                returning_default = False
                break
    elif len(direct_objects) == 2:
        # if a dative exists, and is not right next to its head
        # the target phrase is between it and its head.
        exists_dative = False
        dative_loc = -1
        head_loc = -1
        for i in range(len(doc)):
            if doc[i].dep_ == "dative" and doc[i].head.i != i-1:
                exists_dative = True
                dative_loc = i
                head_loc = doc[i].head.i

        if exists_dative:
            for i in range(head_loc, dative_loc):
                # if the subject here is a pronoun, use the first object.
                if doc[i].pos_ == "PRON":
                    returning_default = False
                    pick_object = first_object
                    place_object = second_object
                elif in_range(i, first_object) or in_range(i, second_object):
                    returning_default = False
                    pick_object = first_object if in_range(i, first_object) else second_object
                    place_object = second_object if in_range(i, first_object) else second_object
        if not exists_dative:
            returning_default = False
            # if there is no dative, the first is the place object
            place_object = first_object
            pick_object = second_object

    if returning_default:
        print(f"RETURNING DEFAULT: {doc}")
    pick_object = str(doc[pick_object[0]:pick_object[1]])
    place_object = str(doc[place_object[0]:place_object[1]])
    return {'pick': pick_object, 'place': place_object}


def get_direct_object_indices(doc):
    direct_object_indices = []
    for i in range(len(doc)):
        if doc[i].dep_ == "dobj" and doc[i].pos_ != "PRON":
            direct_object_indices.append(i)

    return direct_object_indices

def valid_shift(doc, cur_range, direction):
    if direction < 0:
        if cur_range[0] - 1 < 0:
            return False
        proposed_element = doc[cur_range[0]-1]
        if proposed_element.dep_ == "det" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if proposed_element.dep_ == "amod" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if proposed_element.dep_ == "compound" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if proposed_element.pos_ == "ADP" and doc[cur_range[0]].head == proposed_element and doc[cur_range[0]].head.head.pos_ == "NOUN":
            return True
        if proposed_element.pos_ == "NOUN" and doc[cur_range[0]].head == proposed_element:
            return True
        if proposed_element.dep_ == "prep":
            for i in range(cur_range[0], cur_range[1]):
                if doc[i].head == proposed_element:
                    return True
        if proposed_element.pos_ == "DET":
            return True
        if "Tense=Pres" in proposed_element.morph:
            return True
    
    if direction > 0:
        if cur_range[1] >= len(doc):
            return False
        proposed_element = doc[cur_range[1]]

        if proposed_element.dep_ == "prep" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if proposed_element.dep_ == "amod" and proposed_element.head.head.i >= cur_range[0] and proposed_element.head.head.i < cur_range[1]:
            return True
        if proposed_element.dep_ == "pobj" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        #if proposed_element.pos_ == "DET" and proposed_element.head.head.i ==cur_range[1]-1:
        if proposed_element.pos_ == "DET" and proposed_element.head.head.i >= cur_range[0] and proposed_element.head.head.i < cur_range[1]:
            return True
        if proposed_element.dep_ == "acl" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if proposed_element.dep_ == "dobj" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if doc[cur_range[0]].dep_ == "compound" and doc[cur_range[0]].head == proposed_element:
            return True
        if proposed_element.pos_ == "ADJ" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
            return True
        if proposed_element.pos_ == "ADP" and proposed_element.dep_ == "prep" and str(proposed_element.head).upper() == "TAKE":
            # this might be too specific.
           return True
        if proposed_element.pos_ == "NOUN" and proposed_element.dep_ == "relcl" and proposed_element.head.i >= cur_range[0] and proposed_element.head.i < cur_range[1]:
           return True


    return False
def expand_phrase(doc, seed):
    start_idx = seed
    end_idx = seed + 1
    shift_left_valid = True
    shift_right_valid = True
    while shift_left_valid or shift_right_valid:
       shift_left_valid = valid_shift(doc, [start_idx, end_idx], -1)
       shift_right_valid = valid_shift(doc, [start_idx, end_idx], 1)
       if shift_left_valid:
           start_idx -= 1
       if shift_right_valid:
           end_idx += 1

    #if start_idx == 8 and end_idx == 9:
    #    embed()
    return [start_idx, end_idx]

def get_pick_object(doc, return_indices=False):
    direct_objects = get_direct_object_indices(doc)
    if len(direct_objects) == 0:
        # nominative subjects
        nominative_subjects = []
        for i in range(len(doc)):
            if doc[i].dep_ == "nsubj":
                nominative_subjects.append(i)
        if len(nominative_subjects) == 1:
            indices = expand_phrase(doc, nominative_subjects[0])
        else:
            # Failure mode
            indices = [0, 1]
    elif len(direct_objects) == 1:
        indices = expand_phrase(doc, direct_objects[0])
    elif len(direct_objects) == 2:
        # Check for a dative that is not right next to its head.
        exists_dative = False
        dative_loc = -1
        head_loc = -1
        for i in range(len(doc)):
            if doc[i].dep_ == "dative" and doc[i].head.i != i-1:
                exists_dative = True
                dative_loc = i
                head_loc = doc[i].head.i
        # if a dative exists, the target phrase is between it and its head.
        # TODO: If there are two datives?
        if exists_dative:
            # If there's a pronoun, the first indices are probably right.
            indices = expand_phrase(doc, direct_objects[0])
            for target_object in direct_objects:
                if target_object > head_loc and target_object < dative_loc:
                    indices = expand_phrase(doc, target_object)
        else:
            # otherwise, use the last direct object
            indices = expand_phrase(doc, direct_objects[-1])
    else:
        return ""

    if return_indices:
        return indices
    indices = clean_edges(doc, indices) 
    return str(doc[indices[0]:indices[1]])

def clean_edges(doc, edges):
    """certain conditions should be met at the begnning and end of a refexp.
    """
    # Start with the front
    for i in range(edges[0], len(doc)):
        if doc[i].pos_ != "VERB" and doc[i].pos_ != "ADP":
            break
    edges[0] = i

    # and the back now
    for i in range(edges[1]-1, edges[0], -1):
        if doc[i].pos_ != "VERB" and doc[i].pos_ != "ADP":
            break
    edges[1] = i+1

    return edges

def get_place_object(doc):
    # Let's try the easy thing first. Figure out what the pick object is, then
    # find other objects.
    pick_indices = get_pick_object(doc, True)

    noun_indices = []
    # find all the nouns
    for i in range(len(doc)):
        if doc[i].pos_ == "NOUN" and (i < pick_indices[0] or i >= pick_indices[1]):
            noun_indices.append(i)
        if doc[i].dep_ == "dative" and doc[i].pos_ == "PROPN" and (i < pick_indices[0] or i >= pick_indices[1]):
            noun_indices.append(i)

    referring_expressions = []
    for noun_index in noun_indices:
        referring_expressions.append(expand_phrase(doc, noun_index))

    if len(referring_expressions) == 0:
        # failure mode
        referring_expressions = [[0, 1]]
    
    target_refexp = referring_expressions[0]
    for referring_expression in referring_expressions:
        if referring_expression != target_refexp:
            referring_expressions = [[0, 1]]
            # FAILURE MODE

    indices = clean_edges(doc, target_refexp) 
    return str(doc[indices[0]:indices[1]])
