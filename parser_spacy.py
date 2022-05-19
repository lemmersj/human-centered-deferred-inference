import spacy
from spacy import displacy
from IPython import embed
from copy import deepcopy
nlp = spacy.load('en_core_web_trf')

directional_words = ["on top", "left", "right", "top", "bottom", "from right", "from left", "back", "back row", "row", "closer to", "closer", "front", "in front", "center", "next"]
strip_from_beg_and_end = ["on top of", "top of", "front of"]

def clean_beg_and_end(doc, index_range):
    """removes noise phrases from beginning and end of refexps

    args: 
        doc: the spacy document
        index_range: index range to clean

    returns:
        a revised index range
    """
    # TODO: Add strip from end if necessary.
    removed = True
    while removed:
        removed = False
        for phrase in strip_from_beg_and_end:
            remove_this = True
            if phrase.lower() in str(doc[index_range[0]:index_range[1]]).lower():
                phrase_as_words = phrase.split(" ")
                for phrase_loc in range(len(phrase_as_words)):
                    if phrase_as_words[phrase_loc].lower() != str(doc[index_range[0] + phrase_loc]).lower():
                        remove_this = False
            else:
                remove_this = False
            if remove_this:
                removed = True
                print(f"Removing {phrase}. Old index range: {index_range}")
                index_range[0] = index_range[0] + len(phrase_as_words)
                print(f"New index range: {index_range}")
    return index_range

def remove_nested_phrases(doc, index_ranges):
    """Removes phrases embedded in other phrases.

    args:
        doc: the spacy document
        idx_range_list: the current list of idx ranges.

    returns:
        an updated idx_range_list
    """

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
            new_idx_range_list.append(deepcopy(index_ranges[i]))

    return new_idx_range_list

def expand_based_on_direction_words(doc, start, end, direction_words):
    """expands based on "direction words" (left, right, top, bottom, etc)

    searches from start---the end of the previous noun clause---to end---
    the beginning of the next.

    args:
        doc: spacy document
        start: start index
        end: search end index (inclusive)
        direction_words: words to find

    returns:
        the index corresponding to the last point where a direction word existed.
    """
    last_found = start

    for word in direction_words:
        word_word_count = len(word.split(" "))
        for i in range(start, end+2-word_word_count):
            if str(doc[i:i+word_word_count]).lower() == word:
                if i + word_word_count > last_found:
                    last_found = i+word_word_count

    return last_found

def in_range(i, search_range):
    """checks if i is in search range (inclusive bottom, exclusive top)

    args:
        i: the value we're finding
        search_range: the search range.

    returns:
        bool
    """
    return (i >= search_range[0]) and (i < search_range[1])

def find_verb_indices(doc):
    """Finds indices of verbs.

    args:
        doc: the spacy document

    returns:
        list of ints.
    """
    verb_list = []
    for i in range(len(doc)):
        if doc[i].pos_ == "VERB":
            verb_list.append(i)
    return verb_list

def find_noun_indices(doc):
    """Finds indices of nouns.

    args:
        doc: the spacy document

    returns:
        list of ints.
    """
    noun_list = []
    for i in range(len(doc)):
        if doc[i].pos_ == "NOUN" or doc[i].pos_ == "PROPN" or str(doc[i]) in directional_words:
            # Sometimes spacy misclassifies nouns as propositions.
            # keep an eye on this. Hopefully it doesn't break anything
            # else. Same for directional words.
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

    # Don't expand adjectives (direction words)
    if doc[start_idx].pos_ == "ADJ":
        return all_dependents_list

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
    nouns = find_noun_indices(doc)
    verbs = find_verb_indices(doc)

    if len(verbs) == 0:
        # not a sentence! Don't even bother.
        return [[0, 1], [0, 1]]
    # Some failure cases.
    if len(doc) == 1 or len(nouns) == 0 or len(nouns) == len(doc):
        return [[0, 1], [0, 1]]
    elif len(nouns) == 1:
        propagated = propagate_to_dependents(doc, nouns[0])
        return (propagated, propagated)
    idx_range_list = []
    for noun in nouns:
        index_list = propagate_to_dependents(doc, noun)

        # need to add one to the max value to accomodate for indexing.
        index_range = [min(index_list), max(index_list)+1]
        idx_range_list.append(index_range)

    idx_range_list = remove_nested_phrases(doc, idx_range_list)
    # if we only have one, find a split.
    # split on an adposition that has a noun as a head, and is a head.
    if len(idx_range_list) == 1:
        split_point = None
        for i in range(idx_range_list[0][0], idx_range_list[0][1]):
            if doc[i].pos_ == "ADP":
                if doc[i].head.pos_ == "NOUN":
                    for j in range(len(doc)):
                        if doc[j].head == doc[i]:
                            split_point = i
        idx_range_list = [[idx_range_list[0][0], split_point],[split_point+1, idx_range_list[0][1]]]

    # if the only noun is in the directional words list, join it
    # with the previous group
    new_idx_range_list = []
    if len(idx_range_list) > 2:
        for idx_range in idx_range_list:
            noun_list = []
            for word in range(idx_range[0], idx_range[1]):
                if doc[word].pos_ == "NOUN":
                    noun_list.append(word)
            exists_not_directional = False
            for noun in noun_list:
                if str(doc[noun]) not in directional_words:
                    exists_not_directional = True
            if exists_not_directional:
                new_idx_range_list.append(idx_range)
            else:
                new_idx_range_list[-1][1] = idx_range[1]
        idx_range_list = new_idx_range_list
    new_idx_range_list = []
    skip_next = False
    for i in range(len(idx_range_list)-1):
        if len(idx_range_list) == 2:
            skip_next = True
            new_idx_range_list = idx_range_list
            break
        if skip_next:
            skip_next = False
            continue
        this_start = idx_range_list[i][1]
        this_end = idx_range_list[i+1][0]
        #if this_start == this_end:
        #    new_idx_range_list.append(idx_range_list[i])
        #    continue
        expanded_end = expand_based_on_direction_words(
            doc, this_start, this_end, directional_words)

        if expanded_end == this_end+1:
            expanded_end = idx_range_list[i+1][1]
            skip_next = True

        new_idx_range_list.append([idx_range_list[i][0], expanded_end])

    if not skip_next:
        new_idx_range_list.append(idx_range_list[-1])

    idx_range_list = remove_nested_phrases(doc, new_idx_range_list)

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
                if idx_range_list[i][0] - idx_range_list[j][1] == 1 and doc[idx_range_list[i][0]].dep_ != "dative":
                    one_word_sep.append([i, j])
                    one_word_sep_flat.append(i)
                    one_word_sep_flat.append(j)

        # keep all elements that are not in one_word_sep
        new_idx_range_list = []
        for idx_range_idx in range(len(idx_range_list)):
            if idx_range_idx not in one_word_sep_flat:
                new_idx_range_list.append(idx_range_list[idx_range_idx])

        # if there's only one element that meets our criteria, we can combine
        # but if there's more than one, things can get weird.
        if len(idx_range_list) - len(one_word_sep) == 2:
            for pair in one_word_sep:
                if doc[idx_range_list[pair[1]][1]].pos_ == "ADP":
                    combined_range = idx_range_list[pair[0]] + idx_range_list[pair[1]]
                    new_idx_range_list.append((min(combined_range), max(combined_range)))
        else:
            # Let's run a second directional word pass, but this time to the
            # end of the next noun phrase.
            skip_next = False
            new_idx_range_list = []
            for idx_range_list_idx in range(len(idx_range_list)-1):
                if skip_next:
                    skip_next = False
                    continue
                this_start = idx_range_list[idx_range_list_idx][1]
                this_end = idx_range_list[idx_range_list_idx+1][1]
                if this_start == this_end:
                    new_idx_range_list.append(idx_range_list[idx_range_list_idx])
                    continue
                expanded_end = expand_based_on_direction_words(
                    doc, this_start, this_end, directional_words)

                
                if expanded_end > idx_range_list[idx_range_list_idx+1][0]:
                    expanded_end = idx_range_list[idx_range_list_idx+1][1]
                    skip_next = True

                new_idx_range_list.append([idx_range_list[idx_range_list_idx][0], expanded_end])

            if not skip_next:
                new_idx_range_list.append(idx_range_list[-1])


        idx_range_list = new_idx_range_list
    for i in range(len(idx_range_list)):
        idx_range_list[i] = clean_beg_and_end(doc, idx_range_list[i])

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

    # if all direct objects are in the same noun clause, we can treat
    # them as one direct object for our purposes
    all_in_same = len(direct_objects) > 0 # Handle case where zero
    for index in indices:
        exists_in_index = False
        exists_not_in_index = False
        for direct_object in direct_objects:
            if direct_object >= index[0] and direct_object < index[1]:
                exists_in_index = True
            else:
                exists_not_in_index = True
        
        if exists_in_index and exists_not_in_index:
            all_in_same = False
            break
    # Previous loop also returns true when there's only one d.o.
    if all_in_same:
        returning_default = False
        if direct_objects[0] >= indices[0][0] and direct_objects[0] < indices[0][1]:
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

