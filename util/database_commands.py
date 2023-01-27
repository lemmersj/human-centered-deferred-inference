"""Function wrappers for common SQL commands."""
import sqlite3
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from util import calculation_utils

coco_dir = "/z/dat/mscoco/images/train2014/"

def get_all_re_by_target_notemp(target_id, model, distribution, split, cur):
    """Gets all of the model outputs corresponding to the target id.

    This is different from get_all_re_by_target in that it does not reference
    temp.temptable (which may not reference the right model and dist). This
    will be deprecated when everything is running through util.database_object

    args:
        target_id: Database ID of the target.
        model: Database ID of the model.
        distribution: Database ID of the distribution.
        split: the split. Val, testA, or testB.
        cur: the DB cursor.

    Returns:
        A list containing all responses matching the target/network.
    """
    # Setting up the appropriate temporary table is many many times faster.
    query = "DROP TABLE IF EXISTS temp.temptable2_join1"
    cur.execute(query)
    query = "DROP TABLE IF EXISTS temp.temptable2"
    cur.execute(query)

    # Setting up the appropriate temporary table is many many times faster.
    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable2_join1"\
            "AS SELECT outputs.detections AS outputs_detections,"\
            " outputs.probabilities AS outputs_probabilities, "\
            "sentences.target as sentence_target, "\
            "outputs.sentence AS outputs_sentence, "\
            "outputs.failure_mode AS outputs_failure_mode FROM "\
            "outputs JOIN sentences on sentences.id=outputs.sentence "\
            "WHERE outputs.model=? AND outputs.distribution=? AND "\
            "outputs.split=?"
    cur.execute(query, (model, distribution, split))

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable2 "\
            "AS SELECT targets.tlx AS tlx, targets.tly AS tly, targets.brx "\
            "AS brx, targets.bry AS bry, "\
            "temp.temptable2_join1.outputs_detections AS outputs_detections, "\
            "temp.temptable2_join1.outputs_probabilities AS "\
            "outputs_probabilities, temp.temptable2_join1.sentence_target "\
            "as sentence_target, temp.temptable2_join1.outputs_sentence AS "\
            "outputs_sentence, temp.temptable2_join1.outputs_failure_mode AS "\
            "outputs_failure_mode FROM temp.temptable2_join1 JOIN targets "\
            "on temp.temptable2_join1.sentence_target = targets.id"
    cur.execute(query)

    query = "SELECT outputs_sentence, outputs_failure_mode, "\
            "outputs_probabilities AS 'probabilities [probabilities]', "\
            "outputs_detections AS 'detections [detections]', tlx, tly, "\
            "brx, bry FROM temp.temptable2 WHERE sentence_target=?"
    cur.execute(query, (target_id,))
    response = cur.fetchall()
    if len(response) == 0:
        print("RESPONSE OF LENGTH ZERO")

    return response

def get_re_by_target_notemp(target_id, model, distribution, split, cur):
    """Get a single referring expression for all models, without referencing
    the temp table.

    This is different from get_re_by_target in that it does not reference
    temp.temptable (which may not reference the right model and dist). This
    will be deprecated when everything is running through util.database_object

    args:
        target_id: Database ID of the target.
        model: Database ID of the model.
        distribution: Database ID of the distribution.
        split: the split. Val, testA, or testB.
        cur: the DB cursor.

    Returns:
        A randomly chosen response matching the target/network.
    """
    # Setting up the appropriate temporary table is many many times faster.
    # Speed doesn't matter so much here, but creating these tables makes
    # the queries more readable.
    query = "DROP TABLE IF EXISTS temp.temptable2_join1"
    cur.execute(query)
    query = "DROP TABLE IF EXISTS temp.temptable2"
    cur.execute(query)

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable2_join1 "\
            "AS SELECT outputs.detections AS outputs_detections, "\
            "outputs.probabilities AS outputs_probabilities, "\
            "sentences.target as sentence_target, outputs.sentence "\
            "AS outputs_sentence, outputs.failure_mode AS "\
            "outputs_failure_mode FROM outputs JOIN sentences on "\
            "sentences.id=outputs.sentence WHERE outputs.model=? AND "\
            "outputs.distribution=? AND outputs.split=?"
    cur.execute(query, (model, distribution, split))

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable2 AS "\
            "SELECT targets.tlx AS tlx, targets.tly AS tly, targets.brx "\
            "AS brx, targets.bry AS bry, "\
            "temp.temptable2_join1.outputs_detections AS "\
            "outputs_detections, "\
            "temp.temptable2_join1.outputs_probabilities AS "\
            "outputs_probabilities, temp.temptable2_join1.sentence_target "\
            "as sentence_target, temp.temptable2_join1.outputs_sentence AS "\
            "outputs_sentence, temp.temptable2_join1.outputs_failure_mode "\
            "AS outputs_failure_mode FROM temp.temptable2_join1 JOIN "\
            "targets on temp.temptable2_join1.sentence_target = targets.id"
    cur.execute(query)

    query = "SELECT outputs_sentence, outputs_failure_mode, "\
            "outputs_probabilities AS 'probabilities [probabilities]', "\
            "outputs_detections AS 'detections [detections]', tlx, tly, "\
            "brx, bry FROM temp.temptable2 WHERE sentence_target=?"
    cur.execute(query, (target_id,))
    response = cur.fetchall()
    if len(response) == 0:
        print("RESPONSE OF LENGTH ZERO")
    choice = random.choice(response)

    return choice

def get_re_by_target(target_id, cur):
    """Get a single referring expression from the model in the temp table.

    The temp table must be generated by get_random_draw for this to work.
    Otherwise, use get_re_by_target_notemp.

    args:
        target_id: Database ID of the target.

    Returns:
        A randomly chosen response matching the target/network.
    """

    query = "SELECT outputs_sentence, outputs_failure_mode, "\
    "outputs_probabilities AS 'probabilities [probabilities]', "\
    "outputs_detections AS 'detections [detections]', tlx, tly, brx, bry "\
            "FROM temp.temptable WHERE sentence_target=?"
    cur.execute(query, (target_id,))
    response = cur.fetchall()
    if len(response) == 0:
        print("RESPONSE OF LENGTH ZERO")
    choice = random.choice(response)

    return choice

def get_all_re_by_target(target_id, cur):
    """Get a all referring expressions for a target from the model
    in the temp table.

    The temp table must be generated by get_random_draw for this to work.
    Otherwise, use get_re_by_target_notemp.

    args:
        target_id: Database ID of the target.

    Returns:
        All responses matching the target/network.
    """

    query = "SELECT outputs_sentence, outputs_failure_mode, "\
            "outputs_probabilities AS 'probabilities [probabilities]', "\
            "outputs_detections AS 'detections [detections]', tlx, tly, "\
            "brx, bry FROM temp.temptable WHERE sentence_target=?"
    cur.execute(query, (target_id,))
    response = cur.fetchall()
    if len(response) == 0:
        print("RESPONSE OF LENGTH ZERO")

    return response

def get_converged_accuracy(model, distribution, split, cur, method="combined"):
    """Gets the converged accuracy for a model, distribution, and split.

    In this case, the converged guess is simply the guess produced by the
    product of the distributions produced by all refexps.

    Args:
        model: the model ID
        distribution: which distribution we're using
        split: val, testA, testB
        cur: the cursor
        method: combined or mean

    Returns:
        The converged accuracy of that model, split, and distribution.
    """

    # This temptable makes it easier to get the right info.
    query = "DROP TABLE IF EXISTS temp.temptable_join1"
    cur.execute(query)
    query = "DROP TABLE IF EXISTS temp.temptable"
    cur.execute(query)

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable_join1 "\
            "AS SELECT outputs.detections AS outputs_detections, "\
            "outputs.probabilities AS outputs_probabilities, "\
            "sentences.target as sentence_target, outputs.sentence AS "\
            "outputs_sentence, outputs.failure_mode AS outputs_failure_mode "\
            "FROM outputs JOIN sentences on sentences.id=outputs.sentence "\
            "WHERE outputs.model=? AND outputs.distribution=? AND "\
            "outputs.split=?"
    cur.execute(query, (model, distribution, split))

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable AS SELECT "\
            "targets.tlx AS tlx, targets.tly AS tly, targets.brx AS brx, "\
            "targets.bry AS bry, temp.temptable_join1.outputs_detections AS "\
            "outputs_detections, temp.temptable_join1.outputs_probabilities "\
            "AS outputs_probabilities, temp.temptable_join1.sentence_target "\
            "as sentence_target, temp.temptable_join1.outputs_sentence AS "\
            "outputs_sentence, temp.temptable_join1.outputs_failure_mode AS "\
            "outputs_failure_mode FROM temp.temptable_join1 JOIN targets on "\
            "temp.temptable_join1.sentence_target = targets.id"
    cur.execute(query)

    # Get all of the targets
    query = "SELECT DISTINCT sentence_target FROM temp.temptable"
    cur.execute(query)
    targets = cur.fetchall()

    if len(targets) == 0:
        return -1
    distribution_dict = get_distribution_dict(cur)

    # Loop through all the targets, and track how many end up being correct.
    correct = 0
    total = 0
    for target in targets:
        target_reduced = target[0]
        # get every output for this target/model.
        all_targets = get_all_re_by_target(target_reduced, cur)

        # Find the product of all targets.
        # For variation ratio, add a small value to prevent zero arrays.
        if method != "consensus":
            probability_product = all_targets[0]['probabilities'].copy()
        else:
            probability_product = np.zeros(all_targets[0]['probabilities'].shape)
            probability_product[all_targets[0]['probabilities'].argmax()] += 1

        if "varratio" in distribution_dict[distribution]:
            probability_product += 1e-6
        for i in range(1, len(all_targets)):
            if method == "mean":
                probability_product = probability_product+all_targets[i]['probabilities']
            elif method == "combined":
                probability_product = probability_product*all_targets[i]['probabilities']
            elif method == "consensus":
                probability_product[all_targets[i]['probabilities'].argmax()] += 1
            elif method == "smart":
                prev_entropy = -(probability_product*np.log(probability_product)).sum()
                cur_entropy = -(all_targets[i]['probabilities']*np.log(
                    all_targets[i]['probabilities'])).sum()

                if cur_entropy < prev_entropy:
                    probability_product = all_targets[i]['probabilities']

        # Find the guess.
        guess = probability_product.argmax()

        # Figure out whether or not it's correct.
        # Start by getting the detected bboxes.
        converted_boxes = convert_boxes_to_corners(
            all_targets[0]['detections'])
        # Then calculate the IoUs with the ground truth.
        ious = calculation_utils.compute_all_IoUs(
            (all_targets[0]['tlx'], all_targets[0]['tly'],
             all_targets[0]['brx'], all_targets[0]['bry']), converted_boxes)

        if ious[guess] > 0.5:
            correct += 1
        total += 1

    # Return the accuracy.
    return correct*1.0/total

def get_random_draw(model, distribution, split, cur):
    """Draws one referring expression for every object.

    Args:
        model: the model ID
        distribution: which distribution we're using
        split: val, testA, testB

    Returns:
        A dict with keys 'target_ids', 'failure_modes', 'probabilities',
        'detections' containing target ID, failure mode, probability, and
        detections respectively.
    """

    query = "DROP TABLE IF EXISTS temp.temptable_join1"
    cur.execute(query)
    query = "DROP TABLE IF EXISTS temp.temptable"
    cur.execute(query)

    # Setting up the appropriate temporary table is many many times faster.
    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable_join1 "\
            "AS SELECT outputs.detections AS outputs_detections, "\
            "outputs.probabilities AS outputs_probabilities, "\
            "sentences.target as sentence_target, outputs.sentence AS "\
            "outputs_sentence, outputs.failure_mode AS outputs_failure_mode "\
            "FROM outputs JOIN sentences on sentences.id=outputs.sentence "\
            "WHERE outputs.model=? AND outputs.distribution=? AND "\
            "outputs.split=?"
    cur.execute(query, (model, distribution, split))

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable AS SELECT "\
            "targets.tlx AS tlx, targets.tly AS tly, targets.brx AS brx, "\
            "targets.bry AS bry, temp.temptable_join1.outputs_detections AS "\
            "outputs_detections, temp.temptable_join1.outputs_probabilities "\
            "AS outputs_probabilities, temp.temptable_join1.sentence_target "\
            "as sentence_target, temp.temptable_join1.outputs_sentence AS "\
            "outputs_sentence, temp.temptable_join1.outputs_failure_mode AS "\
            "outputs_failure_mode FROM temp.temptable_join1 JOIN targets on "\
            "temp.temptable_join1.sentence_target = targets.id"
    cur.execute(query)

    # Get all of the targets
    query = "SELECT DISTINCT sentence_target FROM temp.temptable"
    cur.execute(query)
    targets = cur.fetchall()
    responses = {'target_ids': [], 'failure_modes':[],
                 'probabilities':[], 'detections': []}

    # Now for every target, pick one sentence.
    for target in targets:
        target_reduced = target[0]
        responses['target_ids'].append(target_reduced)
        choice = get_re_by_target(target_reduced, cur)
        responses['failure_modes'].append(choice['outputs_failure_mode'])
        responses['probabilities'].append(choice['probabilities'])
        responses['detections'].append(choice['detections'])

    # Return the target value.
    return responses

def get_matching_ground_truth_net(model, cur):
    """Given a model with object source det, returns the same model with
    source gt (i.e., same weights, different object source).

    Args:
        model: the ID of the model with source det.
        cur: the database cursor.

    Returns:
        The ID of the model corresponding to the input model, with source gt.
    """
    query = "SELECT architecture, instance FROM models WHERE id=?"
    cur.execute(query, (model, ))
    model_id = cur.fetchall()

    query = "SELECT id FROM models where architecture=? AND "\
            "instance=? AND object_source='gt'"
    cur.execute(query, model_id[0])

    return cur.fetchall()

def get_all_draw(model, distribution, split, cur):
    """All referring expressions for every object.

    Args:
        model: the model ID
        distribution: which distribution we're using
        split: val, testA, testB

    Returns:
        A dict with keys 'target_ids', 'failure_modes', 'probabilities',
        'detections' containing target ID, failure mode, probability, and
        detections respectively.
    """

    query = "DROP TABLE IF EXISTS temp.temptable_join1"
    cur.execute(query)
    query = "DROP TABLE IF EXISTS temp.temptable"
    cur.execute(query)

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable_join1 AS "\
            "SELECT outputs.detections AS outputs_detections, "\
            "outputs.probabilities AS outputs_probabilities, "\
            "sentences.target as sentence_target, outputs.sentence AS "\
            "outputs_sentence, outputs.failure_mode AS outputs_failure_mode "\
            "FROM outputs JOIN sentences on sentences.id=outputs.sentence "\
            "WHERE outputs.model=? AND outputs.distribution=? AND "\
            "outputs.split=?"
    cur.execute(query, (model, distribution, split))

    query = "CREATE TEMPORARY TABLE IF NOT EXISTS temp.temptable AS SELECT "\
            "targets.tlx AS tlx, targets.tly AS tly, targets.brx AS brx, "\
            "targets.bry AS bry, temp.temptable_join1.outputs_detections AS "\
            "outputs_detections, temp.temptable_join1.outputs_probabilities "\
            "AS 'outputs_probabilities [probabilities]', "\
            "temp.temptable_join1.sentence_target as sentence_target, "\
            "temp.temptable_join1.outputs_sentence AS outputs_sentence, "\
            "temp.temptable_join1.outputs_failure_mode AS "\
            "outputs_failure_mode FROM temp.temptable_join1 JOIN targets on "\
            "temp.temptable_join1.sentence_target = targets.id"
    cur.execute(query)

    # Get all of the targets
    query = "SELECT * FROM temp.temptable"
    cur.execute(query)
    return cur.fetchall()

def does_correct_exist(target_id, model, distribution, split, cur):
    """ For a target, check whether there is a referring expression
    in the dataset that results in the correct answer.

    args:
        target_id: the target id

    returns:
        true if a referring expression exists for a correct answer,
        false otherwise.
    """

    # for now select from the temptable, but I suppose there
    # isn't really a guarantee that it exists.
    query = "SELECT outputs_sentence, outputs_failure_mode FROM "\
            "temp.temptable WHERE sentence_target=? AND outputs_model=? "\
            "AND outputs_distribution=? AND outputs_split=?"
    cur.execute(query, (int(target_id), model, distribution, split))
    response = cur.fetchall()

    for row in response:
        if row['outputs_failure_mode'] == 5:
            return True

    return False

def convert_boxes_to_corners(box_list):
    """Converts boxes from tlx tly width height to brx bry

    args:
        box_list: a list of boxes in the width height format

    returns:
        the same boxes with brx bry.
    """
    return_list = []
    for i in range(box_list.shape[0]):
        return_list.append([box_list[i][0],
                            box_list[i][1],
                            box_list[i][0]+box_list[i][2],
                            box_list[i][1]+box_list[i][3]])
    return return_list

def check_render(image_loc, bboxes, target_bbox=None):
    """Renders an image with its detections.

    The goal of this function is to ensure that the bounding boxes and
    image are all in the same format.

    args:
        image_loc: Where the image is located on disk
        bboxes: bounding boxes to draw.
        target_bbox: the target bounding box

    returns:
        none
    """
    try:
        image_in = Image.open(coco_dir+image_loc)
    except:
        image_loc = "_".join(image_loc.split("_")[:-1])+".jpg"
        image_in = Image.open(coco_dir+image_loc)

    draw_image = ImageDraw.Draw(image_in)
    for row in bboxes:
        draw_image.rectangle((row[0],row[1],row[2],row[3]))

    if target_bbox is not None:
        draw_image.rectangle(target_bbox, outline="red")

    #fig, ax = plt.subplots()
    #ax.imshow(image_in)
    plt.imshow(image_in)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

def adapt_array(arr):
    """Converts an array to bytes to store in database.

    args:
        arr: the numpy array.

    returns:
        a bytestring to store in the array.
    """
    #out = io.BytesIO()
    #np.save(out, arr)
    #out.seek(0)
    return arr.tobytes()

def convert_array(text):
    """Converts bytes from the DB to an array.

    args:
        text: a bytestring

    returns:
        a numpy array.
    """
    #print(text)
    #print("---")
    return np.frombuffer(text, dtype=np.float32)

def convert_reshape_array(text):
    """Converts bytes from the DB to an array.

    args:
        text: a bytestring

    returns:
        a numpy array.
    """
    #print(text)
    #print("---")
    return np.frombuffer(text, dtype=np.float32).reshape(-1, 4)

def get_distribution_dict(cur):
    """Get a dictionary of distributions

    args:
        cur: the database cursor.

    returns:
        a dictionary where you can look up distribution name using id #
    """

    # Select all the distributions
    dist_dict = {}
    query = "SELECT id, name FROM distributions"
    cur.execute(query)
    failure_modes = cur.fetchall()
    for failure_mode in failure_modes:
        dist_dict[failure_mode[0]] = failure_mode[1]
    return dist_dict

def get_distribution_dict_reversed(cur):
    """Get a dictionary of distributions

    args:
        cur: the database cursor.

    returns:
        a dictionary where you look up id # using distribution name
    """

    # Select all the distributions
    dist_dict = {}
    query = "SELECT id, name FROM distributions"
    cur.execute(query)
    failure_modes = cur.fetchall()
    for failure_mode in failure_modes:
        dist_dict[failure_mode[1]] = failure_mode[0]
    return dist_dict

def get_correct_outputs_one_run(model, split, distribution, cur=None):
    """Gets all the correct outputs for one run

        args:
            model: the id of the target model
            split: val, testA, testB
            distribution: softmax, ..., varratio
            cur: sqlite3 cursor. If none, a new connection will be made.

        returns:
            all the correct outputs for that model, split, distribution.
    """
    if cur is None:
        con = sqlite3.connect("data/redatabase.sqlite3")
        cur = con.cursor()

    query = "SELECT id FROM failure_modes WHERE name=?"
    cur.execute(query, ("correct",))
    correct_idx = cur.fetchall()[0][0]
    if isinstance(distribution, str):
        query = "SELECT id FROM distributions WHERE name=?"
        cur.execute(query, (distribution,))
        distribution = cur.fetchall()[0][0]
    query = "SELECT * FROM outputs WHERE model=? AND distribution=? and split=? and failure_mode=?"
    cur.execute(query, (model, distribution, split, correct_idx))
    return cur.fetchall()

def get_outputs_one_run(model, split, distribution, cur):
    """Gets all the outputs for one run

        args:
            model: the id of the target model
            split: val, testA, testB
            distribution: softmax, ..., varratio
            cur: sqlite3 cursor. If none, a new connection will be made.

        returns:
            all the outputs for that model, split, distribution.
    """
    if isinstance(distribution, str):
        query = "SELECT id FROM distributions WHERE name=?"
        cur.execute(query, (distribution,))
        distribution = cur.fetchall()[0][0]
    query = "SELECT * FROM outputs WHERE model=? AND distribution=? and split=?"
    cur.execute(query, (model, distribution, split))

    return cur.fetchall()

def get_target_from_sentence(sentence_id, cur):
    """Gets the target given a sentence id

    args:
        sentence_id: the sentence id

    returns:
        the value of the "target" column for that sentence.
    """
    query = "SELECT target FROM sentences WHERE id=?"
    cur.execute(query, (sentence_id,))

    return cur.fetchall()[0]['target']
def get_failure_mode_dict(cur):
    """Get the dictionary of failure modes.

    args:
        cur: the database cursor

    returns:
        dict where the failure mode name is the key, and id is the val.
    """
    failure_mode_dict = {}
    query = "SELECT * FROM failure_modes"
    cur.execute(query)
    failure_modes = cur.fetchall()
    for failure_mode in failure_modes:
        failure_mode_dict[failure_mode[1]] = failure_mode[0]
    return failure_mode_dict

def get_failure_mode_dict_reversed(cur):
    """Get the reversed dictionary of failure modes.

    args:
        cur: the database cursor

    returns:
        dict where the id is the key, and the failure mode name is the the val.
    """
    failure_mode_dict = {}
    query = "SELECT * FROM failure_modes"
    cur.execute(query)
    failure_modes = cur.fetchall()
    for failure_mode in failure_modes:
        failure_mode_dict[failure_mode[0]] = failure_mode[1]
    return failure_mode_dict

def get_all_models(cur):
    """Get a listing of all the models in the database

    args:
        cur: the sqlite3 cursor.

    returns:
        list of tuples corresponding to database rows.
    """
    query = "SELECT * FROM models"
    cur.execute(query)
    data = cur.fetchall()

    return data

def add_row_to_detections(model_arch, model_instance, obj_src,
                          sentence, distribution_type, detections, output,
                          failure_mode):
    """Adds a row to the detection dict

    Args:
        model_arch: string model architecture
        model_instance: unique id for that instance of the model.
        obj_src: detector or gt
        sentence: the sentence id (int)
        distribution: the distribution type (int)
        detections: the numpy array of detections
        output: the model output
        failure_mode: the failure_mode

    returns:
        Nothing
    """
    con = sqlite3.connect("data/redatabase.sqlite3")
    cur = con.cursor()

    query = "SELECT id FROM models WHERE architecture=? AND instance=? "\
            "AND object_source=?"
    cur.execute(query, (model_arch, model_instance, obj_src))
    target_model_id = cur.fetchall()

    if len(target_model_id) == 0:
        query = "INSERT INTO models(architecture, instance, object_source) VALUES (?, ?, ?)"
        cur.execute(query, (model_arch, model_instance, obj_src))
        cur.commit()
        query = "SELECT id FROM models WHERE architecture=? AND instance=? AND object_source=?"
        cur.execute(query, (model_arch, model_instance, obj_src))
        target_model_id = cur.fetchall()

    query = "SELECT id FROM distribution_type WHERE name=?"
    cur.execute(query, (distribution_type))
    target_distribution_id = cur.fetchall()

    if len(target_distribution_id) == 0:
        query = "INSERT INTO distributions(name) VALUES (?)"
        cur.execute(query, (distribution_type, ))
        cur.commit()
        query = "SELECT id FROM distribution_type WHERE name=?"
        cur.execute(query, (distribution_type))
        target_distribution_id = cur.fetchall()

def add_rows_from_dataset_entry(dataset_entry, mscoco, cur):
    """Adds a row of data from the dataset entry.

    This is a row directly out of the refs(unc).py file, which will contain
    one object, but multiple referring expressions.

    args:
        mscoco:
        dataset_entry: an entry from aforementioned list.

    returns:
        None
    """
    if dataset_entry["split"] == "train":
        return

    # First place the image file in the images table.
    # Necessary to have these checks, since multiple objects exist in
    # the same image.

    # Insert the target object into the database.
    target = mscoco[dataset_entry['ann_id']]
    query_string = "insert into targets(id, tlx, tly, brx, bry, image_loc) "\
            "VALUES (?, ?, ?, ?, ?, ?)"
    cur.execute(query_string, (dataset_entry['ann_id'], target[0],
                               target[1], target[0]+target[2],
                               target[1]+target[3],
                               dataset_entry["file_name"]))

    # Insert the sentence into the referring expression.
    for sentence in dataset_entry['sentences']:
        query_string = "insert into sentences(id, phrase, "\
                "phrase_formatted, target) VALUES (?, ?, ?, ?)"
        collapsed_string = sentence['raw'].lower()
        collapsed_string = collapsed_string.replace("w/ ","with")
        for char in [".",",",";"," ","/","'","!","-","?",")",
                   "(",":","&","@","#","$","%","^","*","<",
                   ">","\\","`","~","\""]:
            collapsed_string = collapsed_string.replace(char, "")
        cur.execute(query_string, (sentence['sent_id'], sentence['raw'],
                                   collapsed_string, dataset_entry['ann_id']))

def get_random_gt_render():
    """Returns the data required to draw a random referring expression.

    args none

    returns:
        dict containing raw sentence, image location, and gt bbox.
    """
    # open the database
    con = sqlite3.connect("data/redatabase.sqlite3")
    cur = con.cursor()

    # Pick the sentence.
    query_string = "SELECT * FROM sentences ORDER BY random() LIMIT 1"
    cur.execute(query_string)
    response = cur.fetchall()
    phrase = response[0][1]

    # Find the corresponding target object.
    query_string = "SELECT * from targets WHERE id=?"
    cur.execute(query_string, (response[0][3],))
    target_response = cur.fetchall()[0]

    con.close()

    return {"phrase": phrase,
            "tlx": target_response[1],
            "tly": target_response[2],
            "brx": target_response[3],
            "bry": target_response[4],
            "image": target_response[5]}
