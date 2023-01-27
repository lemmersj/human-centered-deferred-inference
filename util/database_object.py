"""A class we use to connect to and manipulate the database.

Typical usage:
    database = DatabaseObject()
"""
import sqlite3
import numpy as np
from util.database_commands import adapt_array, convert_array,\
        convert_reshape_array,\
        get_failure_mode_dict, get_failure_mode_dict_reversed,\
        get_distribution_dict,\
        get_distribution_dict_reversed

class DatabaseObject():
    """A class that allows for easy access to the database.

    Abstracts away a lot of the setup, and holds temporary tables to speed
    up computation.
    """
    def __init__(self, memory=True):
        """Connects to the database.

        Args:
            memory: Copy the database to memory (faster, no permanent write)?
        """
        # This will probably be useful in other methods
        self.memory = memory

        # We may generate temporary tables to speed up computation.
        # Track them here.
        self.temp_tables = {}

        # We might generate temporary tables to
        # Connect to the database
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("detections", convert_reshape_array)
        sqlite3.register_converter("probabilities", convert_array)
        con_tmp = sqlite3.connect(
            "data/redatabase.sqlite3", detect_types=sqlite3.PARSE_COLNAMES)

        if not memory:
            self.con = con_tmp
        else:
            # Copy db to memory, for faster access.
            self.con = sqlite3.connect(
                ':memory:', detect_types=sqlite3.PARSE_COLNAMES)
            con_tmp.backup(self.con)
            con_tmp.close()

        # This line allows us to access the query results as a dict
        self.con.text_factory = lambda b: b.decode(errors='ignore')
        self.con.row_factory = sqlite3.Row

        # And create the cursor.
        self.cur = self.con.cursor()

        # Keep track of the link between model names and DB indices
        self.model_id_dict = {}

        # Temporary tables that can be queried.
        self.temp_table_list = []

        self.idx_to_failure_dict = get_failure_mode_dict_reversed(self.cur)
        self.failure_to_idx_dict = get_failure_mode_dict(self.cur)
        self.idx_to_distribution_dict = get_distribution_dict(self.cur)
        self.distribution_to_idx_dict = get_distribution_dict_reversed(self.cur)

    def distribution_to_idx(self, distribution):
        """Converts a string distribution to the corresponding db ID.

        args:
            distribution: the distribution in string form.

        returns:
            the database ID for that distribution.
        """
        return self.distribution_to_idx_dict[distribution]

    def idx_to_distribution(self, idx):
        """Converts a database ID to a string distribution.

        args:
            idx: the database ID.

        Returns:
            A string describing the distribution in human terms.
        """
        return self.idx_to_distribution_dict[idx]

    def failure_to_idx(self, failure):
        """Converts a string failure to the corresponding db ID.

        args:
            failure: the failure in string form.

        returns:
            the database ID for that failure.
        """
        return self.failure_to_idx_dict[failure]

    def idx_to_failure(self, idx):
        """Converts a database ID to a string failure.

        args:
            idx: the database ID.

        Returns:
            A string describing the failure in human terms.
        """
        return self.idx_to_failure_dict[idx]

    def get_cursor(self):
        """Gets the cursor.

        Returns:
            The db cursor.
        """
        return self.cur

    def get_architectures_and_sources(self):
        """gets a list of architectures and sources

        Returns:
            a list of architectures and sources."""
        query = "SELECT DISTINCT architecture, object_source FROM models"
        self.cur.execute(query)
        architectures_and_sources = self.cur.fetchall()

        return architectures_and_sources

    def get_model_ids(self, model, obj_src):
        """Get the database IDs corresponding to a model name.

        args:
            model: String database name.
            obj_src: detect or ground truth objects?

        Returns:
            A list of IDs corresponding to model ids in the database.
        """
        if obj_src is None:
            obj_src = "None"
        id_dict_key = f'{model}-{obj_src}'
        # Have we found these IDs already?
        if id_dict_key not in self.model_id_dict.keys():
            if obj_src == "None":
                query = "SELECT id FROM models WHERE architecture=?"
                self.cur.execute(query, (model,))
            else:
                query = "SELECT id FROM models WHERE architecture=? AND object_source=?"
                self.cur.execute(query, (model, obj_src))
            results = self.cur.fetchall()
            self.model_id_dict[id_dict_key] = []
            for row in results:
                self.model_id_dict[id_dict_key].append(row['id'])

        return self.model_id_dict[id_dict_key]

    def get_temp_table(self, network, distribution, split):
        """Retrieves a temporary table containing all outputs (and more) for a
        model, split, object source.

        Creates said table if it does not already exist.

        args:
            network: The model architecture.
            distribution: What distribution are we using?
            split: the data split.

        returns:
            the key of the temp table. If it doesn't exist, create it.
        """
        temp_table_key = f'{split}{network}{distribution}'

        if temp_table_key not in self.temp_table_list:

            # Setting up the appropriate temporary table is many many times faster.
            query = f"CREATE TEMPORARY TABLE IF NOT EXISTS temp.{temp_table_key}"\
                    " AS SELECT outputs.detections AS outputs_detections,"\
                    " outputs.probabilities AS outputs_probabilities, "\
                    "sentences.target as sentence_target, "\
                    "outputs.sentence AS outputs_sentence, "\
                    "outputs.failure_mode AS outputs_failure_mode FROM "\
                    "outputs JOIN sentences on sentences.id=outputs.sentence "\
                    "WHERE outputs.model=? AND outputs.distribution=? AND "\
                    "outputs.split=?"
            self.cur.execute(query, (network, distribution, split))

        return f'temp.{temp_table_key}'

    def get_all_targets(self, network, split, obj_src):
        """Gets all the target ids for a network and split

        Args:
            network: the network architecture.
            split: the data split.
            obj_src: are the objects from ground truth or det?

        Returns:
            a list of target ids corresponding to the inputs.
        """
        model_id = self.get_model_ids(network, obj_src)[0]
        temp_table_key = self.get_temp_table(model_id, 1, split)
        query = f"SELECT DISTINCT sentence_target FROM {temp_table_key}"
        self.cur.execute(query)
        data = self.cur.fetchall()

        target_ids = []
        for row in data:
            target_ids.append(row['sentence_target'])

        return target_ids

    def get_sentence_text(self, sentence_id):
        """Retrieves the text corresponding to a sentence_id

        args:
            sentence_id: the sentence id in the database.

        returns:
            the text (column phrase) corresponding to that id.
        """
        query = "SELECT phrase FROM sentences WHERE id=?"
        self.cur.execute(query, (sentence_id,))
        return self.cur.fetchall()[0]['phrase']

    def get_image_loc_and_target_loc(self, target_id):
        """Gets the image location and target bbox from the target id

        args:
            target_id: the target id

        returns:
            dict with keys tlx, tly, brx, bry, image_loc.
        """
        self.cur.execute("SELECT * FROM targets WHERE id=?", (target_id,))
        return self.cur.fetchall()[0]

    def get_all_re_by_target(self, target, model, distribution, split):
        """Gets all the referring expressions corresponding to a target.

        args:
            target: the target id in the database
            model: the id in the model table
            distribution: the distribution that is used
            split: the split

        returns:
            detections, probabilities, failure mode, and sentence.
        """
        temp_table_key = self.get_temp_table(model, distribution, split)

        query = "SELECT outputs_detections as 'detections [detections]',"\
                " outputs_probabilities as 'probabilities [probabilities]',"\
                "outputs_failure_mode as failure_mode, outputs_sentence as "\
                f"sentence FROM {temp_table_key} WHERE sentence_target=?"
        self.cur.execute(query, (target,))

        return self.cur.fetchall()
def __del__(self):
    """Run before deleting the object.

    Mostly just closes the database."""
    print("Closing database")
    self.con.close()
