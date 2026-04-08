"""
Create a pytorch dataset class for loading scenario files and padding data entries.
"""
import copy
import json
import os
import pathlib
import pickle

import hydra
import numpy as np
from scenarionet import read_dataset_summary, read_scenario
from torch.utils.data import Dataset

from bmt.dataset.preprocessor import preprocess_scenario_description
from bmt.utils import global_config
from bmt.utils import utils

# import lmdb

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
QA_DATASET_MAPPING = {}
ADV_INFO_DICT = {}


class NoMapFeatureError(Exception):
    pass


class LMDBDatasetReader:
    def __init__(self, base_path):
        self.base_path = base_path
        # Load the lookup table that maps sample keys to LMDB file names
        # Search recursively all subfolder to find lookup.json
        self.lookup = {}
        for root, dirs, files in os.walk(self.base_path):
            if "lookup.json" in files:
                lookup_path = os.path.join(root, "lookup.json")
                with open(lookup_path, "r") as f:
                    lookup = json.load(f)
                    self.lookup.update(lookup)
        self.lmdb_cache = {}  # Cache for open LMDB environments


#     def _get_lmdb_env(self, lmdb_name):
#         """Fetches or opens an LMDB environment for reading."""
#         if lmdb_name not in self.lmdb_cache:
#             self.lmdb_cache[lmdb_name] = lmdb.open(lmdb_name, readonly=True)
#         return self.lmdb_cache[lmdb_name]

#     def load_sample(self, key):
#         """Loads a preprocessed sample by key."""
#         lmdb_name = self.lookup.get(key)
#         if lmdb_name is None:
#             raise KeyError(f"Sample {key} not found in lookup.")
#         env = self._get_lmdb_env(lmdb_name)
#         with env.begin() as txn:
#             npz_bytes = txn.get(key.encode('ascii'))
#             if npz_bytes:
#                 with io.BytesIO(npz_bytes) as buffer:
#                     data = np.load(buffer, allow_pickle=True)
#                     return {name: data[name] for name in data.files}  # Return data as a dictionary
#             return None

#     def close(self):
#         """Closes all open LMDB environments."""
#         for env in self.lmdb_cache.values():
#             env.close()


def process_QA_text_label(QA_dict):
    # TODO: do we need to form label for each individual agent? Rightnow it is just a single label
    labels = {}

    env_a = QA_dict['env_a']
    labels['env'] = ' '.join(env_a)

    ego_a = QA_dict['ego_a']
    labels['ego'] = ' '.join(ego_a)

    int_a = QA_dict['int_a']
    labels['int'] = ' '.join(int_a)

    return labels


def get_file_paths(directory):
    file_paths = []
    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the full path and add it to the list
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return


def load_json_to_dict(file_path):
    """
    Load a JSON file into a Python dictionary.

    :param file_path: Path to the JSON file
    :return: Dictionary containing the JSON data
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None


class InfgenDataset(Dataset):
    """
    Infgen dataset class. Returns data_dict for each scenario.
    Init args:
        mode: "training" or "test".
        config: 
            - model: Details about the model architecture.
            - data: Data directories, sample intervals, number of agents, etc.
            - evaluation: predict_all_agents, delete_eval_result (TODO: Add ScenarioDescription passthrough as a flag in the config.)
            - optimization: Training hyperparameters.
            - preprocessing: Max number of agents, map features, traffic lights, padding, etc.
            - root_dir: Self-explanatory.
            - sampling: Inference sampling parameters.
            - tokenization: The part of the config passed to the tokenizer.
    """
    def __init__(self, config, mode, backward_prediction=False):
        super().__init__()
        self.mode = mode
        self.config = config
        dataset_cfg = self.config.DATA

        self.max_map_features = config.PREPROCESSING.MAX_MAP_FEATURES
        self.max_vectors_per_map_feature = config.PREPROCESSING.MAX_VECTORS
        self.max_agents = config.PREPROCESSING.MAX_AGENTS
        self.max_traffic_lights = config.PREPROCESSING.MAX_TRAFFIC_LIGHTS
        self.padding_to_max = config.PREPROCESSING.PADDING_TO_MAX
        self.backward_prediction = backward_prediction

        # We are expecting the data_dir to be either an absolute path or a relative path w.r.t. the repo root.
        if mode == "training":
            self.data_dir = global_config.ROOT_DIR / dataset_cfg.TRAINING_DATA_DIR
        elif mode == "test":
            self.data_dir = global_config.ROOT_DIR / dataset_cfg.TEST_DATA_DIR
        else:
            raise ValueError(f"Unknown mode {mode}.")

        # summary_dict: A dictionary of .pkl filenames to ingest. Filenames (keys) are mapped to metadata objects.
        # summary_list: Keys of summary_dict, in order of ingestion.
        # mapping: A dict mapping scenario IDs to the folder that hosts their files.
        summary_dict, summary_list, mapping = read_dataset_summary(self.data_dir)

        # We might want to use a subset of scenarios.
        if self.mode == "training":
            interval = dataset_cfg.SAMPLE_INTERVAL_TRAINING
        elif self.mode == "test":
            interval = dataset_cfg.SAMPLE_INTERVAL_TEST
        else:
            raise ValueError(f"Unknown mode {self.mode}.")

        if "SD_PASSTHROUGH" in config.DATA:
            self.return_scenario_description = config.DATA["SD_PASSTHROUGH"]
        else:  # Default to False.
            self.return_scenario_description = False

        summary_list = summary_list[::interval]
        # self.data_summary_dict = {k: summary_dict[k] for k in summary_list}
        self.data_mapping = {k: mapping[k] for k in summary_list}
        self.length = len(summary_list)
        self.use_cache_logged = False

        if self.config.BACKWARD_PREDICTION and self.mode == "training":
            self.real_length = self.length
            self.length = self.length * 2

        # Convert each string to sequence of codepoints (integer),
        # and then pack them into a numpy array.
        # NOTE(pzh): I forgot why I wrote this. Seems like some issues in multiprocessing.

        # seqs: A list of np.arrays, each representing the ascii values of a string.
        seqs = [utils.string_to_sequence(s) for s in summary_list]

        # strings_v: ascii values of all strings, concatenated.
        # strings_o: offsets of each string in strings_v.
        if len(seqs) == 0:
            raise ValueError("No scenarios found in the dataset: {}".format(self.data_dir))
        self.strings_v, self.strings_o = utils.pack_sequences(seqs)

        # if self.config.DATA.USE_LMDB and self.mode == "training":
        #     cache_folder = pathlib.Path(self.data_dir) / "cache"
        #     assert cache_folder.is_dir()
        #     self.reader = LMDBDatasetReader(cache_folder)  # LMDB Reader to load samples

        from bmt.tokenization import get_tokenizer
        self.tokenizer = get_tokenizer(config=self.config)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Unpack the stored codepoints at the correct index into a filename string.

        use_backward_prediction = self.backward_prediction
        if self.config.BACKWARD_PREDICTION and self.mode == "training":
            if index >= self.real_length:
                index = index - self.real_length
                use_backward_prediction = True

        seq = utils.unpack_sequence(self.strings_v, self.strings_o, index)
        string = utils.sequence_to_string(seq)
        file_name = string

        try:
            data_dict = self.create_scene_level_data(file_name, index, use_backward_prediction)
        except NoMapFeatureError:
            # This is workaround for Waymo test set where some scenarios do not have map features.
            return self.__getitem__(index + 1)

        # If self.return_scenario_description is true, data_dict has an extra key [raw_scenario_description] that contains the ScenarioDescription object.
        return data_dict

    def create_scene_level_data(self, file_name, index, use_backward_prediction=False):
        """
        Reads a scenario file and preprocesses it.
        """
        assert not self.config.DATA.USE_LMDB, "LMDB is not supported."
        try:
            # scenario: A ScenarioDescription instance.
            cache = None
            scenario = None
            cache_path = None

            if self.config.DATA.USE_CACHE:
                cache_folder = pathlib.Path(self.data_dir) / "cache"
                if cache_folder.is_dir() is False:
                    cache_folder.mkdir(exist_ok=True)

                cache_path = pathlib.Path(self.data_dir) / "cache" / file_name
                if cache_path.is_file():

                    try:
                        with open(cache_path, "rb") as f:
                            cache = pickle.load(f)

                        if self.use_cache_logged is False:
                            print("=====================================")
                            print("=====================================")
                            print("\t*** WARNING ***")
                            print("\tYou are using cache files!!!")
                            print("\tIn folder: ", cache_folder)
                            print("\tThere are ", len(list(cache_folder.glob("*"))), " cache files!!!")
                            print("=====================================")
                            print("=====================================")

                            self.use_cache_logged = True

                        return cache
                    except EOFError as e:
                        print(f"Error in reading cache file: {cache_path=}")

                    scenario = read_scenario(
                        dataset_path=self.data_dir, mapping=self.data_mapping, scenario_file_name=file_name
                    )

                else:
                    scenario = read_scenario(
                        dataset_path=self.data_dir, mapping=self.data_mapping, scenario_file_name=file_name
                    )
                    # print("Cannot find cache file: ", cache_path, "Creating one.")

            else:
                # if self.config.DATA.USE_LMDB and self.mode == "training":
                #     cache = self.reader.load_sample(file_name)
                # else:
                scenario = read_scenario(
                    dataset_path=self.data_dir, mapping=self.data_mapping, scenario_file_name=file_name
                )

        except EOFError as e:
            print(f"{self.data_dir=}, {self.data_mapping=}, {file_name=}")
            raise e
        assert self.mode in ["training", "test"], self.mode
        ret = {}

        if len(scenario["map_features"]) == 0:
            raise NoMapFeatureError

        if self.return_scenario_description:
            ret["raw_scenario_description"] = copy.deepcopy(scenario)

        # TODO: Remove error handling after debugging.
        try:
            preprocessed_scenario_description = preprocess_scenario_description(
                scenario=scenario,
                # cache=cache,
                config=copy.deepcopy(self.config),
                in_evaluation=self.mode != "training",
                keep_all_data=self.config.PREPROCESSING.get("keep_all_data", False),
                backward_prediction=use_backward_prediction,
                tokenizer=self.tokenizer,
                # cache_path=cache_path,
            )
            preprocessed_scenario_description["file_name"] = file_name
        except Exception as e:
            print(f"Error in preprocessing {file_name=}, {index=}, {scenario['id']=}")
            # Ensure that the exception is not swallowed by adding this.
            raise RuntimeError(
                f"{file_name=}, {index=}, {scenario['id']=}. Error in create_scene_level_data: {e}"
            ) from e

        ret.update(preprocessed_scenario_description)
        ret.update({"metadata/scenario_id": scenario['id']})

        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(ret, f)
                # print("Writing cache file: ", cache_path)

        return ret

    def collate_batch(self, batch_list):
        """
        Output format:

        agent_feature:              [B, T, #agents, D]
        agent_feature_position:     [B, T, #agents, 3]
        map_feature:                [B, T, #mapfeat, #points, D]
        map_feature_valid_mask:     [B, T, #mapfeat, #points]
        map_feature_position:       [B, T, #mapfeat, 3]
        """
        data_dict_sample = batch_list[0]

        num_map_feat, num_points, _ = data_dict_sample["encoder/map_feature"].shape

        data_dict = {}
        object_keys = [
            "raw_scenario_description",
            "encoder/track_name",
            "decoder/track_name",
            "eval/track_name",
            # "scenario_id",
            # "in_evaluation"
        ]  # Keys exempt from padding and tensor conversion.

        for k in set(data_dict_sample.keys()):
            if k not in object_keys:
                if not isinstance(data_dict_sample[k], np.ndarray):
                    assert isinstance(data_dict_sample[k], (int, float, bool, str))
                    if isinstance(data_dict_sample[k], str):
                        data_dict[k] = np.array([b[k] for b in batch_list])
                    else:
                        data_dict[k] = utils.numpy_to_torch(np.array([b[k] for b in batch_list]))
                    continue
                # else:
                #     if batch_list[0][k].dtype == np.object:
                #         data_dict[k] = [b[k] for b in batch_list]
                #         continue

                val_list = [utils.numpy_to_torch(b[k]) for b in batch_list]

            # Map features that have vectors' information
            if k in [
                    "encoder/map_feature",
                    "vis/map_feature",
                    "raw/map_feature",
                    "encoder/map_feature_valid_mask",
            ]:
                data_dict[k] = utils.padding_1st_and_2nd_dim(
                    val_list,
                    max_1st_dim=self.max_map_features if self.padding_to_max else None,
                    max_2nd_dim=self.max_vectors_per_map_feature if self.padding_to_max else None
                )

            # Map features that have aggregated info from vectors
            elif k in [
                    "encoder/map_heading",
                    "encoder/map_position",
                    "encoder/map_valid_mask",
            ]:
                data_dict[k] = utils.padding_1st_dim(
                    val_list, max_1st_dim=self.max_map_features if self.padding_to_max else None
                )

            # Traffic light features that have temporal dim
            elif k in [
                    "encoder/traffic_light_feature",
                    "encoder/traffic_light_valid_mask",
            ]:

                if self.config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE:
                    data_dict[k] = utils.padding_1st_dim(
                        val_list, max_1st_dim=self.max_traffic_lights if self.padding_to_max else None
                    )
                else:
                    data_dict[k] = utils.padding_1st_and_2nd_dim(
                        val_list, max_2nd_dim=self.max_traffic_lights if self.padding_to_max else None
                    )

            # Traffic light features that do not have temporal dim
            elif k in [
                    "encoder/traffic_light_position",
                    "encoder/traffic_light_heading",
            ]:
                data_dict[k] = utils.padding_1st_dim(
                    val_list, max_1st_dim=self.max_traffic_lights if self.padding_to_max else None
                )

            # Agent features
            elif k in [
                    "encoder/agent_feature",
                    "encoder/agent_position",
                    "encoder/agent_valid_mask",
                    "encoder/agent_heading",
                    "encoder/agent_velocity",
                    "decoder/modeled_agent_position",
                    "decoder/modeled_agent_heading",
                    "decoder/modeled_agent_velocity",
                    "decoder/modeled_agent_delta",
            ]:
                data_dict[k] = utils.padding_1st_and_2nd_dim(
                    val_list, max_2nd_dim=self.max_agents if self.padding_to_max else None
                )

            # Other data that does not pass the model or does not need regular shapes
            elif k in [
                    # "encoder/modeled_agent_id",
                    # "action_label/labeled_agent_id",
                    "metadata/map_center",  # "decoder/input_step",
                    # "decoder/input_intra_step",
                    "encoder/current_agent_heading",
                    "decoder/current_agent_heading",
                    "encoder/current_agent_shape",
                    "decoder/current_agent_shape",
                    "eval/current_agent_heading",
                    "encoder/current_agent_valid_mask",
                    "decoder/current_agent_valid_mask",
                    "eval/current_agent_valid_mask",
                    # "decoder/current_agent_valid_mask",  #
                    # "decoder/modeled_agent_indices",
                    # For gen model:
                    # "decoder/input_token_valid_mask",
                    # "decoder/should_predict",
                    # "decoder/is_gt",
                    # "eval/should_predict_motion",
            ]:
                data_dict[k] = utils.padding_1st_dim(val_list)

            elif k in [
                    "decoder/input_action_valid_mask",
                    "encoder/current_agent_position",
                    "decoder/current_agent_position",
                    "encoder/current_agent_velocity",
                    "decoder/current_agent_velocity",
                    "decoder/target_action_valid_mask",
                    #"decoder/future_agent_position",
                    #"decoder/future_agent_heading",
                    #"decoder/future_agent_valid_mask",
                    #"decoder/future_agent_velocity",
                    #"encoder/future_agent_position",
                    #"encoder/future_agent_heading",
                    #"encoder/future_agent_valid_mask",
                    #"encoder/future_agent_velocity",
                    "decoder/agent_position",
                    "decoder/agent_heading",
                    "decoder/agent_velocity",
                    "decoder/agent_valid_mask",
                    "eval/agent_velocity",
                    "eval/agent_heading",
                    "eval/agent_position",
                    "eval/agent_valid_mask",
                    "encoder/agent_shape",
                    "decoder/agent_shape",
                    "eval/agent_shape",  # "decoder/target_valid_mask",
                    "decoder/input_agent_motion",
                    "decoder/target_agent_motion",
            ]:
                data_dict[k] = utils.padding_1st_and_2nd_dim(val_list)

            elif k in [
                    "encoder/agent_type",
                    "decoder/agent_type",
                    "encoder/modeled_agent_type",
                    "eval/agent_type",  # "eval/raw_agent_name",
                    "encoder/object_of_interest_name",
                    "decoder/object_of_interest_name",
                    "metadata/sdc_name",  # "eval/modeled_agent_id",
                    "encoder/object_of_interest_id",
                    "decoder/object_of_interest_id",
                    "encoder/modeled_agent_id",  # "decoder/modeled_agent_id",
                    "encoder/agent_id",
                    "decoder/agent_id",
                    "decoder/labeled_agent_id",
                    "decoder/label_turning",
                    "decoder/label_acceleration",
                    "decoder/label_safety",
                    # For gen model:
                    #                   "decoder/input_token_id",
                    #                   "decoder/causal_mask_offset",
            ]:
                data_dict[k] = utils.padding_1st_dim(val_list, fill=-1)

            elif k in [
                    "decoder/input_action",
                    "decoder/target_action",
            ]:
                data_dict[k] = utils.padding_1st_and_2nd_dim(val_list, fill=-1)

            elif k in object_keys:
                # Passthrough: Have the data_dict[object] contain a list of objects.
                data_dict[k] = [b[k] for b in batch_list]

            elif k in [
                    "encoder/sdc_index",
            ]:
                pass

            else:
                raise ValueError("Unknown key: {}".format(k))

        return data_dict


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1009_safety_action_debug.yaml")
def debug(config):
    test_dataset = InfgenDataset(config, "training")
    ddd = iter(test_dataset)
    count = 0
    buggy_count = 0
    while True:
        if count == 3:
            return
        try:
            data = next(ddd)
            count += 1

            assert data['decoder/label_safety'][data['decoder/labeled_agent_id']].sum() > 1

        except StopIteration:
            break

        except AssertionError:
            print("ni collision")
            buggy_count += 1
            print('scenario_id', data['scenario_id'])
            print("data['decoder/label_safety']", data['decoder/label_safety'])
            print("data['decoder/labeled_agent_id']", data['decoder/labeled_agent_id'])
            print("track_name", data['decoder/track_name'][data['decoder/labeled_agent_id']])

    print("buggy_count:", buggy_count)
    print("count", count)
    print("End")


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def read_traffic_light_state(config):
    test_dataset = InfgenDataset(config, "training")
    ddd = iter(test_dataset)
    # count = 0
    # buggy_count = 0

    total_tl = 0
    total_green = 0
    total_yellow = 0
    total_red = 0
    total_unknown = 0
    total_mix = 0
    import tqdm
    for data in tqdm.tqdm(test_dataset):
        tl = data["encoder/traffic_light_feature"]
        mask = data["encoder/traffic_light_valid_mask"]

        for i in range(tl.shape[1]):
            if mask[:, i].any():
                is_green = tl[:, i, 3].astype(bool).any()
                is_yellow = tl[:, i, 4].astype(bool).any()
                is_red = tl[:, i, 5].astype(bool).any()
                is_unknown = tl[:, i, 6].astype(bool).any()

                total_tl += 1
                total_green += is_green
                total_yellow += is_yellow
                total_red += is_red
                total_unknown += is_unknown
                total_mix += (is_green and is_yellow) or (is_green and is_red) or (is_yellow and is_red)

    print("total_tl:", total_tl)
    print("total_green: {}\t{:.4f}".format(total_green, total_green / total_tl))
    print("total_yellow: {}\t{:.4f}".format(total_yellow, total_yellow / total_tl))
    print("total_red: {}\t{:.4f}".format(total_red, total_red / total_tl))
    print("total_unknown: {}\t{:.4f}".format(total_unknown, total_unknown / total_tl))
    print("total_mix: {}\t{:.4f}".format(total_mix, total_mix / total_tl))


if __name__ == '__main__':
    # debug()
    read_traffic_light_state()
