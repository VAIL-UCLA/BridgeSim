"""
Common utilities for dataset converters.
"""
import ast
import copy
import inspect
import logging
import math
import multiprocessing
import os
import os.path as osp
import pickle
import shutil
from functools import partial
from typing import Callable, List

import numpy as np
import psutil
import tqdm
from metadrive.scenario import ScenarioDescription as SD

logger = logging.getLogger(__file__)


# ============================================================================
# Helper functions (inlined from scenarionet.common_utils)
# ============================================================================

def dict_recursive_remove_array_and_set(d):
    """Convert numpy arrays and sets to serializable types."""
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, set):
        return tuple(d)
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = dict_recursive_remove_array_and_set(d[k])
    return d


def save_summary_and_mapping(summary_file_path, mapping_file_path, summary, mapping):
    """Save summary and mapping files."""
    with open(summary_file_path, "wb") as file:
        pickle.dump(dict_recursive_remove_array_and_set(summary), file)
    with open(mapping_file_path, "wb") as file:
        pickle.dump(mapping, file)
    logging.info(
        "\n ================ Dataset Summary and Mapping are saved at: {} "
        "================ \n".format(summary_file_path)
    )


def try_generating_summary(file_folder):
    """Create a summary from scenario files when no summary file exists."""
    files = os.listdir(file_folder)
    summary = {}
    for file in files:
        if SD.is_scenario_file(file):
            with open(osp.join(file_folder, file), "rb+") as f:
                scenario = pickle.load(f)
            summary[file] = copy.deepcopy(scenario[SD.METADATA])
    return summary


def merge_database(
    output_path,
    *dataset_paths,
    exist_ok=False,
    overwrite=False,
    try_generate_missing_file=True,
    filters: List[Callable] = None,
    save=True,
):
    """
    Combine multiple datasets. Each database should have a dataset_summary.pkl

    Args:
        output_path: The path to store the output database
        exist_ok: If True, though the output_path already exist, still write into it
        overwrite: If True, overwrite existing dataset_summary.pkl and mapping.pkl
        try_generate_missing_file: If summary/mapping are missing, try generating them
        dataset_paths: Path of each database
        filters: a set of filters to choose which scenario to select
        save: save to output path immediately

    Returns:
        summary, mapping
    """
    filters = filters or []
    output_abs_path = osp.abspath(output_path)
    os.makedirs(output_abs_path, exist_ok=exist_ok)
    summary_file = osp.join(output_abs_path, SD.DATASET.SUMMARY_FILE)
    mapping_file = osp.join(output_abs_path, SD.DATASET.MAPPING_FILE)
    for file in [summary_file, mapping_file]:
        if os.path.exists(file):
            if overwrite:
                os.remove(file)
            else:
                raise FileExistsError("{} already exists at: {}!".format(file, output_abs_path))

    summaries = {}
    mappings = {}

    # collect
    for dataset_path in tqdm.tqdm(dataset_paths, desc="Merge Data"):
        abs_dir_path = osp.abspath(dataset_path)
        # summary
        assert osp.exists(abs_dir_path), "Wrong database path. Can not find database at: {}".format(abs_dir_path)
        if not osp.exists(osp.join(abs_dir_path, SD.DATASET.SUMMARY_FILE)):
            if try_generate_missing_file:
                summary = try_generating_summary(abs_dir_path)
            else:
                raise FileNotFoundError("Can not find summary file for database: {}".format(abs_dir_path))
        else:
            with open(osp.join(abs_dir_path, SD.DATASET.SUMMARY_FILE), "rb+") as f:
                summary = pickle.load(f)
        intersect = set(summaries.keys()).intersection(set(summary.keys()))
        if len(intersect) > 0:
            existing = []
            for v in list(intersect):
                existing.append(mappings[v])
            logging.warning("Repeat scenarios: {} in : {}. Existing: {}".format(intersect, abs_dir_path, existing))
        summaries.update(summary)

        # mapping
        if not osp.exists(osp.join(abs_dir_path, SD.DATASET.MAPPING_FILE)):
            if try_generate_missing_file:
                mapping = {k: "" for k in summary}
            else:
                raise FileNotFoundError("Can not find mapping file for database: {}".format(abs_dir_path))
        else:
            with open(osp.join(abs_dir_path, SD.DATASET.MAPPING_FILE), "rb+") as f:
                mapping = pickle.load(f)
        new_mapping = {}
        for file, rel_path in mapping.items():
            # mapping to real file path
            new_mapping[file] = os.path.relpath(osp.join(abs_dir_path, rel_path), output_abs_path)

        mappings.update(new_mapping)

    # apply filter stage
    file_to_pop = []
    for file_name in tqdm.tqdm(summaries.keys(), desc="Filter Scenarios"):
        metadata = summaries[file_name]
        if not all([fil(metadata, os.path.join(output_abs_path, mappings[file_name], file_name)) for fil in filters]):
            file_to_pop.append(file_name)
    for file in file_to_pop:
        summaries.pop(file)
        mappings.pop(file)
    if save:
        save_summary_and_mapping(summary_file, mapping_file, summaries, mappings)

    return summaries, mappings


# ============================================================================
# Original utility functions
# ============================================================================

def single_worker_preprocess(x, worker_index):
    """
    All scenarios passed to write_to_directory_single_worker will be preprocessed.
    The input is expected to be a list. The output should be a list too.
    By default, you don't need to provide this processor.
    We override it for waymo convertor to release the memory in time.
    """
    return x


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    """All vec in nuplan should be centered in (0,0) to avoid numerical explosion"""
    vector = np.array(vector)
    vector -= np.asarray(nuplan_center)
    return vector


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
        initial_heading: The initial heading in radians.
        final_heading: The final heading in radians.
        dt: The time interval between the two headings in seconds.

    Returns:
        The angular velocity in radians per second.
    """
    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def mph_to_kmh(speed_in_mph: float):
    """Convert miles per hour to kilometers per hour."""
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh


def contains_explicit_return(f):
    """Check if a function contains an explicit return statement."""
    return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(f))))


def write_to_directory(
    convert_func,
    scenarios,
    output_path,
    dataset_version,
    dataset_name,
    overwrite=False,
    num_workers=8,
    preprocess=single_worker_preprocess,
    **kwargs
):
    """
    Convert scenarios and write to directory with multiple workers.

    Args:
        convert_func: Function to convert each scenario
        scenarios: List of scenarios to convert
        output_path: Output directory path
        dataset_version: Version string for the dataset
        dataset_name: Name of the dataset
        overwrite: Whether to overwrite existing output
        num_workers: Number of parallel workers
        preprocess: Preprocessing function for scenarios
        **kwargs: Additional arguments passed to convert_func
    """
    # make sure dir not exist
    kwargs_for_workers = [{} for _ in range(num_workers)]
    for key, value in kwargs.items():
        for i in range(num_workers):
            kwargs_for_workers[i][key] = value[i]

    save_path = copy.deepcopy(output_path)
    if os.path.exists(output_path):
        if not overwrite:
            raise ValueError(
                "Directory {} already exists! Abort. "
                "\n Try setting overwrite=True or adding --overwrite".format(output_path)
            )
        else:
            shutil.rmtree(output_path)
    os.makedirs(save_path, exist_ok=False)

    basename = os.path.basename(output_path)
    for i in range(num_workers):
        subdir = os.path.join(output_path, "{}_{}".format(basename, str(i)))
        if os.path.exists(subdir):
            if not overwrite:
                raise ValueError(
                    "Directory {} already exists! Abort. "
                    "\n Try setting overwrite=True or adding --overwrite".format(subdir)
                )
    # get arguments for workers
    num_files = len(scenarios)
    if num_files < num_workers:
        # single process
        logger.info("Use one worker, as num_scenario < num_workers:")
        num_workers = 1

    argument_list = []
    output_pathes = []
    num_files_each_worker = int(num_files // num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            end_idx = num_files
        else:
            end_idx = (i + 1) * num_files_each_worker
        subdir = os.path.join(output_path, "{}_{}".format(basename, str(i)))
        output_pathes.append(subdir)
        argument_list.append([scenarios[i * num_files_each_worker:end_idx], kwargs_for_workers[i], i, subdir])

    # prefill arguments
    func = partial(
        writing_to_directory_wrapper,
        convert_func=convert_func,
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
        overwrite=overwrite
    )

    # Run, workers and process result from worker
    with multiprocessing.Pool(num_workers, maxtasksperchild=10) as p:
        ret = list(p.imap(func, argument_list))
        # call ret to block the process
    merge_database(save_path, *output_pathes, exist_ok=True, overwrite=False, try_generate_missing_file=False)


def writing_to_directory_wrapper(
    args, convert_func, dataset_version, dataset_name, overwrite=False, preprocess=single_worker_preprocess
):
    """Wrapper function for multiprocessing."""
    return write_to_directory_single_worker(
        convert_func=convert_func,
        scenarios=args[0],
        output_path=args[3],
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
        overwrite=overwrite,
        worker_index=args[2],
        **args[1]
    )


def write_to_directory_single_worker(
    convert_func,
    scenarios,
    output_path,
    dataset_version,
    dataset_name,
    worker_index=0,
    overwrite=False,
    report_memory_freq=None,
    preprocess=single_worker_preprocess,
    **kwargs
):
    """
    Convert a batch of scenarios (single worker).
    """
    if not contains_explicit_return(convert_func):
        raise RuntimeError("The convert function should return a metadata dict")

    if "version" in kwargs:
        kwargs.pop("version")
        logger.info("the specified version in kwargs is replaced by argument: 'dataset_version'")

    # preprocess
    scenarios = preprocess(scenarios, worker_index)

    save_path = copy.deepcopy(output_path)
    output_path = output_path + "_tmp"
    # meta recorder and data summary
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)

    # make real save dir
    delay_remove = None
    if os.path.exists(save_path):
        if overwrite:
            delay_remove = save_path
        else:
            raise ValueError("Directory already exists! Abort." "\n Try setting overwrite=True or using --overwrite")

    summary_file = SD.DATASET.SUMMARY_FILE
    mapping_file = SD.DATASET.MAPPING_FILE

    summary_file_path = os.path.join(output_path, summary_file)
    mapping_file_path = os.path.join(output_path, mapping_file)

    summary = {}
    mapping = {}

    count = 0
    for scenario in scenarios:
        # convert scenario
        sd_scenario = convert_func(scenario, dataset_version, **kwargs)
        scenario_id = sd_scenario[SD.ID]
        export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)

        if hasattr(SD, "update_summaries"):
            SD.update_summaries(sd_scenario)
        else:
            raise ValueError("Please update MetaDrive to latest version.")

        # update summary/mapping dict
        if export_file_name in summary:
            logger.warning("Scenario {} already exists and will be overwritten!".format(export_file_name))
        summary[export_file_name] = copy.deepcopy(sd_scenario[SD.METADATA])
        mapping[export_file_name] = ""  # in the same dir

        # sanity check
        sd_scenario = sd_scenario.to_dict()
        SD.sanity_check(sd_scenario, check_self_type=True)

        # dump
        p = os.path.join(output_path, export_file_name)
        with open(p, "wb") as f:
            pickle.dump(sd_scenario, f)

        if report_memory_freq is not None and (count) % report_memory_freq == 0:
            print("Current Memory: {}".format(process_memory()))
        count += 1

        if count % 500 == 0:
            logger.info(f"Worker {worker_index} has processed {count} scenarios.")

    # store summary file
    save_summary_and_mapping(summary_file_path, mapping_file_path, summary, mapping)

    # rename and save
    if delay_remove is not None:
        assert delay_remove == save_path
        shutil.rmtree(delay_remove)
    os.rename(output_path, save_path)

    logger.info(f"Worker {worker_index} finished! Files are saved at: {save_path}")


def process_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # mb
