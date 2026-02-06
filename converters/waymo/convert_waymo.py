"""
Waymo to ScenarioNet Converter

Convert Waymo Motion Dataset scenarios to ScenarioNet format for MetaDrive simulation.

Usage:
    python convert_waymo.py --raw_data_path /path/to/waymo/tfrecords --database_path /path/to/output
"""

desc = "Build database from Waymo scenarios"

if __name__ == '__main__':
    import shutil
    import argparse
    import logging
    import os

    # Disable TensorFlow GPU to avoid memory issues during conversion
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")

    from converters.common.utils import write_to_directory
    from converters.waymo.utils import convert_waymo_scenario, get_waymo_scenarios, \
        preprocess_waymo_scenarios

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path", "-d",
        required=True,
        help="Output directory for converted scenarios"
    )
    parser.add_argument(
        "--dataset_name", "-n",
        default="waymo",
        help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument(
        "--version", "-v",
        default='v1.2',
        help="Waymo dataset version"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If the database_path exists, whether to overwrite it"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers to use"
    )
    parser.add_argument(
        "--raw_data_path",
        required=True,
        help="Directory containing Waymo tfrecord files"
    )
    parser.add_argument(
        "--start_file_index",
        default=0,
        type=int,
        help="Start index for file selection. Default: 0."
    )
    parser.add_argument(
        "--num_files",
        default=None,
        type=int,
        help="Number of files to process. Default: None (all files)."
    )
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.version

    if os.path.exists(output_path):
        if not overwrite:
            raise ValueError(
                f"Directory {output_path} already exists! Abort. "
                "\nTry adding --overwrite to overwrite existing data."
            )
        else:
            logger.info(f"Removing existing directory: {output_path}")
            shutil.rmtree(output_path)

    files = get_waymo_scenarios(args.raw_data_path, args.start_file_index, args.num_files)

    logger.info(
        f"Found {len(files)} raw files to process with {args.num_workers} workers "
        f"(~{len(files) / args.num_workers:.1f} files per worker)"
    )

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=files,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers,
        preprocess=preprocess_waymo_scenarios,
    )

    logger.info(f"Conversion complete! Output saved to: {output_path}")
