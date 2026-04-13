import argparse
import os
from tqdm import tqdm
import copy
import pickle

def get_filenames(folder_path, prefix="sd_"):
    all_files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)) and file.startswith(prefix):
            all_files.append(os.path.join(folder_path, file))
                          
    return all_files



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="The data folder storing raw tfrecord from Waymo dataset.")
    parser.add_argument("--num_scenario", type=int, default=1)
    parser.add_argument("--save_dir", required=True, help="The place to store .pkl file if different from given in --dir")
    parser.add_argument("--TF_mode", type=str, default="no_TF")
    parser.add_argument("--num_mode", type=int, default=1)

    args = parser.parse_args()

    from bmt.rl_train.train.scgen_generator import SCGEN_Generator
    generator = SCGEN_Generator()
    num_modes = args.num_mode
    num_scenarios = args.num_scenario
    save_dir = args.save_dir
    TF_mode = args.TF_mode

    all_scenario_files = get_filenames(args.dir)

    for i in tqdm(range(num_scenarios)):

        SD_path = all_scenario_files[i % num_scenarios]

        with open(SD_path, "rb") as f:
            SD = pickle.load(f)

        sid = SD["id"]

        for i in range(num_modes):
    
            SCGEN_SD = generator.generate_from_raw_SD(scenario_data=copy.deepcopy(SD), track_length=91)

            if SCGEN_SD is None:
                break # parking scene

            with open(f"{save_dir}/sd_{sid}_SCGEN_{TF_mode}_mode_{i}.pkl", "wb") as f:
                pickle.dump(SCGEN_SD, f)


    


if __name__ == '__main__':
    main()


