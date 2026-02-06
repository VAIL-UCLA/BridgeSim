import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_and_prepare_data(filepath: Path, start_index: int, num_samples: int) -> np.ndarray:
    """
    Loads a slice of BEV features from an .npz file and flattens them.

    Args:
        filepath: Path to the .npz file.
        start_index: The starting frame/sample index for the slice.
        num_samples: The number of samples to include in the slice.

    Returns:
        A 2D numpy array of shape (num_samples, num_features), or None if slice is invalid.
    """
    print(f"Loading data from: {filepath}")
    try:
        # Use mmap_mode to avoid loading the whole file into RAM just to slice it
        data = np.load(filepath, mmap_mode='r')
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    if 'bev_features' not in data:
        print(f"Error: 'bev_features' key not found in {filepath}")
        return None

    features = data['bev_features']
    total_samples = features.shape[0]
    print(f"  - File contains {total_samples} total samples.")

    # Validate the requested slice
    if start_index >= total_samples:
        print(f"Error: --start-index ({start_index}) is out of bounds for file with {total_samples} samples.")
        return None
    
    end_index = start_index + num_samples
    if end_index > total_samples:
        print(f"Warning: Requested slice ({start_index} to {end_index}) exceeds file length.")
        end_index = total_samples
        num_samples = total_samples - start_index
        print(f"         Adjusting to use {num_samples} samples instead.")

    # Take the slice and then bring it into memory
    print(f"  - Taking slice of {num_samples} samples starting at index {start_index}.")
    features_sliced = features[start_index : end_index]

    # Reshape the sliced data to (num_samples, C * H * W)
    reshaped_features = features_sliced.reshape(num_samples, -1)

    print(f"  - Reshaped to (num_samples, num_features): {reshaped_features.shape}\n")
    return reshaped_features


def main():
    # parser = argparse.ArgumentParser(
    #     description="""
    #     Perform a pairwise cosine similarity analysis on a SLICE of two sets of BEV embeddings.
    #     This script compares frame `i` from file 1 to frame `i` from file 2 for the given slice.
    #     """,
    #     formatter_class=argparse.RawTextHelpFormatter
    # )
    # parser.add_argument("file1", type=Path, help="Path to the first .npz file.")
    # parser.add_argument("file2", type=Path, help="Path to the second .npz file.")
    # parser.add_argument("-o", "--output", type=Path, default="cosine_similarity_analysis.png", help="Path to save the output plot.")
    # parser.add_argument("--start-index", type=int, default=100, help="Starting sample/frame index for the slice. Default: 100")
    # parser.add_argument("--num-samples", type=int, default=20, help="Number of samples/frames to take in the slice. Default: 20")
    # args = parser.parse_args()

    file1 = Path("/home/abhijit/Work/BridgeSim/metadrive_evaluation/metadrive_uniad_output/bev_features.npz")
    file2 = Path("/home/abhijit/Work/BridgeSim/Bench2Drive/eval_hardbreak_route30_uniad_traj/RouteScenario_30_rep0_Town01_HardBreakRoute_1_3_10_09_19_49_11/bev_features.npz")
    output = Path("tools/cosine_bev_comparison.png")
    start_index = 90
    num_samples = 50
    
    # --- 1. Load Sliced Data ---
    features1 = load_and_prepare_data(file1, start_index, num_samples)
    features2 = load_and_prepare_data(file2, start_index, num_samples)

    if features1 is None or features2 is None:
        print("Could not load data. Exiting.")
        exit(1)

    # Ensure the slices have the same length for a fair pairwise comparison
    min_len = min(features1.shape[0], features2.shape[0])
    if features1.shape[0] != features2.shape[0]:
        print(f"Warning: Slices have different lengths. Truncating to the shorter length of {min_len}.")
        features1 = features1[:min_len]
        features2 = features2[:min_len]

    if min_len == 0:
        print("Error: No samples to compare after slicing. Check start index and file lengths.")
        exit(1)

    # --- 2. Calculate Pairwise Cosine Similarity ---
    print("Calculating pairwise cosine similarity...")
    # The cosine_similarity function computes a matrix. The diagonal of this
    # matrix contains the pairwise similarity between corresponding vectors.
    similarity_matrix = cosine_similarity(features1, features2)
    similarity_scores = np.diag(similarity_matrix)
    print("Calculation complete.")

    # --- 3. Report Statistics ---
    mean_sim = np.mean(similarity_scores)
    median_sim = np.median(similarity_scores)
    min_sim = np.min(similarity_scores)
    max_sim = np.max(similarity_scores)
    
    print("\n--- Cosine Similarity Statistics ---")
    print(f"Frames Compared: {start_index} to {start_index + min_len - 1}")
    print(f"Mean     : {mean_sim:.4f}")
    print(f"Median   : {median_sim:.4f}")
    print(f"Min      : {min_sim:.4f}")
    print(f"Max      : {max_sim:.4f}")
    print("------------------------------------\n")

    # --- 4. Plot the Results ---
    print("Generating plot...")
    frame_indices = np.arange(start_index, start_index + min_len)
    
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(frame_indices, similarity_scores, marker='o', linestyle='-', label='Pairwise Similarity')
    ax.axhline(mean_sim, color='crimson', linestyle='--', label=f'Mean: {mean_sim:.4f}')

    ax.set_title(f'Pairwise Cosine Similarity of BEV Embeddings\n(Frames {start_index} to {start_index + min_len - 1})', fontsize=16)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim(bottom=max(0, min_sim - 0.05), top=min(1, max_sim + 0.05)) # Sensible Y-axis limits
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    # --- 5. Save and Show ---
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output}")
    plt.show()

if __name__ == '__main__':
    main()