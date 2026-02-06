import numpy as np
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath: Path, start_index: int, num_samples: int) -> np.ndarray:
    """
    Loads a slice of BEV features from an .npz file and flattens them for t-SNE.

    Args:
        filepath: Path to the .npz file.
        start_index: The starting frame/sample index for the slice.
        num_samples: The number of samples to include in the slice.

    Returns:
        A 2D numpy array of shape (num_samples, num_features).
    """
    print(f"Loading data from: {filepath}")
    try:
        # Use mmap_mode to avoid loading the whole file into RAM just to slice it
        data = np.load(filepath, mmap_mode='r')
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        exit(1)

    if 'bev_features' not in data:
        print(f"Error: 'bev_features' key not found in {filepath}")
        exit(1)

    features = data['bev_features']
    total_samples = features.shape[0]
    print(f"  - File contains {total_samples} total samples.")

    # Validate the requested slice
    if start_index >= total_samples:
        print(f"Error: --start-index ({start_index}) is out of bounds for file with {total_samples} samples.")
        exit(1)
    if start_index + num_samples > total_samples:
        print(f"Warning: Requested slice ({start_index} to {start_index + num_samples}) exceeds file length.")
        num_samples = total_samples - start_index
        print(f"         Adjusting to use {num_samples} samples instead.")

    # Take the slice and then bring it into memory
    print(f"  - Taking slice of {num_samples} samples starting at index {start_index}.")
    features_sliced = features[start_index : start_index + num_samples]

    print(f"  - Original shape of slice: {features_sliced.shape}")

    # Reshape the sliced data to (num_samples, C * H * W)
    reshaped_features = features_sliced.reshape(num_samples, -1)

    print(f"  - Reshaped to (num_samples, num_features): {reshaped_features.shape}\n")
    return reshaped_features


def main():
    """
    Main function to run the t-SNE visualisation.
    """
    file1 = Path("/home/abhijit/Work/BridgeSim/metadrive_evaluation/metadrive_uniad_output/bev_features.npz")
    file2 = Path("/home/abhijit/Work/BridgeSim/Bench2Drive/eval_hardbreak_route30_uniad_traj/RouteScenario_30_rep0_Town01_HardBreakRoute_1_3_10_09_19_49_11/bev_features.npz")
    output = Path("tools/tsne_bev_comparison.png")
    perplexity = 30.0
    start_index = 90
    num_samples = 50
    # parser = argparse.ArgumentParser(
    #     description="""
    #     Perform t-SNE on two sets of BEV features and generate a comparative plot.
    #     The script expects .npz files created by your evaluation pipeline.
    #     """,
    #     formatter_class=argparse.RawTextHelpFormatter
    # )
    # parser.add_argument(
    #     "file1",
    #     type=Path,
    #     help="Path to the first .npz file containing BEV features."
    # )
    # parser.add_argument(
    #     "file2",
    #     type=Path,
    #     help="Path to the second .npz file containing BEV features."
    # )
    # parser.add_argument(
    #     "-o", "--output",
    #     type=Path,
    #     default="tsne_bev_comparison.png",
    #     help="Path to save the output plot image (default: tsne_bev_comparison.png)."
    # )
    # parser.add_argument(
    #     "-p", "--perplexity",
    #     type=float,
    #     default=30.0,
    #     help="t-SNE perplexity parameter (default: 30.0). Typical values are between 5 and 50."
    # )
    # args = parser.parse_args()

    # --- 1. Load Sliced Data ---
    features1 = load_and_prepare_data(file1, start_index, num_samples)
    features2 = load_and_prepare_data(file2, start_index, num_samples)
    num_samples1 = features1.shape[0]

    # Combine into a single dataset
    all_features = np.vstack([features1, features2])
    print(f"Combined feature set has shape: {all_features.shape}")

    # --- 2. Run t-SNE ---
    # Note: Perplexity should be less than the number of samples.
    # With 40 total samples, a perplexity of ~15-30 is reasonable.
    if perplexity >= all_features.shape[0]:
        print(f"Warning: Perplexity ({perplexity}) should be less than the total number of samples ({all_features.shape[0]}).")

    print(f"\nRunning t-SNE... (Perplexity={perplexity})")
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=42,
        init='pca', learning_rate='auto', n_iter=1000, verbose=1
    )
    tsne_results = tsne.fit_transform(all_features)
    print("t-SNE computation complete.")

    # --- 3. Split Results and Plot ---
    results1 = tsne_results[:num_samples1, :]
    results2 = tsne_results[num_samples1:, :]

    print("\nGenerating plot...")
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        results1[:, 0], results1[:, 1], c='royalblue',
        label=f"MetaDrive: {file1.name} (Samples {start_index}-{start_index+num_samples1-1})",
        alpha=0.8, edgecolor='k', s=80
    )
    ax.scatter(
        results2[:, 0], results2[:, 1], c='crimson',
        label=f"Bench2Drive: {file2.name} (Samples {start_index}-{start_index+num_samples1-1})",
        alpha=0.8, edgecolor='k', s=80
    )
    ax.set_title(f't-SNE Visualisation of BEV Embeddings (Sliced Data)', fontsize=16)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()

    # --- 4. Save and Show ---
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output}")
    plt.show()

if __name__ == '__main__':
    main()