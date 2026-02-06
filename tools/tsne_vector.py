import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import glob

def run_tsne_and_plot_images(image_vectors_1, image_vectors_2, perplexity, output_path):
    """
    Takes aggregated image vectors from two sources, runs t-SNE, and plots the result.
    Each point in the t-SNE plot now represents an entire image.
    """
    print("\n--- Starting t-SNE Analysis on Image Vectors ---")
    
    # --- 1. Combine and Prepare Final Data ---
    combined_image_vectors = np.vstack([image_vectors_1, image_vectors_2])

    # Create labels to identify which source each image came from
    labels = np.array([0] * len(image_vectors_1) + [1] * len(image_vectors_2))
    
    num_images_processed = combined_image_vectors.shape[0]
    print(f"Total image vectors being processed: {num_images_processed}")
    print(f"Each image vector has {combined_image_vectors.shape[1]} dimensions.")

    # --- 2. Run t-SNE ---
    # With very high-dimensional data points (like flattened images),
    # t-SNE can be sensitive. `init='pca'` is often recommended.
    print("Running t-SNE... This will take some time, potentially longer than pixel-wise t-SNE.")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000, # Number of iterations, often needs more for high dim data
        learning_rate='auto',
        init='pca', # PCA initialization can help in high-dimensional spaces
        verbose=1   # Provides progress updates
    )
    tsne_results = tsne.fit_transform(combined_image_vectors)

    # --- 3. Visualise the Results ---
    print("Creating plot...")
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Separate the results for each image source
    results_img1_src = tsne_results[labels == 0]
    results_img2_src = tsne_results[labels == 1]

    # Create scatter plots with distinct, uniform colours
    ax.scatter(
        results_img1_src[:, 0], results_img1_src[:, 1],
        c='royalblue',
        label='MetaDrive Images',
        alpha=0.8,
        s=50, # Larger markers since each represents an entire image
        marker='o'
    )
    ax.scatter(
        results_img2_src[:, 0], results_img2_src[:, 1],
        c='darkorange',
        label='Bench2Drive Images',
        alpha=0.8,
        s=50,
        marker='X'
    )

    ax.set_title(f't-SNE Visualisation of Image-Vectors (Perplexity: {perplexity})', fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)

    ax.legend(title="Image Source", loc='best', markerscale=1.5)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

def main():
    """
    Main function to find image pairs, flatten them into vectors, and run image-wise t-SNE.
    """
    # --- Configuration ---
    folder_path_1 = '/home/abhijit/Work/BridgeSim/metadrive_evaluation/metadrive_uniad_output/raw_camera_images/CAM_FRONT/'
    folder_path_2 = '/home/abhijit/Work/BridgeSim/Bench2Drive/Bench2Drive-mini/HardBreakRoute_Town01_Route30_Weather3/camera/rgb_front/'
    
    start_image_index = 0   # Start from the first image (00000.jpg)
    end_image_index = 200 # Up to (but not including) this index. So 0-199 images.
    
    output_path = 'tools/tsne_image_vectors_distribution.png'
    image_process_size = (100, 100) # Resize all images to this for consistent vector length
    perplexity = 30 # Adjust based on your data and number of images

    try:
        # --- Find and Pair Images ---
        print("Finding and pairing images...")
        paths1 = sorted(glob.glob(os.path.join(folder_path_1, '*.jpg')))
        paths2 = sorted(glob.glob(os.path.join(folder_path_2, '*.jpg')))

        if not paths1 or not paths2:
            print("Error: One or both folders do not contain any .jpg images.")
            return

        num_available = min(len(paths1), len(paths2))
        print(f"Found {len(paths1)} images in folder 1 and {len(paths2)} in folder 2.")
        print(f"Total available pairs: {num_available}.")
        
        # --- Validate and Adjust Slice Range ---
        if start_image_index >= num_available:
            print(f"Error: Start index ({start_image_index}) is out of bounds. Available images: {num_available}.")
            return
        if end_image_index > num_available:
            print(f"Warning: End index ({end_image_index}) is out of bounds. Adjusting to {num_available}.")
            end_image_index = num_available

        print(f"Processing image slice from index {start_image_index} to {end_image_index - 1}.")
        
        all_image_vectors_1 = []
        all_image_vectors_2 = []
        
        # --- Loop, Load, and Flatten Images into Vectors ---
        for i in range(start_image_index, end_image_index):
            if (i - start_image_index + 1) % 10 == 0: # Print progress update every 10 images
                print(f"  Processing image {i + 1}/{end_image_index}...")
                
            # Load images
            img1 = Image.open(paths1[i]).convert('RGB')
            img2 = Image.open(paths2[i]).convert('RGB')
            
            # Resize
            img1_resized = img1.resize(image_process_size)
            img2_resized = img2.resize(image_process_size)
            
            # Convert to NumPy arrays, flatten, and normalise
            # Each image becomes one very long vector
            img_vector_1 = np.array(img1_resized).ravel() / 255.0
            img_vector_2 = np.array(img2_resized).ravel() / 255.0
            
            all_image_vectors_1.append(img_vector_1)
            all_image_vectors_2.append(img_vector_2)
            
        # --- Prepare Final Data Arrays ---
        final_image_vectors_1 = np.vstack(all_image_vectors_1)
        final_image_vectors_2 = np.vstack(all_image_vectors_2)
        
        # --- Run Analysis and Plotting ---
        run_tsne_and_plot_images(final_image_vectors_1, final_image_vectors_2, perplexity, output_path)

    except FileNotFoundError as e:
        print(f"Error: A folder or image was not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()