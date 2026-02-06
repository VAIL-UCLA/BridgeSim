import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne_pixel_distribution(image_path_1, image_path_2, sample_size=(100, 100), perplexity=30):
    """
    Loads two images, performs t-SNE on their pixel data, and plots the result
    to visualise and compare their colour distributions.

    Args:
        image_path_1 (str): File path to the first image.
        image_path_2 (str): File path to the second image.
        sample_size (tuple): The (width, height) to resize images to for sampling.
                             Smaller sizes run faster.
        perplexity (int): The perplexity value for the t-SNE algorithm.
    """
    try:
        # --- 1. Load and Pre-process Images ---
        print("Loading and pre-processing images...")
        img1 = Image.open(image_path_1).convert('RGB')
        img2 = Image.open(image_path_2).convert('RGB')

        # Resize images to a standard size for consistent sampling and performance
        img1_resized = img1.resize(sample_size)
        img2_resized = img2.resize(sample_size)

        # Convert images to NumPy arrays
        pixels1 = np.array(img1_resized)
        pixels2 = np.array(img2_resized)

        # --- 2. Prepare Data for t-SNE ---
        # Reshape the 3D image arrays (width, height, channels) into 2D arrays (num_pixels, channels)
        # Normalise pixel values to be between 0 and 1
        pixels1_flat = pixels1.reshape(-1, 3) / 255.0
        pixels2_flat = pixels2.reshape(-1, 3) / 255.0

        # Combine the pixel data from both images into a single dataset
        combined_pixels = np.vstack([pixels1_flat, pixels2_flat])

        # Create labels to identify which image each pixel came from
        # 0 for image 1, 1 for image 2
        num_pixels_per_image = pixels1_flat.shape[0]
        labels = np.array([0] * num_pixels_per_image + [1] * num_pixels_per_image)

        print(f"Total number of pixels being processed: {combined_pixels.shape[0]}")

        # --- 3. Run t-SNE ---
        print("Running t-SNE... This may take a few moments.")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            learning_rate='auto',
            init='pca',
            verbose=1 # Provides progress updates
        )
        tsne_results = tsne.fit_transform(combined_pixels)

        # --- 4. Visualise the Results ---
        print("Creating plot...")
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Separate the results for each image
        results_img1 = tsne_results[labels == 0]
        results_img2 = tsne_results[labels == 1]

        # *** MODIFICATION START ***
        # Create scatter plots with distinct, uniform colours for each image
        ax.scatter(
            results_img1[:, 0], results_img1[:, 1],
            c='royalblue',        # All points for Image 1 are blue
            label='MetaDrive',
            alpha=0.7,
            s=15,                 # Slightly larger markers
            marker='o'            # Use circle markers for Image 1
        )
        ax.scatter(
            results_img2[:, 0], results_img2[:, 1],
            c='darkorange',      # All points for Image 2 are orange
            label='Bench2Drive',
            alpha=0.7,
            s=15,
            marker='X'            # Use 'X' markers for Image 2
        )

        ax.set_title(f't-SNE Visualisation of Pixel Colour Distributions (Perplexity: {perplexity})', fontsize=16)
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)

        # The legend can now be created automatically
        ax.legend(title="Image Source", loc='best', markerscale=2)
        # *** MODIFICATION END ***
        
        #plt.show()
        plt.savefig('tools/tsne_pixel_distribution.png')

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the image files exist at the specified paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # Specify the paths to your two images
    # MetaDrive image
    image_path_1 = '/home/abhijit/Work/BridgeSim/metadrive_evaluation/metadrive_uniad_output/raw_camera_images/CAM_FRONT/frame_0137.jpg'
    # Bench2Drive image
    image_path_2 = '/home/abhijit/Work/BridgeSim/Bench2Drive/Bench2Drive-mini/HardBreakRoute_Town01_Route30_Weather3/camera/rgb_front/00137.jpg'

    # Run the function
    # You can adjust perplexity. Typical values are between 5 and 50.
    plot_tsne_pixel_distribution(image_path_1, image_path_2, perplexity=40)