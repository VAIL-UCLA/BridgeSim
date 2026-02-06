import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import glob

def plot_similarity_scores(scores, start_index, output_path):
    """
    Plots the frame-by-frame cosine similarity scores.
    """
    if not scores:
        print("No scores to plot.")
        return

    mean_score = np.mean(scores)
    frame_numbers = range(start_index, start_index + len(scores))
    
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(frame_numbers, scores, label='Per-Frame Similarity', marker='.', linestyle='-', markersize=4)
    ax.axhline(mean_score, color='r', linestyle='--', label=f'Mean Similarity: {mean_score:.4f}')
    
    ax.set_title('Image-Vector Cosine Similarity Over Sequence', fontsize=16)
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Cosine Similarity Score', fontsize=12)
    ax.set_ylim(0, 1.05) # Cosine similarity for non-negative vectors is between 0 and 1
    ax.legend()
    ax.grid(True)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path)
    print(f"\nSimilarity plot saved to: {output_path}")

def main():
    """
    Main function to perform paired image-vector comparison using cosine similarity.
    """
    # --- Configuration ---
    folder_path_1 =  '/home/abhijit/Work/BridgeSim/metadrive_evaluation/metadrive_uniad_output/raw_camera_images/CAM_FRONT/'
    folder_path_2 = '/home/abhijit/Work/BridgeSim/Bench2Drive/Bench2Drive-mini/HardBreakRoute_Town01_Route30_Weather3/camera/rgb_front/'
    
    start_image_index = 0
    end_image_index = 270 # Up to (but not including) this index
    
    output_path = 'tools/cosine_similarity_plot.png'
    image_process_size = (100, 100) # Must be consistent for all images

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
        
        similarity_scores = []
        
        # --- Loop, Flatten, and Compare Image Pairs ---
        for i in range(start_image_index, end_image_index):
            if (i - start_image_index + 1) % 20 == 0:
                print(f"  Comparing image pair {i}...")
                
            # Load images
            img1 = Image.open(paths1[i]).convert('RGB')
            img2 = Image.open(paths2[i]).convert('RGB')
            
            # Resize
            img1_resized = img1.resize(image_process_size)
            img2_resized = img2.resize(image_process_size)
            
            # Flatten into vectors and normalise
            img_vector_1 = np.array(img1_resized).ravel() / 255.0
            img_vector_2 = np.array(img2_resized).ravel() / 255.0
            
            # Reshape for scikit-learn's cosine_similarity function
            img_vector_1 = img_vector_1.reshape(1, -1)
            img_vector_2 = img_vector_2.reshape(1, -1)
            
            # Calculate similarity and append score
            score = cosine_similarity(img_vector_1, img_vector_2)[0][0]
            similarity_scores.append(score)

        # --- Report Results ---
        if similarity_scores:
            scores_array = np.array(similarity_scores)
            print("\n--- Cosine Similarity Results ---")
            print(f"Mean Similarity:   {np.mean(scores_array):.4f}")
            print(f"Std Deviation:     {np.std(scores_array):.4f}")
            print(f"Minimum Similarity:  {np.min(scores_array):.4f} (at frame {start_image_index + np.argmin(scores_array)})")
            print(f"Maximum Similarity:  {np.max(scores_array):.4f} (at frame {start_image_index + np.argmax(scores_array)})")
            print("-----------------------------------")
            
            # --- Visualise the Scores ---
            plot_similarity_scores(similarity_scores, start_image_index, output_path)
        else:
            print("\nNo images were processed to generate similarity scores.")

    except FileNotFoundError as e:
        print(f"Error: A folder or image was not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()