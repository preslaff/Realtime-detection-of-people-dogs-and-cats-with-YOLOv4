import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float, io
import cv2

def random_grid(path_to_images):
    """ 
    Draws a grid of sample images located in the path_to_images directory. 
    Takes a maximum of 12 images. Images have to be .jpg or .png format.

    Parameters:
    path_to_images -> str
    """
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(path_to_images):
        for filename in [f for f in filenames if f.endswith(".jpg") or f.endswith(".png")]:
            image_paths.append(os.path.join(dirpath, filename))
        
    # Randomly select up to 12 images from the dataset
    num_images = min(len(image_paths), 12)
    selected_paths = random.sample(image_paths, num_images)
        
    # Load images using OpenCV
    images = [cv2.imread(path) for path in selected_paths]
    # Convert BGR images (OpenCV default) to RGB for visualization
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
       
    # Create a grid of images
    rows = int(num_images / 4) if num_images % 4 == 0 else int(num_images / 4) + 1
    fig, axs = plt.subplots(rows, 4, figsize=(12, 9))
    
    # If there's only one row, axs is a 1D array
    axs = axs.reshape(-1)
        
    for i, ax in enumerate(axs):
        if i < num_images:
            ax.imshow(images_rgb[i])
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Hide unused subplots
        
    plt.tight_layout()
    plt.show()

def plot_img_and_hist(image, axes, bins=256):
    """
    Plot an image along with its histogram and cumulative distribution on one plot.

    """
    matplotlib.rcParams['font.size'] = 8
    
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', density = True)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def plot_aspect_ratios_and_areas(all_frames, all_frames2=None, dataset1="All frames", dataset2=None):
    """Plots histograms of aspect ratios and areas for given frames. Optionally compares with a second set."""
    
    # Calculate aspect ratios and areas for the first dataset
    aspect_ratios = (all_frames['width'] / all_frames['height']).tolist()
    box_areas = (all_frames['width'] * all_frames['height']).tolist()
    
    # Convert to NumPy arrays
    aspect_ratios = np.array(aspect_ratios)
    box_areas = np.array(box_areas)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(aspect_ratios, bins=20, color='blue', alpha=0.7, label=dataset1)
    
    # If the second dataset is provided, calculate and plot its aspect ratios
    if all_frames2 is not None:
        aspect_ratios2 = (all_frames2['width'] / all_frames2['height']).tolist()
        aspect_ratios2 = np.array(aspect_ratios2)
        plt.hist(aspect_ratios2, bins=20, color='red', alpha=0.5, label=dataset2)
    
    plt.title('Aspect Ratios')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(box_areas, bins=20, color='green', alpha=0.7, label=dataset1)
    
    # If the second dataset is provided, calculate and plot its box areas
    if all_frames2 is not None:
        box_areas2 = (all_frames2['width'] * all_frames2['height']).tolist()
        box_areas2 = np.array(box_areas2)
        plt.hist(box_areas2, bins=20, color='orange', alpha=0.5, label=dataset2)
    
    plt.title('Box Areas')
    plt.xlabel('Box Area')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_label_centers(all_frames, all_frames2=None, dataset1 = "All frames", dataset2 = None):
    """Plots the centers of labels from given frames. Optionally compares with a second set of frames."""
    
    # Calculate centers for the first dataset
    center_x1 = all_frames['x'] + all_frames['width'] / 2
    center_y1 = all_frames['y'] + all_frames['height'] / 2
    centers1 = list(zip(center_x1, center_y1))
    
    plt.figure(figsize=(12, 10))
    for x, y in centers1:
        # plt.scatter(x, y, alpha=0.6, cmap='viridis')  # use a color map
        plt.scatter(x, y, alpha=0.6, color = "blue")
    # for x, y in centers1:
    #     plt.scatter(x, y, alpha=0.6, color='blue', label='Dataset 1')
    
    # If the second dataset is provided, calculate and plot its centers
    if all_frames2 is not None:
        center_x2 = all_frames2['x'] + all_frames2['width'] / 2
        center_y2 = all_frames2['y'] + all_frames2['height'] / 2
        centers2 = list(zip(center_x2, center_y2))
        for x, y in centers2:
            plt.scatter(x, y, alpha=0.6, color='red', label=dataset2)
    
    plt.title("Centers of All Labels")
    plt.xlabel("Normalized X Coordinate")
    plt.ylabel("Normalized Y Coordinate")
    plt.xlim(0, 1)  # because it's normalized
    plt.ylim(0, 1)  # because it's normalized
    plt.gca().invert_yaxis()  # to match image coordinates

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=dataset1)]
    if all_frames2 is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=dataset2))
    plt.legend(handles=legend_elements, loc='upper right')

    plt.show()