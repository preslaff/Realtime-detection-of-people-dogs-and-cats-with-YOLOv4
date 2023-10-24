import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import numpy as np

def compute_iou(ground_truth, prediction):
    """
    Calculate the Intersection-over-Union (IoU) between ground truth and predicted data.
    
    Parameters:
    - ground_truth: DataFrame row containing ground truth bounding box data.
    - prediction: DataFrame row containing predicted bounding box data.

    Returns:
    - IoU value as a float.
    """
    # Extract coordinates for the ground truth and prediction
    gt_box = [ground_truth['x'], ground_truth['y'], ground_truth['x'] + ground_truth['width'], ground_truth['y'] + ground_truth['height']]
    pred_box = [prediction['x'], prediction['y'], prediction['x'] + prediction['width'], prediction['y'] + prediction['height']]

    # Compute the area of intersection
    x_left = max(gt_box[0], pred_box[0])
    y_top = max(gt_box[1], pred_box[1])
    x_right = min(gt_box[2], pred_box[2])
    y_bottom = min(gt_box[3], pred_box[3])
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both boxes
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

    # Compute union
    union = gt_area + pred_area - intersection

    # Avoid division by zero
    iou_value = intersection / union if union != 0 else 0
    return float(iou_value)
    

def evaluate_model_performance(ground_truth_dataset, predicted_dataset, threshold):
    """
    Evaluate the performance of an object detection model using Intersection-over-Union (IoU) as the metric.

    This function performs the following analyses:
    1. Calculates the IoU scores between corresponding ground truth and predicted bounding boxes.
    2. Plots a scatter plot of the IoU scores.
    3. Plots a histogram of the IoU scores.
    4. Computes and prints the accuracy of the predictions using a given IoU threshold.
    5. Plots a precision-recall curve based on the IoU scores.
    6. Prints descriptive statistics of the IoU scores.
    7. Plots a confusion matrix of the predictions based on the IoU threshold.
    8. Plots a learning curve of the accuracy over different training set sizes.

    Parameters
    ----------
    ground_truth_dataset : pandas.DataFrame
        DataFrame containing ground truth bounding box data. Each row should represent a bounding box with columns:
        'class', 'x', 'y', 'width', 'height'.
    predicted_dataset : pandas.DataFrame
        DataFrame containing predicted bounding box data. Each row should represent a bounding box with columns:
        'class', 'x', 'y', 'width', 'height'.
    threshold : float
        The IoU threshold used to determine a correct detection. Predicted bounding boxes with IoU scores above this
        threshold are considered correct.

    Returns
    -------
    None

    Note
    ----
    This function uses matplotlib and seaborn for plotting. It also uses scikit-learn for the precision-recall curve,
    average precision score, and confusion matrix computations.
    """
    ious = []

    # Iterate over unique filenames in ground_truth_v1
    for filename in ground_truth_dataset['filename'].unique():

        # Filter rows based on filename and image_name
        ground_truth_subset = ground_truth_dataset[ground_truth_dataset['filename'] == filename]
        predictions_subset = predicted_dataset[predicted_dataset['image_name'] == filename]

        # Get the minimum length of the two subsets
        min_length = min(len(ground_truth_subset), len(predictions_subset))

        # Calculate IoU for each corresponding row in the filtered subsets
        for i in range(min_length):
            iou_value = compute_iou(ground_truth_subset.iloc[i], predictions_subset.iloc[i])
            if not isinstance(iou_value, float):
                print(f"Non-float IoU value found for filename {filename} at index {i}: {iou_value}")
            else:
                ious.append(iou_value)

    # Convert the list of IoU values to a pandas Series
    iou_series = pd.Series(ious)
            
    # Scatter plot of IoU values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ious)), ious, alpha=0.6)
    plt.title('Scatter plot of IoU scores')
    plt.xlabel('Sample index')
    plt.ylabel('IoU score')
    plt.grid(True)
    plt.show()

    # Plotting a histogram of IoU values with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(iou_series, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.title('Distribution of IoU scores')
    plt.xlabel('IoU score')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.show()

    # Compute accuracy
    accuracy = (iou_series > threshold).mean()
    print(f"Accuracy (IoU > {threshold}): {accuracy*100:.2f}%")

    print(pd.Series(ious).describe())
  


    # Confusion matrix
    cm = confusion_matrix(iou_series > threshold, iou_series > threshold)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
