import os
import csv

# Define the directory path
label_directory = 'data_split_v1/train'
output_file = 'data/train_v1.csv'

# List all the files in the directory
files = [f for f in os.listdir(label_directory) if f.endswith('.txt')]

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # Write a header if necessary
    writer.writerow(["filename","class", "x", "y", "width", "height"])

    for file in files:
        base_filename = os.path.splitext(file)[0]

        with open(os.path.join(label_directory, file), 'r') as txt_file:
            for line in txt_file:
                # Splitting based on whitespace for simplicity.
                annotation_data = line.strip().split()
                # Writing filename as the first column and then the annotations
                writer.writerow([base_filename] + annotation_data)

print(f'Annotations written to {output_file}')
