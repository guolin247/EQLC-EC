import os
import pandas as pd
import numpy as np
import logging

def extract_and_save_data(input_file, output_folder, mz_pairs):
    try:
        # Load CSV data, header=None to ensure the first row is not used as column names
        data = pd.read_csv(input_file, header=None, low_memory=False)
        logging.info(f"Successfully loaded file: {input_file}")
    except Exception as e:
        logging.error(f"Error loading file {input_file}: {e}")
        return

    # Ensure the mz values are converted to numeric
    data.iloc[2:, 0] = pd.to_numeric(data.iloc[2:, 0], errors='coerce')

    # Extract file names and remove ".RAW", then add ".npy" as file names
    file_names = data.iloc[0, 1:].str.replace(".RAW", "", regex=False) + ".npy"

    # Extract category labels
    label_names = data.iloc[1, 1:]

    # Create label-to-integer mapping
    unique_labels = label_names.unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = label_names.map(label_to_int)

    # Log mapping relationship
    logging.info(f"Label to integer mapping: {label_to_int}")

    # Extract intensity values from the second row onwards, and remove the first column
    intensity_values = data.iloc[2:, 1:].astype(float)

    all_data_files = []
    all_labels = []
    all_sample_ids = []

    # Define mz pairs
    mz_pairs_list = [
        (151.0478, 185.0420), (151.0478, 210.0760), (151.0478, 230.9900), (151.0478, 257.0740), (151.0478, 311.1160),
        (159.0628, 230.9900), (159.0628, 237.0748), (159.0628, 249.0380), (159.0628, 257.0740), (159.0628, 287.1370),
        (160.0756, 230.9900), (160.0756, 237.0748), (160.0756, 249.0370), (160.0756, 251.1622), (160.0756, 406.1320),
        (189.1597, 210.0760), (189.1597, 237.0748), (189.1602, 311.1160), (192.0767, 287.1370), (210.0760, 257.0740),
        (230.9900, 237.0748), (230.9900, 406.1320), (230.9910, 251.1622), (237.0748, 311.1160), (249.0370, 311.1160),
        (249.0380, 257.0740), (251.1622, 311.1160), (287.1370, 406.1320)
    ]

    for i, file_name in enumerate(file_names):
        intensity = intensity_values.iloc[:, i]

        # Extract specified mz pairs and sum their values
        extracted_values = []
        for mz1, mz2 in mz_pairs_list:
            mz1_idx = data.iloc[2:, 0].sub(mz1).abs().idxmin()  # Find the row index closest to the specified mz1 value
            mz2_idx = data.iloc[2:, 0].sub(mz2).abs().idxmin()  # Find the row index closest to the specified mz2 value
            combined_value = intensity_values.iloc[mz1_idx, i] + intensity_values.iloc[mz2_idx, i]
            extracted_values.append(combined_value)
        extracted_values = np.array(extracted_values)

        # Save the combined intensity values corresponding to the specified mz pairs
        npy_file_path = os.path.join(output_folder, f"{file_name}")
        np.save(npy_file_path, extracted_values)
        logging.info(f"Extracted intensity data saved to {npy_file_path}")

        all_data_files.append(file_name)
        all_labels.append(labels.iat[i])

        # Parse sample_id, assuming it is delimited by underscores in the file name
        sample_id = "_".join(file_name.split('_')[-3:-1])  # Take the second and third last parts as sample_id
        all_sample_ids.append(sample_id)

    # Save file names, labels, and sample_ids to labels.npy
    labels_output = np.array(list(zip(all_data_files, all_labels, all_sample_ids)), dtype=object)
    labels_path = os.path.join(output_folder, 'labels.npy')
    np.save(labels_path, labels_output)
    logging.info(f"File names, labels, and sample_ids saved to {labels_path}")

    # Save label mapping relationship to a text file
    mapping_file_path = os.path.join(output_folder, 'label_mapping.txt')
    with open(mapping_file_path, 'w') as f:
        for label, idx in label_to_int.items():
            f.write(f"{label}: {idx}\n")
    logging.info(f"Label mapping saved to {mapping_file_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_file = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_tomato\raw\raw.csv"  # CSV data file
    output_folder = "processed_data"  # Output directory for saving .npy files

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process the dataset
    extract_and_save_data(input_file, output_folder, mz_pairs=None)