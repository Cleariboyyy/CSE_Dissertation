import os
import shutil

def copy_csv(source_dir, target_dir):
    # Create target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Walk through the directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check for CSV files
            if file.endswith('.csv'):
                # Construct full file path
                full_file_path = os.path.join(root, file)
                # Construct target file path
                target_file_path = os.path.join(target_dir, file)
                # Copy file to target directory
                shutil.copy(full_file_path, target_file_path)
                print(f'Copied {full_file_path} to {target_file_path}')

# Example usage
source_directory = '2EVTX-to-MITRE-Attack-master/TA0040-Impact'
target_directory = 'CSVevents/TA0040-Impact'
copy_csv(source_directory, target_directory)