
### Installation

1. **Download the Data**:
   - Download the archived PhysioNet database from Google Drive:
     [Download PhysioNet Database](https://drive.google.com/file/d/17cN3mV-858kjnptVi5Pwh7LKw1n1JG7Y/view?usp=drive_link)

2. **Extract the Data**:
   - Extract the archive into this specific folder:
     ```
     raw_ecg_folder_name
     ```

### Preprocessing

To generate heartbeat samples from the continuous ECG signal, follow these steps:

3. **Run the Preprocessing Script**:
   - Navigate to the preprocessing directory and run the `preprocessing.py` script with the necessary parameters:

     ```bash
     python preprocessing/preprocessing.py --test True --raw_ecg_dir "../raw_ecg_folder_name/git_ecg_data_full" --sample_ecg_dir "ecg_samples" --sample_ecg_test_dir "ecg_test_samples"
     ```
     
     ```bash
     python preprocessing/preprocessing.py --test False --raw_ecg_dir "../raw_ecg_folder_name/git_ecg_data_full" --sample_ecg_dir "ecg_samples" --sample_ecg_test_dir "ecg_test_samples"
     ```

   - This command sets up the script to process data assuming a test environment, specifying directories for the raw data and where to save the processed samples.

## Usage

