
### Installation

1. **Download the Data**:
   - Download the archived PhysioNet database from Google Drive:
     [Download PhysioNet Database](https://drive.google.com/file/d/17cN3mV-858kjnptVi5Pwh7LKw1n1JG7Y/view?usp=drive_link)

2. **Extract the Data**:
   - Extract the archive into this specific folder:
     ```
     raw_ecg
     ```
3. **Install Required Packages**:
   - Install all the necessary Python packages from the `requirements.txt` file:
     ```
     pip install -r requirements.txt
     ```
### Preprocessing

To generate heartbeat samples from the continuous ECG signal, follow these steps:

4. **Run the Preprocessing Script**:
   - Navigate to the preprocessing directory and run the `preprocessing.py` script with the necessary parameters:
   ### Command Line Arguments

- `--test`
  - **Type**: Int
  - **Description**: Sets the mode of the script. If `0`, the script operates in test mode.
  - **Default Value**: `0`
  - **Example**: `--test 0`

- `--raw_ecg_dir`
  - **Type**: String
  - **Description**: Specifies the directory where the raw ECG data files are stored.
  - **Default Value**: `"../raw_ecg/physionet_ecg_data_full/git_ecg_data_full"`
  - **Example**: `--raw_ecg_dir "path/to/your/ecg_data"`

- `--sample_ecg_dir`
  - **Type**: String
  - **Description**: Specifies the directory where processed ECG samples are saved.
  - **Default Value**: `"ecg_samples"`
  - **Example**: `--sample_ecg_dir "path/to/your/ecg_samples"`

- `--sample_ecg_test_dir`
  - **Type**: String
  - **Description**: Specifies the directory where ECG test samples are saved.
  - **Default Value**: `"ecg_test_samples"`
  - **Example**: `--sample_ecg_test_dir "path/to/your/ecg_test_samples"`  
 **Usage**
     ```bash
     cd preprocessing
     ```
    
     ```bash
      python preprocessing.py --test 0 --raw_ecg_dir "../raw_ecg/physionet_ecg_data_full/git_ecg_data_full" --sample_ecg_dir "ecg_samples" --sample_ecg_test_dir "ecg_test_samples"
     ```
     
     ```bash
     python preprocessing.py --test 1 --raw_ecg_dir "../raw_ecg/physionet_ecg_data_full/git_ecg_data_full" --sample_ecg_dir "ecg_samples" --sample_ecg_test_dir "ecg_test_samples"
     ```

5. **Run the Training Script**:
    #### Parameters:
   - `--num_train_subjects`: The number of training subjects. Specifies how many subjects' data to include in the training process.
   - `--m`: The margin parameter for the loss function. Adjusts the strictness of the margin in the loss calculation.
   - `--data_folder`: Path to the folder containing the ECG data samples.
    
    Make sure that the specified `data_folder` contains the preprocessed ECG data ready for training. 
    After training model will be saved into the path:  
`training/models/models_{num_train_subjects}/ver2_m_{str(m)[0]+str(m)[2]}_{num_train_subjects}.pth`

    To train the model, execute the following command:  
Command line parameters values are just the example, you can put yours `num_train_subjects` and `m`
    
    ```bash
    cd training
    ```
     
    ```bash
    python model_training.py --num_train_subjects 100 --m 0.5 --data_folder "../preprocessing/ecg_samples"
    ```






6.**Run the Matcher Testing Script**:
   - Navigate to the testing directory and run the `test_matcher.py` script with the necessary parameters:
   ### Command Line Arguments

- `--num_train_subjects`
  - **Type**: int
  - **Description**: number of train parameters. needed to load a model with such a  number of training users
  - **Default Value**: `100`
  - **Example**: `--num_train_subjects 100`

- `--num_test_subjects `
  - **Type**: int
  - **Description**: number of test parameters. needed to load a model with such a  number of training users
  - **Default Value**: `50`
  - **Example**: `--num_test_subjects 50`
- `--m `
  - **Type**: float
  - **Description**: margin parameter. needed to load a model with such a  number of training users
  - **Default Value**: `0.1`
  - **Example**: `--m 0.3`
- `--data_folder`
  - **Type**: String
  - **Description**: Specifies the directory where the ECG samples are stored.
  - **Default Value**: `../preprocessing/ecg_samples`
  - **Example**: `--data_folder "path/to/your/ecg_data"`
 **Usage**
Command line parameters values are just the example.  
  You can put yours `num_train_subjects`, `num_test_subjects` and `m`, but ensure, that model with such parameters exists.
     ```bash
    cd testing  
    ```
     ```bash
    python test_matcher.py --num_train_subjects 100 --num_test_subjects 50 --m 0.5 --data_folder "../preprocessing/ecg_samples"
     ```
    
    test_mather script will generate log file in `testing/logs` directory with name:  `log_{num_train_subjects}_{num_test_subjects}_{str(m)[0]+str(m)[2]}`
  - If log file exist, the new logging will be added to the file,  
  so the newest results for this model are presented as the last part of the log file.  
  - Different results in the log file are separated by `########################################################`
  

**Note!**: Obtained results might be slightly different from the ones stated in PDF, because of random part in the testing script







