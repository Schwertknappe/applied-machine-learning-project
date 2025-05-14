import numpy as np
import pandas as pd  
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If the data is empty after removing unlabeled data and dropping NaN values.
    """
    # check if file exists under specified data path
    if not data_path.is_file():
        raise FileNotFoundError('Data file not found.')
    
    # read file into data frame
    data = pd.read_csv(data_path)

    # remove unlabeled rows
    data = remove_unlabeled_data(data)

    if data.empty:
        raise ValueError('No labeled data found in specified data file.')
    
    # remove rows with missing values
    data = data.dropna()

    return data


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """

    # ensure given dataset has a labels column
    if any(col not in data.columns for col in ['labels', 'exp_ids']):
        raise KeyError("Dataset does not include at least one required column.")

    # remove rows with unlabeled data, i.e. where the labels value is -1
    data.drop(data[data.labels == -1].index, inplace=True)

    return data


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'labels', 'exp_ids', and feature columns.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): Array of labels
            - exp_ids (np.ndarray): Array of experiment IDs
            - data (np.ndarray): Combined array of current and voltage features
    """

    # extract labels and exp_ids into arrays and drop them from the dataset
    features = data.drop(columns=["labels", "exp_ids"])
    labels = data['labels'].to_numpy()
    exp_ids = data['exp_ids'].to_numpy()

    # extract current and voltage data into separate arrays
    cols_v = data.columns[data.columns.str.startswith("V")]
    cols_i = data.columns[data.columns.str.startswith("I")]
    
    current_data = data[cols_i].to_numpy()
    voltage_data = data[cols_v].to_numpy()

    # stack current and voltage data into two related values per timestamp
    features = np.stack([current_data, voltage_data], axis=2)

    data = (labels, exp_ids, features)

    # check if any array from the tuple are empty
    if any(arr.size == 0 for arr in data):
        raise ValueError("Data does not contain any values.")

    # check if any array from the tuple has a non-numeric datatype
    if not all(np.issubdtype(arr.dtype, np.number) for arr in data):
        raise ValueError("At least one column contains non-numeric data.")

    return data


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    
    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window
    
    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """

    # Apply sliding window over axis 0 (samples) and keep the timesteps and features intact
    windows = np.lib.stride_tricks.sliding_window_view(data, sequence_length, axis=0)

    # Reshape windows from (n_windows, timesteps, features, sequence_length) to 
    #    (n_windows, sequence_length * timesteps, features)

    n_windows = windows.shape[0]
    timesteps = data.shape[1]
    features = data.shape[2]

    # set axes to the correct position to prepare reshape from 4D to 3D
    windowed_data = windows.transpose(0, 3, 1, 2)
    windowed_data = windowed_data.reshape(n_windows, timesteps * sequence_length, features)

    return windowed_data


def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """

    # check if numpy cache files exist
    missing_npy_files = [file for file in ['labels.npy', 'exp_ids.npy', 'data.npy'] if not (path.parent / file).exists()]

    data = ()
    if missing_npy_files:
        # there is at least one cache file missing, therefore read from csv file and create cache files
        data = load_data(path)
        data = convert_to_np(data)

        labels, exp_ids, features = data
        np.save(path.parent/"data.npy", features)
        np.save(path.parent/"labels.npy", labels)
        np.save(path.parent/"exp_ids.npy", exp_ids)
        data = (features, labels, exp_ids)

    else:
        # cache files already exist, therefore read from cache
        features = np.load(path.parent/"data.npy")
        labels = np.load(path.parent/"labels.npy")
        exp_ids = np.load(path.parent/"exp_ids.npy")

        data = (features, labels, exp_ids)

    # get random samples from data if requested
    if n_samples:
        # generate random indeces
        indices = np.random.choice(len(labels), size = n_samples, replace=False)
        # apply random indices to arrays and update the tuple
        data = (features[indices], labels[indices], exp_ids[indices])
        features, labels, exp_ids = data

    # create sliding window sequences if requested
    if return_sequences:
        if len(labels) < sequence_length:
            raise ValueError("Not enough samples to form a single window")

        features = create_sliding_windows_first_dim(features, sequence_length)

        # Slice labels and exp_ids to match the window size: take the last label in each window
        labels = np.lib.stride_tricks.sliding_window_view(labels, sequence_length, axis=0)
        exp_ids = np.lib.stride_tricks.sliding_window_view(exp_ids, sequence_length, axis=0)

        data = (features, labels, exp_ids)

    return data


def test():
    get_welding_data(Path("data/test_data.csv"), None, True, 3)

#test()