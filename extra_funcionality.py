import json
import os

# Define the file path
FILE_PATH = 'jupyter_notebooks/intern_notebook_information_share.json'

def load_data(name):
    """
    Load data from the JSON file based on the provided name.

    Parameters:
        name (str): The name of the list or dictionary to load.

    Returns:
        object: The data loaded from the file (list, dictionary, etc.).
    """
    if not os.path.exists(FILE_PATH):
        return None

    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    return data.get(name)

def save_data(name, data):
    """
    Save data to the JSON file based on the provided name.

    Parameters:
        name (str): The name of the list or dictionary.
        data (object): The data to save to the file.
    """
    # Load existing data from file
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'r') as f:
            data_dict = json.load(f)
    else:
        # If the file doesn't exist, create a new dictionary to store data
        data_dict = {}

    # Update or add the new data
    data_dict[name] = data

    # Write data to file
    with open(FILE_PATH, 'w') as f:
        json.dump(data_dict, f)
