import pandas as pd
import os

from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH, CSV_FILE


def import_local_dataset():
    """
    return the raw dataset from local disk or cloud storage
    """
    path = os.path.join(LOCAL_DATA_PATH,
                        CSV_FILE)
    try:
        df = pd.read_csv(path)  # read all rows
    except pd.errors.EmptyDataError:
        return None  # end of data

    return df


def get_local_model(file_model):
    state_dict = os.path.join(LOCAL_DATA_PATH, file_model)

    return state_dict
