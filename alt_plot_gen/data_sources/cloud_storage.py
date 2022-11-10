#from google.cloud import bigquery

import pandas as pd
import os

from alt_plot_gen.ml_logic.params import PROJECT, GOOGLE_APPLICATION_CREDENTIALS, BUCKET_NAME, CSV_FILE
from smart_open import open as smart_open
import io
from google.cloud import storage
import torch
from transformers import GPT2LMHeadModel


def import_cloud_dataset() -> pd.DataFrame:
    """
    return a big query dataset table
    format the output dataframe according to the provided data types
    """

    '''
    use this code to upload dataset if you have it on a bucket
    '''
    bucket_path = os.path.join('gs://',
                                BUCKET_NAME,
                                CSV_FILE)
    big_query_df = pd.read_csv(bucket_path)  # read all rows

    '''
    use this code to upload dataset if you have it on tables (connection with bq client)
    '''
    #table = f"{PROJECT}.{DATASET}.{table}"

    #client = bigquery.Client()

    #rows = client.list_rows(table)

    # convert to expected data types
    #big_query_df = rows.to_dataframe()

    #if big_query_df.shape[0] == 0:
    #    return None  # end of data


    return big_query_df


def get_cloud_model(file_model):
    client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS, project=PROJECT)
    model_path = os.path.join('gs://',
                                BUCKET_NAME,
                                file_model)
    with smart_open(model_path, 'rb', transport_params=dict(client=client)) as f:
        state_dict = io.BytesIO(f.read())
    '''
    try:
        model.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')))
        print("\n✅ model loaded")
    except:
        print(f"\n❌ model no loaded")
        return None
    '''
    return state_dict
