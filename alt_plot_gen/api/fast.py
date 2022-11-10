from datetime import datetime
from sys import modules

import pandas as pd
import os
import torch
import json

from alt_plot_gen.ml_logic.data import clean_data, clean_plot
#from alt_plot_gen.ml_logic.registry import load_model
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH, CSV_TEST_FILE
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset
#import torch.nn.functional as F
from alt_plot_gen.ml_logic.generation import generate

from alt_plot_gen.interface.main import text_generation

from smart_open import open as smart_open
import io
from google.cloud import storage

from alt_plot_gen.ml_logic.model import load_model

#--------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#define a new clean_data function
def clean_data_2(dataseries):

    # Cut plot wuth more than 1024 tokens to adapt to gpt-2 medium limitations
    dataseries['Plot'] = dataseries['Plot'].map(lambda x: " ".join(x.split()[:350]))  #cut all plots until the 350th word
    dataseries['Plot'] = dataseries['Plot'].str.split().str[:-50].apply(' '.join)
    return dataseries

#define a new preprocess function
def preprocess_2(dataseries):
    print("\n⭐️ use case: preprocess")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    ds_cleaned = clean_data(dataseries)
    plot = ds_cleaned['Plot'].values[0]
    plot_tokenized = tokenizer.encode(plot)

    print(f"\n✅ data tokenized")

    return plot_tokenized

#----------------------
# Endpoints
#----------------------

# Load the movie catalog and return it to the caller
@app.get("/index")
def index():
    dataset = pd.read_csv(os.path.join(LOCAL_DATA_PATH, CSV_TEST_FILE), header=0, usecols=['Title', 'Release Year', 'Genre', 'Plot', 'True_end_plot'])
    plot_catalog = json.loads(dataset.to_json(orient='records'))
    app.state.dataset = dataset
    return plot_catalog


def select_model_by_genre(genre):
    if not genre:
        file = "trained_model.pt"
    else:
        file = f"trained_{genre}.pt"
    return file

# Run the generation model and return the generated text to the caller
@app.get("/generate_new_end")
def generate_new_end(title: str, release_year: int, genre=None):
    dataset = app.state.dataset
    locate_plot = dataset.loc[(dataset['Title'] == title) & (dataset['Release Year'] == release_year)]
    selected_plot = locate_plot['Plot'].values[0]
    print(selected_plot)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print("\n⭐️ load model")
    model = load_model(genre)

    #Run the functions to generate the alternative endings
    alternative_end, full_test_generated_plot = text_generation(model, tokenizer, selected_plot)

    return alternative_end
