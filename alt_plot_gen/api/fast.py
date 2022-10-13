from datetime import datetime
from sys import modules
# $WIPE_BEGIN
import pytz
import pandas as pd
import os
import torch
from alt_plot_gen.ml_logic.data import clean_data, clean_plot
#from alt_plot_gen.ml_logic.registry import load_model
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH
from transformers import GPT2Tokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset
import torch.nn.functional as F
from alt_plot_gen.ml_logic.generation import generate


from alt_plot_gen.interface.main import text_generation
# $WIPE_END'''

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# # $WIPE_BEGIN
# # 💡 Preload the model to accelerate the predictions
# # We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# # The trick is to load the model in memory when the uvicorn server starts
# # Then to store the model in an `app.state.model` global variable accessible across all routes!
# # This will prove very useful for demo days

#app.state.model = load_model()
# # $WIPE_END

#define a new clean_data function
def clean_data_2(dataseries):

    # Cut plot wuth more than 1024 tokens to adapt to gpt-2 medium limitations
    dataseries['Plot'] = dataseries['Plot'].map(lambda x: " ".join(x.split()[:350]))  #cut all plots until the 350th word
    dataseries['Plot'] = dataseries['Plot'].str.split().str[:-50].apply(' '.join)
    return dataseries

#define a new tokenizer function
def tokenize_plots_2(dataseries):
    class Token_plot(Dataset):
        def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
            self.plots = []

            for row in dataseries['Plot']:
                self.plots.append(torch.tensor(
                        self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
                    ))
            if truncate:
                self.plots = self.plots[:20000]
            self.plots_count = len(self.plots)

        def __len__(self):
            return self.plots_count

        def __getitem__(self, item):
            return self.plots[item]

    dataset = Token_plot(dataseries['Plot'].values[0], truncate=True, gpt2_type="gpt2")   #list of tensors (tokenized plots)

    return dataset
#define a new preprocess function
def preprocess_2(dataseries):
    print("\n⭐️ use case: preprocess")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    ds_cleaned = clean_data(dataseries)
    plot = ds_cleaned['Plot'].values[0]
    plot_tokenized = tokenizer.encode(plot)

    print(f"\n✅ data tokenized")

    return plot_tokenized

#app.state.model = load_model()

@app.get("/index")
def index():
    pass

def select_model_by_genre(genre):
    if not genre:
        model = "trained_model.pt"
    else:
        model = f"trained_{genre}.pt"
    return os.path.join(os.environ.get("LOCAL_DATA_PATH"), model)

@app.get("/generate_new_end")
def generate_new_end(title: str, release_year: int, genre=None):

    dataset = pd.read_csv(f'{LOCAL_DATA_PATH}/test_set_demo.csv')

    locate_plot = dataset.loc[(dataset['Title'] == title) & (dataset['Release Year'] == release_year)]
    selected_plot = locate_plot['Plot'].values[0]

    #plot_preproc = preprocess_2(selected_plot)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #plot_cleaned = clean_data_2(selected_plot)

    #plot = 'young raj malhotra (akshay kumar) lives with his elder brother, rohit, his sister-in-law, kiran, and his niece. after an accident fractures his leg, he is unable to walk for some time. due to psychological reasons, he cannot walk even after the fracture heals. when the malhotras move to dehra dun, raj befriends their neighbour, young tomboy kajal (lara dutta), as both share a common passion for aeroplanes. kajal encourages raj to walk, and he succeeds. years later the two continue to be fast friends, and everyone expects them to marry soon. raj secretly loves kajal and is waiting for the right time to propose to her. however, kajal sees him only as her best friend. raj is recruited by the indian air force and goes for training for a year and a half. after his training is over, he rushes to kajal to propose to her, only to find out that she is in love with multimillionaire businessman, karan singhania (aman verma). raj tells kajal that karan is the best life partner for her, and does not reveal his true feelings. kajal and karan get married and during the reception party, kajal comes to know of raj feelings for her. hurt by the fact that she did not see his feelings for her despite being raj best friend, kajal asks him to move on with his life. thereafter the malhotras move to nainital, and raj relocates to cape town, south africa for training. during a visit to a club, raj meets the vivacious and fun-loving jiya (priyanka chopra). jiya falls in love with raj due to his clean personality, but raj is unable to forget kajal. after completing his training, raj goes back to india where he finds that jiya has already arrived and is living as a paying guest'
    #plot_tokenized = tokenizer.encode(plot)

    model_path = select_model_by_genre(genre)
    model = torch.load(model_path)

    #Run the functions to generate the alternative endings
    alternative_end, full_test_generated_plot = text_generation(model, tokenizer, selected_plot)
    #generated_plot = generate(model, tokenizer, plot_tokenized, entry_count=1, entry_length=30, #maximum number of words
    #top_p=0.8, temperature=1.)

    #return alternative_end[-1]
    return alternative_end
