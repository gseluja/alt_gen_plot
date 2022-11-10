#from alt_plot_gen.ml_logic.params import (CHUNK_SIZE,
#                                      DATASET_SIZE,
#                                      VALIDATION_DATASET_SIZE)

from alt_plot_gen.ml_logic.data import clean_data, split_dataset
#from alt_plot_gen.ml_logic.preprocessor import preprocess_features
from alt_plot_gen.ml_logic.generation import generate
from alt_plot_gen.ml_logic.encoders import tokenize_plots
from alt_plot_gen.ml_logic.model import get_pretrained, train

import os
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH, CSV_TEST_FILE

def preprocess():
    """
    Preprocess the dataset by
    1) cleaning
    2) preprocessing
    3) tokenizing
    """
    print("\n⭐️ use case: preprocess")

    df = clean_data()

    train_set, test_set = split_dataset(df)
    # save test set to be used in website
    test_set_file = os.path.join(LOCAL_DATA_PATH, CSV_TEST_FILE)

    test_set.to_csv(test_set_file, index = False)

    cleaned_row_count = len(train_set)

    print(f"\n✅ data processed: ({cleaned_row_count} cleaned)")

    #df = preprocess_features(df)

    train_set_token = tokenize_plots(train_set)  #list of tensors (tokenized plots) #all genres

    print(f"\n✅ data tokenized")

    return train_set_token, test_set


def build_train_model(dataset):

    # initialize model: get_pretrained from gpt-2
    tokenizer, model = get_pretrained()

    print(f"\n✅ got pretrained model")

    # model params
    batch_size=16
    epochs=5
    lr=2e-5
    max_seq_len=400
    warmup_steps=200

    # pack tensor and train the model incrementally
    model = train(dataset, model, tokenizer,
                batch_size=batch_size, epochs=epochs, lr=lr,
                max_seq_len=max_seq_len, warmup_steps=warmup_steps,
                gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
                test_mode=False,save_model_on_epoch=False)

    print(f"\n✅ data trained")

    return model, tokenizer


# Generate multiple sentences
def text_generation(model, tokenizer, test_data):
    entry_length = 60
    x = generate(model, tokenizer, test_data, entry_count=1, entry_length=entry_length, top_p=0.8, temperature=1.)

    print(x)
    print(f"\n✅ generation created")

    # Show only generated text
    a = test_data.split()[-10:] # Get the matching string we want
    b = ' '.join(a)
    c = ' '.join(x) # Get all that comes after the matching string
    my_generation = c.split(b)[-1]

    # Finish the sentences when there is a point, remove after that
    just_alternative =[]
    to_remove = my_generation.split('.')[-1]
    just_alternative = my_generation.replace(to_remove,'')
    return just_alternative, x



if __name__ == '__main__':
    pass
