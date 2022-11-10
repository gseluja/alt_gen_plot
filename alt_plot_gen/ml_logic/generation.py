import torch
import torch.nn.functional as F
from tqdm import trange


def generate(model, tokenizer, prompt,
            entry_count=1, entry_length=30,
            top_p=0.8, temperature=1.):
    # entry_length is the maximum number of words
    # top_p
        # The model will sort the word probabilities in descending order.
        # Then, it will sum those probabilities up to p while dropping the other words.
        # This means the model only keeps the most relevant word probabilities, but does not only keep the best one,
        # as more than one word can be appropriate given a sequence.

    # temperature
        # Is used to scale the probabilities of a given word being generated.
        # Therefore, a high temperature forces the model to make more original predictions while a smaller one keeps the
        # model from going off topic.

    model.eval()
    generated_num = 0
    generated_list = []
    filter_value = -float("Inf")

    with torch.no_grad():
        for entry_idx in trange(entry_count):
            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1
                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
                generated_list.append(output_text)

    return generated_list
