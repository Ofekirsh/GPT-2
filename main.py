from __future__ import annotations
import numpy
import json

import torch

if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from transformer import TransformerLM
    import data
    import lm

    # Change file_type to HEB or SHAKE to train on the Hebrew or Shakespeare dataset
    file_type = "heb"

    with open(f"conf/conf_{file_type}.json", "r") as f:
        config = json.load(f)
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    data_path = config["data_path"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = embed_size * 4
    learning_rate = config["learning_rate"]
    gradient_clipping = config["gradient_clipping"]
    num_batches_to_train = config["num_batches_to_train"]
    weight_decay = config["weight_decay"]

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = True,
        )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95], weight_decay=weight_decay)



    model.train()

    num_batches = 0
    while num_batches < num_batches_to_train:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train: break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            logits, attentions = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sample_sequence, all_attentions = model.better_sample_continuation(tokenizer.tokenize("Hello"),
                                                                                           500, temperature=0.5, topK=5)

                        sampled = tokenizer.detokenize(sample_sequence)

                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")

    # Save the model weights
    torch.save(model.state_dict(), 'model_weights.pth')