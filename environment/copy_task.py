#!/usr/bin/python3

# import matplotlib.pyplot as plt
import numpy as np

import os

MODEL_NAME = "lstm_copy_model.ckpt"
MODEL_PATH = "./" + MODEL_NAME

TRAIN_VIS_PATH = "./"

batch_size = 100
stop_at = 0.0080  # End training at required loss

seq_len = 20  # Change this to change the sequence length (Kept at 20 for initial training)
bits = 8  # The actual vector size for copying task
in_bits = bits + 2  # The extra side track
out_bits = bits

lr = 3e-5
m = 0.9  # Momentum
grad_clip = 10
act_seq_len = (
                          seq_len * 2) + 2  # Actual sequence lenght which includes the delimiters (Start and Stop bits on the side tracks)
no_hidden = [256, 256, 256]  # No of LSTM layer and units per layer


def generate_patterns(no_of_samples=100, max_sequence=20, min_sequence=1, in_bits=3, out_bits=1, pad=0.001,
                      low_tol=0.001, high_tol=1.0):  # Function to generate sequences of different lengths

    ti = []
    to = []

    for _ in range(no_of_samples):
        seq_len_row = np.random.randint(low=min_sequence, high=max_sequence + 1)

        pat = np.random.randint(low=0, high=2, size=(seq_len_row, out_bits))
        pat = pat.astype(np.float32)

        # Applying tolerance (So that values don't go to zero and cause NaN errors)
        pat[pat < 1] = low_tol
        pat[pat >= 1] = high_tol

        # Padding can be added if needed
        x = np.ones(((max_sequence * 2) + 2, in_bits), dtype=pat.dtype) * pad
        y = np.ones(((max_sequence * 2) + 2, out_bits), dtype=pat.dtype) * pad  # Side tracks are not produced

        # Creates a delayed output (Target delay)
        x[1:seq_len_row + 1, 2:] = pat
        y[seq_len_row + 2:(2 * seq_len_row) + 2, :] = pat  # No side tracks needed for the output

        x[1:seq_len_row + 1, 0:2] = low_tol
        x[0, :] = low_tol
        x[0, 1] = 1.0  # Start of sequence
        x[seq_len_row + 1, :] = low_tol
        x[seq_len_row + 1, 0] = 1.0  # End of sequence

        ti.append(x)
        to.append(y)

    return ti, to

if __name__ == "__main__":
    x, y = generate_patterns(1)
    print(x[0])
    print(y[0])