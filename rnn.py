import collections
import datetime
import fluidsynth
import numpy as np
import pathlib
import pretty_midi
import tensorflow as tf
import pandas as pd
import os

from IPython import display
from matplotlib import pyplot as plt
from typing import Optional

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

# Set seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Function to convert MIDI to notes
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

# Function to create sequences for training
def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size=128) -> tf.data.Dataset:
    seq_length = seq_length + 1

    # Create sliding windows
    windows = dataset.window(seq_length, shift=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize pitches and split labels
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(['pitch', 'step', 'duration'])}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

# Function to generate new MIDI file
def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity=100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(midi_note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

# Load custom MIDI file
sample_file = 'elise.mid'  # Replace with your MIDI file path
raw_notes = midi_to_notes(sample_file)

# Plot piano roll for inspection
def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    plt.title(f"First {count or len(notes['pitch'])} notes")

plot_piano_roll(raw_notes, count=100)

# Convert notes to dataset
key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

# Create sequences
seq_length = 50
vocab_size = 128
seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

# Batch and prepare the dataset
batch_size = 64
train_ds = (seq_ds
            .shuffle(len(train_notes) - seq_length)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .repeat()
            .prefetch(tf.data.experimental.AUTOTUNE))

# Model save path
model_path = "saved_music_model"

# Check if model exists
if os.path.exists(model_path):
    print("Loading saved model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Training new model...")
    # Build the model
    input_shape = (seq_length, 3)
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(128)(x)
    outputs = {
        'pitch': tf.keras.layers.Dense(vocab_size, activation='softmax', name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss={
            'pitch': 'sparse_categorical_crossentropy',
            'step': 'mse',
            'duration': 'mse',
        }
    )

    # Commented out training code
    # epochs = 50
    # steps_per_epoch = len(train_notes) // batch_size
    # model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

    # Save the trained model
    # model.save(model_path)

# Generate new music
def generate_music(seed_notes: pd.DataFrame, num_notes=100):
    # Ensure seed_notes only has the required columns
    seed_notes = seed_notes[['pitch', 'step', 'duration']]
    generated_notes = seed_notes.copy()

    for _ in range(num_notes):
        # Ensure input sequence has the correct shape (seq_length, 3)
        input_sequence = np.expand_dims(generated_notes[-seq_length:].values, axis=0)
        predictions = model.predict(input_sequence)

        pitch = np.argmax(predictions['pitch'][0])  # Convert pitch prediction to an integer
        step = max(0, predictions['step'][0, 0])   # Ensure step duration is non-negative
        duration = max(0, predictions['duration'][0, 0])  # Ensure note duration is non-negative

        # Append the generated note to the DataFrame
        generated_notes = pd.concat([
            generated_notes,
            pd.DataFrame({'pitch': [pitch], 'step': [step], 'duration': [duration]})
        ], ignore_index=True)

    return generated_notes


# Generate new music and save to MIDI
seed_notes = raw_notes[:seq_length]
new_music = generate_music(seed_notes)
new_midi = notes_to_midi(new_music, out_file='new_music.midi', instrument_name='Acoustic Grand Piano')

# Play generated music
display.Audio(new_midi.fluidsynth(fs=_SAMPLING_RATE), rate=_SAMPLING_RATE)