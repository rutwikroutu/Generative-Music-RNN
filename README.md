# Music Generation Tool Using Machine Learning

This repository implements a tool that generates new music by training a machine learning model on existing MIDI files. The tool uses a combination of MIDI processing, deep learning with TensorFlow, and a custom data pipeline to create sequences of notes that resemble the input music. The generated music can then be converted back into a MIDI file and played back. 

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [How It Works](#how-it-works)
4. [Key Functions](#key-functions)
5. [Usage Instructions](#usage-instructions)
6. [Model Training and Generation](#model-training-and-generation)
7. [Example](#example)
8. [License](#license)

## Overview

The tool takes an existing MIDI file, processes it into sequences of musical notes (including pitch, step, and duration), and uses this data to train a machine learning model. This model can then generate new music based on patterns learned from the input file. The generated music is then saved in MIDI format, which can be played or edited in any compatible MIDI software.

## Dependencies

The project requires the following Python libraries:
- `tensorflow` – For building and training the deep learning model.
- `pretty_midi` – To parse and manipulate MIDI files.
- `matplotlib` – To plot piano rolls for visualization.
- `pandas` – For data manipulation.
- `numpy` – For numerical operations.
- `fluidsynth` – For playing MIDI files.
- `IPython` – For audio display in Jupyter notebooks.

You can install these dependencies using `pip`:

```bash
pip install tensorflow pretty_midi matplotlib pandas numpy fluidsynth IPython
