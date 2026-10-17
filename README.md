# Urban Sound Binary Classification with CNN

A beginner-friendly deep learning mini-project that classifies short UrbanSound8K recordings as **siren** or **non-siren** using a small Convolutional Neural Network (CNN).

## Problem statement

Convert each audio recording into a mel-spectrogram (an image-like representation of sound over time and frequency), then train a CNN to classify it into two classes:

- `1` — siren
- `0` — non-siren (the other nine UrbanSound8K classes)

## Dataset

- Dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- Data loader: [Soundata](https://github.com/soundata/soundata)
- 8,732 labeled audio excerpts, each up to 4 seconds
- The original dataset has 10 classes, including `siren`

The notebook uses folds 1–9 to build training and validation subsets and holds out fold 10 for testing. It uses up to 400 examples per class for the development subset to keep the experiment small enough for a beginner project.

## Run the notebook

Open `siren_vs_non_siren_cnn.ipynb` in [Google Colab](https://colab.research.google.com/) and run cells from top to bottom. The notebook is deliberately split into small steps:

1. Install and import libraries.
2. Download and validate the dataset using Soundata.
3. Inspect the metadata and create binary labels.
4. Prepare balanced training, validation, and test sets.
5. Convert audio to mel-spectrograms.
6. Train a small CNN and evaluate its results.

The first dataset download is large and may take a while. This repository contains code and documentation, not the audio files. The notebook has not been run end-to-end against a full dataset in this environment, so accuracy must be measured by running it in Colab.

## Model

- Two convolution layers (8 and 16 filters), using ReLU
- Max pooling
- Dense hidden layer with 32 units and ReLU
- One sigmoid output for siren probability
- Adam optimizer and binary cross-entropy loss

## Citation and dataset terms

Read the [official UrbanSound8K terms](https://urbansounddataset.weebly.com/urbansound8k.html) before using the data. For academic work, cite:

J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” 22nd ACM International Conference on Multimedia, 2014.

Soundata is a tool for downloading, validating, and loading supported datasets; it does not replace the dataset's own terms.

## Files

- `siren_vs_non_siren_cnn.ipynb` — annotated step-by-step notebook
- `README.md` — project overview and run instructions
