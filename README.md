# Urban Sound Binary Classification with CNN

A beginner-friendly mini-project that classifies UrbanSound8K audio recordings as **siren** or **non-siren** using a small Convolutional Neural Network (CNN).

## Problem statement
Build a lightweight CNN that identifies whether a short urban audio clip contains a siren. The notebook converts each audio recording into a **mel-spectrogram** (a visual representation of sound energy over time and frequency) and trains a binary classifier on that representation.

## Dataset
- **Dataset:** [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- **Loader:** [Soundata](https://github.com/soundata/soundata)
- **Size:** 8,732 labeled audio excerpts, each up to 4 seconds.
- **Original categories:** air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music.
- **Binary labels:** `siren = 1`; all other original categories are grouped as `non-siren = 0`.

## Notebook
Open `siren_vs_non_siren_cnn.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter and run the cells in order. It installs dependencies, downloads and validates data with Soundata, inspects the CSV metadata, prepares a small balanced subset, creates mel-spectrograms, trains a small CNN, and reports test metrics and a confusion matrix.

The notebook keeps the experiment small by using a subset of training examples. The actual accuracy is unknown until you run the notebook. A first-time download and audio preprocessing can take several minutes and needs enough disk space/RAM.

## Model
- Input: fixed-size mel-spectrogram
- Convolution layers: 8 and 16 filters, ReLU activation
- Dense hidden layer: 32 units, ReLU
- Output: one sigmoid unit (siren probability)
- Loss: binary cross-entropy
- Optimizer: Adam

## Dataset use and citation
UrbanSound8K has its own terms of use. Review the dataset's official terms before using it; the notebook/repository contains code only, not dataset audio. For academic work, cite:

J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” 22nd ACM International Conference on Multimedia, 2014.

The Soundata loader's dataset information and citation are documented in [the loader source](https://github.com/soundata/soundata/blob/main/soundata/datasets/urbansound8k.py). Soundata itself is a tool to download, validate, and load supported audio datasets; it does not replace the dataset's own terms.

## Files
- `siren_vs_non_siren_cnn.ipynb` — end-to-end notebook
- `README.md` — project overview and run instructions
