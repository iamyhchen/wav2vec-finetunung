# wav2vec-finetunung
This project focuses on fine-tuning the Wav2Vec2 model for automatic speech recognition (ASR).

## Data prepare
1. `vocab.json`

You need to prepare a vocabulary file (`vocab.json`) that defines the mapping between characters (or tokens) and their corresponding integer IDs. This vocabulary is essential for the tokenizer to convert text into input IDs for training.

The following three tokens must be included in your vocabulary:
- `[PAD]`: Padding token (used to align input lengths)
- `[UNK]`: Unknown token (used for characters not in the vocabulary)
- `|`: Word separator (used instead of spaces)
- `"0"`, `"1"`, etc.: Your actual vocabulary characters (can be letters, phonemes, etc.)

Example format:

```
{
  "[PAD]": 0,
  "[UNK]": 1,
  "|": 2,
  "0": 3,
  "1": 4,
  "2": 5,
  ...
}
```
2. `dataset`

Prepare your dataset in the following structure:
```
dataset/
├── train/
│ ├── train.csv
│ └── corpus/
│   ├── audio1.wav
│   └── ...
├── val/
│ ├── val.csv
│ └── corpus/
│   ├── audio2.wav
│   └── ...
├── test/
│ ├── test.csv
│ └── corpus/
│   ├── audio3.wav
│   └── ...
```

Prepare a CSV file with the following format:
```
audio,transcript
dataset/train/corpus/audio1.wav,transcript1
...
```

## Installation
1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Getting Started
Run the full pipeline:
```bash
./run.sh
```
