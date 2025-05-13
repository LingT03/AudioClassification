# Audio Classification with PyTorch

This repository contains my code and experiments for audio classifcation of the Audioset dataset from Google.

This repo is my submission for my Deep Learning Final at MSU Denver for Spring 2025.

## Inspiration

General curiosity about audio classifcation and deep learning. I had previous experience with image classifcation with tensorflow and Keras, but I wanted to try PyTorch. I also wanted to try audio classifcation, so I decided to combine the two.

## Dataset

[Audioset](https://research.google.com/audioset/) is a large-scale dataset of human-labeled audio events. It contains over 2 million human-labeled 10-second sound clips drawn from YouTube videos. The dataset is organized into a hierarchy of 527 audio event classes, which are divided into 2 levels: the top level contains 527 classes, and the second level contains 20 superclasses.

### Scripts

- `scrapeYT.py`: Script to scrape YouTube videos and extract audio segments.

```
When running again start at index 1992 -- the last index of the previous run (I got flagged as a bot)
```

## Libraries and Tools

- general
  - [numpy](https://numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
- audio
  - [librosa](https://librosa.org/)
  - [subprocess](https://docs.python.org/3/library/subprocess.html)
  - [soundfile](https://pysoundfile.readthedocs.io/en/latest/)
  - [pydub](https://pydub.com/)
  - [yt_dlp](https://github.com/yt-dlp/yt-dlp)
- deep learning
  - [torch](https://pytorch.org/)

## References

_see /docs_

- [Music and Instrument Classification using Deep Learning Technics](docs/lara.pdf)

- [MUSICAL INSTRUMENT SOUND CLASSIFICATION WITH DEEP CONVOLUTIONAL NEURAL NETWORK USING FEATURE FUSION APPROACH](docs/Taejin.pdf)

- [Audioset](https://research.google.com/audioset/about.html)

- [PyTorch](https://pytorch.org/)
