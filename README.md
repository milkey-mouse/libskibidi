# `libskibidi`: Two Minute Papers for Gen Alpha

`libskibidi` is ed tech for the next generation of ML engineers. It automatically generates minecraft parkour videos with AI narration of the latest arXiv ML papers' abstracts superimposed. More specifically, it:

- Fetches top N latest arXiv papers in a category
- Narrates their abstracts with randomly-chosen OpenAI TTS voices
- Chooses a random segment of a random "free to use" Minecraft parkour video long enough to fit the narration
- Muxes the abstract text onto the video with FFmpeg
- Uses a "novel" forced text alignment "implementation" based on Whisper to identify and highlight each word as it's spoken for maximum brainrot, while retaining obscure spellings and LaTeX present in the original abstract

# Demo

https://github.com/user-attachments/assets/f1444566-d9dc-4ad6-826d-40cd579a338a

# Usage

`libskibidi` requires Python 3.12 or later:

```
python3 --version  # libskibidi requires >=3.12
```

The single non-`pip`-installable prerequisite is ffmpeg:

```
sudo apt install ffmpeg
```

Clone the repo:

```
git clone https://github.com/milkey-mouse/libskibidi
cd libskibidi
```

Create a `venv`:

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Delve into the future of education:

```
python3 libskibidi.py
```
