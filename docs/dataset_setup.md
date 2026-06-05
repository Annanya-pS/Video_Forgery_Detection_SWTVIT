# Dataset Setup — FaceForensics++

## 1. Request access

FaceForensics++ requires registration. Fill out the form at:  
https://github.com/ondyari/FaceForensics

You will receive a download script (`download-FaceForensics.py`).

## 2. Download the four classes used in SWTVIT

Download only the `c23` (H.264 CRF-23) compressed videos for these four classes:

```bash
# Original (Real) videos
python download-FaceForensics.py \
    /path/to/data \
    -d original \
    -c c23 \
    -t videos

# Deepfakes
python download-FaceForensics.py \
    /path/to/data \
    -d Deepfakes \
    -c c23 \
    -t videos

# FaceSwap
python download-FaceForensics.py \
    /path/to/data \
    -d FaceSwap \
    -c c23 \
    -t videos

# Face2Face
python download-FaceForensics.py \
    /path/to/data \
    -d Face2Face \
    -c c23 \
    -t videos
```

Use `--num_videos 125` to limit to 125 videos per class (the amount used in this project).

## 3. Expected directory structure

After downloading, your data directory should look like this:

```
data/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│               ├── 000.mp4
│               ├── 001.mp4
│               └── ...   (125 videos)
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── videos/
    │           ├── 000_003.mp4
    │           └── ...   (125 videos)
    ├── FaceSwap/
    │   └── c23/
    │       └── videos/
    │           └── ...   (125 videos)
    └── Face2Face/
        └── c23/
            └── videos/
                └── ...   (125 videos)
```

Total: 500 videos across 4 classes (125 per class).

## 4. Upload to Google Drive (for Colab)

Mount your Drive in Colab and upload the dataset to:

```
MyDrive/data/
```

The notebook's `FF_SOURCES` dict (Cell 3) already points to this location by default.

## 5. Update paths (if different)

If your data lives elsewhere, edit `FF_SOURCES` in **Cell 3** of the notebook:

```python
FF_SOURCES = {
    0: '/content/drive/MyDrive/YOUR_PATH/original_sequences/youtube/c23/videos',
    1: '/content/drive/MyDrive/YOUR_PATH/manipulated_sequences/Deepfakes/c23/videos',
    2: '/content/drive/MyDrive/YOUR_PATH/manipulated_sequences/FaceSwap/c23/videos',
    3: '/content/drive/MyDrive/YOUR_PATH/manipulated_sequences/Face2Face/c23/videos',
}
```

## 6. Disk space

| Class | Videos | Approx size (c23) |
|---|---|---|
| Original | 125 | ~1.5 GB |
| Deepfakes | 125 | ~1.2 GB |
| FaceSwap | 125 | ~1.0 GB |
| Face2Face | 125 | ~1.1 GB |
| **Total** | **500** | **~5 GB** |

## Notes

- NeuralTextures was excluded from this project (only 3 manipulation types + Real).
- The `c23` format was chosen as it represents realistic distribution conditions (compressed).
- A Colab Pro or Pro+ subscription is recommended for the ~5 hour training run.
