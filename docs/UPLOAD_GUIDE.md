# Step-by-Step GitHub Upload Guide — SWTVIT

Complete walkthrough from zero to a live, polished GitHub repository.

---

## PHASE 1 — Prepare your files locally

### Step 1: Collect everything in one folder

Create a folder on your computer called `SWTVIT/`. Place inside it:

```
SWTVIT/
├── SWTVIT_pipeline.ipynb          ← your notebook (cleaned — see Step 2)
├── README.md                      ← provided
├── requirements.txt               ← provided
├── .gitignore                     ← provided
├── LICENSE                        ← provided
├── CITATION.cff                   ← provided
├── assets/
│   ├── gradcam_deepfake.png       ← screenshot from your report (Figure 4.5)
│   ├── gradcam_faceswap.png       ← screenshot from your report (Figure 4.6)
│   ├── training_curves.png        ← saved in your Drive: checkpoints/training_curves.png
│   └── confusion_matrix.png       ← saved in your Drive: checkpoints/cm_video.png
└── docs/
    ├── dataset_setup.md           ← provided
    └── project_report.pdf         ← your submitted report PDF
```

### Step 2: Clean the notebook before uploading

Open `SWTVIT_pipeline.ipynb` in Colab, then:

1. **Remove personal Google Drive paths** — replace all hardcoded paths like
   `/content/drive/MyDrive/data/...` with a comment:
   ```python
   # Update this path to your FF++ data location
   FF_SOURCES = {
       0: '/content/drive/MyDrive/data/original_sequences/youtube/c23/videos',
       ...
   }
   ```
   Leave the structure but add the comment so others know to change it.

2. **Clear all outputs** — in Colab menu: `Edit → Clear all outputs`  
   Then `Runtime → Run all` once to get fresh outputs.  
   OR just clear outputs and upload without outputs (cleaner for GitHub).

3. **Remove the debug/exploration cells** — cells that just print file lists
   (`running this to see what videos I have`) can be deleted or kept as optional.

4. **Download the cleaned notebook** — `File → Download → Download .ipynb`

### Step 3: Collect your asset images

From your Google Drive (`checkpoints/` folder), download:
- `training_curves.png`
- `cm_frame.png` → rename to `confusion_matrix_frame.png`
- `cm_video.png` → rename to `confusion_matrix_video.png`

From your project report PDF, take screenshots of:
- Figure 4.5 (GradCAM Deepfake output) → save as `gradcam_deepfake.png`
- Figure 4.6 (GradCAM FaceSwap output) → save as `gradcam_faceswap.png`

Place all images in `SWTVIT/assets/`.

---

## PHASE 2 — Create the GitHub repository

### Step 4: Create a GitHub account (if needed)

Go to https://github.com and sign up with your college email.

### Step 5: Create a new repository

1. Click the **+** icon (top right) → **New repository**
2. Fill in:
   - **Repository name:** `SWTVIT` (or `deepfake-detection-swtvit`)
   - **Description:** `4-class deepfake detection using SWT + Vision Transformer with GradCAM. 90% video accuracy on FaceForensics++.`
   - **Visibility:** Public ✅
   - **Initialize with README:** ❌ NO (you already have one)
   - **Add .gitignore:** ❌ NO (you already have one)
   - **Choose license:** ❌ NO (you already have one)
3. Click **Create repository**

### Step 6: Add topics/tags

After creating the repo, click the **gear icon** next to "About" and add:
```
deepfake-detection  vision-transformer  wavelet-transform
pytorch  faceforensics  gradcam  media-forensics  deep-learning
```

---

## PHASE 3 — Upload your files

### Option A: Upload via GitHub website (easiest, no Git knowledge needed)

1. On your new empty repo page, click **uploading an existing file**
2. Drag and drop **all files and folders** from your `SWTVIT/` folder
   - Note: GitHub web UI doesn't support folder drag-drop well.
     Upload files in batches by folder:
     - First: root files (README, requirements, .gitignore, LICENSE, CITATION.cff, notebook)
     - Then: create `assets/` folder → upload images
     - Then: create `docs/` folder → upload docs

   **How to create a folder via web UI:**
   - Click **Add file → Create new file**
   - Type `assets/placeholder.txt` — this creates the folder
   - Commit it, then upload your images into `assets/`

3. For each batch, scroll down → **Commit changes**
   - Commit message: `Add notebook and documentation`

### Option B: Upload via Git CLI (cleaner, one command)

Install Git from https://git-scm.com if you don't have it.

```bash
# 1. Open terminal and go to your SWTVIT folder
cd path/to/SWTVIT

# 2. Initialize git
git init

# 3. Add all files
git add .

# 4. First commit
git commit -m "Initial commit: SWTVIT deepfake detection pipeline"

# 5. Connect to GitHub (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/SWTVIT.git

# 6. Push
git branch -M main
git push -u origin main
```

If asked for password, use a **Personal Access Token** (not your GitHub password):
- GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Check `repo` scope → Generate → copy the token → use as password

---

## PHASE 4 — Upload the model checkpoint (not to GitHub)

The `best_model.pth` file (~85 MB) is too large for GitHub. Host it separately.

### Option A: Google Drive (simplest)

1. Upload `best_model.pth` to your Google Drive
2. Right-click → **Share** → **Anyone with the link can view**
3. Copy the link
4. In `README.md`, find this line and replace the placeholder:
   ```
   Download it from Google Drive: **[best_model.pth — ~85 MB](YOUR_DRIVE_LINK_HERE)**
   ```

### Option B: HuggingFace Hub (more professional, standard in ML community)

```bash
pip install huggingface_hub
huggingface-cli login   # create free account at huggingface.co first
```

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("YOUR_USERNAME/swtvit-deepfake-detection")
api.upload_file(
    path_or_fileobj="checkpoints/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id="YOUR_USERNAME/swtvit-deepfake-detection",
)
```

Then update the README download link to the HuggingFace URL.

---

## PHASE 5 — Update the README with your real links

Open `README.md` and make these replacements:

1. Find `YOUR_USERNAME` → replace with your actual GitHub username (appears in Colab badge and CITATION)
2. Find `YOUR_DRIVE_LINK_HERE` → replace with your model download link
3. After uploading assets, verify the image paths are correct:
   ```markdown
   ![GradCAM Deepfake](assets/gradcam_deepfake.png)
   ```
   You can add these image lines to the "GradCAM Localization" section of the README.

---

## PHASE 6 — Final checks

Go to your GitHub repo URL: `https://github.com/YOUR_USERNAME/SWTVIT`

Verify:
- [ ] README renders correctly with the accuracy badges at the top
- [ ] "Open in Colab" badge links to your notebook
- [ ] Topics/tags are visible under "About"
- [ ] `assets/` folder shows your images
- [ ] `docs/dataset_setup.md` is accessible
- [ ] `requirements.txt` and `.gitignore` are present
- [ ] No `.pth` files or video files were accidentally committed (check `.gitignore` worked)

---

## Common mistakes to avoid

| Mistake | How to avoid |
|---|---|
| Committing `best_model.pth` to GitHub | It's in `.gitignore` — double-check before pushing |
| Committing video files | Also in `.gitignore` |
| Notebook with Drive-specific absolute paths | Change to relative or add comments |
| Missing Colab badge link | Update `YOUR_USERNAME` in README |
| Empty `assets/` folder | README images will show broken — upload pngs first |
