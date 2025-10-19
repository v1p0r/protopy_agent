## LaTeX Figure Caption Extractor

Extract figure image paths and their captions from LaTeX papers. This repo provides:

- A small utility module (`src/utils.py`) to parse `figure`/`figure*` environments and recover each `\includegraphics{...}` path with its `\caption{...}` text (converted to plain text).
- An example Jupyter notebook (`image_caption_extraction.ipynb`) that demonstrates end‑to‑end extraction on papers placed under `data/input/latex/`.

### Why this exists
Building datasets or analytics around research figures often requires linking images to their natural‑language captions. This project streamlines that step for LaTeX sources.

## Project structure

```
protopy_agent/
  data/
    input/
      latex/            # Each paper in its own folder with .tex and figures/
        <paper-name>/
          *.tex
          figures/
            ...        # images referenced by \includegraphics
    output/             # You can write your results here (optional)
  src/
    utils.py            # Extraction helpers
  image_caption_extraction.ipynb  # Example workflow
```

An example paper is included at `data/input/latex/arXiv-2510.14980v1/` with LaTeX sources and a `figures/` directory.

## Methodology: Image–Caption Extraction

This section explains how figures and captions are identified and paired from LaTeX sources.

- **Figure block detection**: We search each `.tex` file recursively for `\begin{figure}` or `\begin{figure*}` blocks and their matching `\end{figure}`/`\end{figure*}` using a DOTALL regex. Each match yields a raw figure snippet.
- **Image path retrieval**: Within a figure snippet, we locate the line containing `\\includegraphics` and parse its first braced argument, which should be an image path (e.g., `figures/teaser.pdf`).
- **Caption text extraction**: We locate the line containing `\\caption{...}` (optionally with a short form like `\\caption[short]{long}`) and use `pylatexenc` to convert LaTeX to plain text for the long caption.
- **Pair construction**: For each figure snippet, we produce `(image_path, caption_text)`. The notebook shows example output and how to post‑process these pairs.

Edge cases to consider in future improvements:

- Multi‑line `\\includegraphics` options/paths and multi‑line `\\caption{...}` blocks
- Images referenced via macros or discovered through `\\graphicspath`
- Multiple images within a single figure block (e.g., subfigures)
- Non‑standard environments or custom figure wrappers

## Requirements

- Python 3.10+
- `pylatexenc` for LaTeX parsing
- Jupyter (optional, for running the notebook)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart (Notebook)

1. Launch Jupyter and open `image_caption_extraction.ipynb`:
   ```bash
   jupyter notebook
   ```
2. Adjust `input_path` (defaults to `data/input/latex`) and run all cells.
3. The notebook prints `(image_path, caption)` pairs extracted from figures in the LaTeX sources.

## Quickstart (Python)

Use the helpers directly if you prefer scripting:

```python
from src.utils import extract_raw_figures, extract_figure_name_and_caption
import os, re

root = "data/input/latex"
pairs = []

for paper_dir in os.listdir(root):
    full_dir = os.path.join(root, paper_dir)
    if not os.path.isdir(full_dir):
        continue
    for fig in extract_raw_figures(full_dir):
        lines = fig["snippet"].splitlines()
        image_line = next(l for l in lines if "\\includegraphics" in l)
        caption_line = next(l for l in lines if re.search(r"\\caption(?:\\s*\[[^\]]*\])?\\s*\{", l))
        image_path, caption = extract_figure_name_and_caption(image_line, caption_line)
        pairs.append((os.path.join(full_dir, image_path), caption))

# pairs now contains (image_path, plain_text_caption)
```

## Output

The basic workflow yields an in‑memory list of `(image_path, caption)` tuples. You can serialize this to CSV/JSON, or copy images to `data/output/` as needed for downstream tasks.

## Notes and limitations

- LaTeX figure formatting varies widely. The regex/structure here targets standard `figure` blocks with `\includegraphics{...}` and `\caption{...}` on one or multiple lines.
- `pylatexenc` is used to convert caption LaTeX to plain text; complex math or custom macros may need additional handling.
- If your paths are given via macros or `\graphicspath`, you may need to resolve those before constructing absolute image paths.

## License

Specify a license for this project (e.g., MIT) if you plan to distribute it.


