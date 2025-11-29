import re
from pathlib import Path
from pylatexenc.latexwalker import LatexWalker
from pylatexenc.latex2text import LatexNodes2Text
import base64

def extract_raw_figures(root_dir: str):
    fig_re = re.compile(
        r"\\begin\{(?:figure|wrapfigure|minipage|tikzpicture)\*?\}.*?\\end\{(?:figure|wrapfigure|minipage|tikzpicture)\*?\}",
        re.DOTALL
    )
    results = []
    for path in Path(root_dir).rglob("*.tex"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in fig_re.finditer(text):
            results.append({
                "file": str(path),
                "snippet": match.group(0).strip()
            })
    return results

def extract_figure_name_and_caption(graph_str: str, caption_str: str):
    # graph
    graph=LatexWalker(graph_str.strip())
    graph, _, _ = graph.get_latex_nodes()

    # caption
    caption = LatexNodes2Text().latex_to_text(caption_str).strip()

    return [graph[0].nodeargs[0].nodelist[0].chars, caption]


def image_to_base64(image_path: str) -> str:
    """
    Convert a single image file to a Base64 string.
    
    Args:
        image_path (str): Path to the image file (.png, .jpg, etc.)
    
    Returns:
        str: Base64-encoded text of the image
    """
    with open(image_path, "rb") as f:
        encoded_bytes = base64.b64encode(f.read())
    return encoded_bytes.decode("utf-8")