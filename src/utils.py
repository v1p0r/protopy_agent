import re
from pathlib import Path
from pylatexenc.latexwalker import LatexWalker
from pylatexenc.latex2text import LatexNodes2Text
import base64
from collections import defaultdict
import json

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

def group_text_by_section(blocks):
    grouped = defaultdict(list)

    for entry in blocks:
        section = entry.get("section", "Unknown")
        text = entry.get("text", "")
        grouped[section].append(text)

    # Join all text in the same section
    return {sec: " ".join(texts) for sec, texts in grouped.items()}

def load_s2orc_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


###########################################################
# unused functions
###########################################################

def count_tokens_for_file(encoding, file_path):
    """Count tokens for a single image (Base64 encoded)."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return len(encoding.encode(b64))

def count_total_tokens_in_folder(folder_path, encoding_name="cl100k_base"):
    """
    Loop through all image files in a folder and sum token counts.
    Supports recursive search.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    total_tokens = 0
    per_file = {}

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                path = os.path.join(root, file)
                tokens = count_tokens_for_file(encoding, path)
                per_file[path] = tokens
                total_tokens += tokens
    
    print("\n=== Token counts per file ===")
    for f, t in per_file.items():
        print(f"{f}: {t} tokens")
    print(f"\nTotal tokens across folder: {total_tokens}")
    return total_tokens, per_file


###########################################################
# referenced functions
###########################################################
def extract_planning(trajectories_json_file_path):
    with open(trajectories_json_file_path) as f:
        traj = json.load(f)

    context_lst = []
    for turn in traj:
        if turn['role'] == 'assistant':
            # context_lst.append(turn['content'])
            content = turn['content']
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            context_lst.append(content)


    context_lst = context_lst[:3] 

    return context_lst

def content_to_json(data):
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)


    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    except json.JSONDecodeError as e:
        # print(e)
        return content_to_json2(data)
        
    
def content_to_json2(data):
    # remove [CONTENT][/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)


    # ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)

    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    
    except json.JSONDecodeError as e:
        # print("Json parsing error", e)
        return content_to_json3(data)

def content_to_json3(data):
    # remove [CONTENT] [/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)

    # remove ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data) 
    clean_data = re.sub(r'"""', '"', clean_data)  # Replace triple double quotes
    clean_data = re.sub(r"'''", "'", clean_data)  # Replace triple single quotes
    clean_data = re.sub(r"\\", "'", clean_data)  # Replace \ 

    # JSON parsing
    try:
        json_data = json.loads(f"""{clean_data}""")
        return json_data
    
    except json.JSONDecodeError as e:
        # print(e)
        
        # print(f"[DEBUG] utils.py > content_to_json3 ")
        # return None 
        return content_to_json4(data)
    
def content_to_json4(data):
    # 1. Extract Logic Analysis, Task list
    pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)

    if match:
        logic_analysis = json.loads(match.group(1))
        task_list = json.loads(match.group(2))

        result = {
            "Logic Analysis": logic_analysis,
            "Task list": task_list
        }
    else:
        result = {}

    # print(json.dumps(result, indent=2))
    return result

def format_json_data(data):
    formatted_text = ""
    for key, value in data.items():
        formatted_text += "-" * 40 + "\n"
        formatted_text += "[" + key + "]\n"
        if isinstance(value, list):
            for item in value:
                formatted_text += f"- {item}\n"
        else:
            formatted_text += str(value) + "\n"
        formatted_text += "\n"
    return formatted_text

def extract_code_from_content(content):
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if len(code) == 0:
        return ""
    else:
        return code[0]
    
def extract_code_from_content2(content):
    pattern = r'```python\s*(.*?)```'
    result = re.search(pattern, content, re.DOTALL)

    if result:
        extracted_code = result.group(1).strip()
    else:
        extracted_code = ""
        print("[WARNING] No Python code found.")
    return extracted_code