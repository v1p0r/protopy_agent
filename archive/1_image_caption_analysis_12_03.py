from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import json
from pathlib import Path
import argparse
import os
import subprocess
from utils import image_to_base64
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--model_name',type=str)
parser.add_argument('--key',type=str)
parser.add_argument('--input_path', type=str) 
parser.add_argument('--output_path',type=str)

args = parser.parse_args()

paper_name = args.paper_name
model_name = args.model_name
DASHSCOPE_API_KEY = args.key
input_path = args.input_path
output_path = args.output_path

# load the image pairs from preprocess.py
with open(output_path + '/' + 'image_pairs.json', "r", encoding="utf-8") as f:
    image_pairs = json.load(f)

payload = json.dumps(image_pairs, ensure_ascii=False, separators=(",", ":"))

# initialize the OpenAI competiable client for qwen3-vl-flash
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

def api_call(model_name, prompt):
    completion = client.chat.completions.create(
            model=model_name, 
            messages=prompt
    )
    return completion

filtering_prompt = [{'role': 'system', 'content': """
You are a strict figure classifier for a research paper.

You receive a JSON object mapping string IDs to:
{
    "0": {
        "image": "path or filename",
        "caption": "LaTeX figure environment including caption"
    },
    "1": {
        "image": "path or filename",
        "caption": "LaTeX figure environment including caption"
    },
    
}

Your job is to select ONLY figures that are:
- Identify only figures that depict **model architectures**, **pipelines**, or **system workflows**.
- Exclude all others, including metric/performance plots, ablations, task illustrations, examples, galleries, datasets, failure cases, and editor screenshots.
- For each selected figure, extract the **human-readable caption text** (remove LaTeX commands like \\begin, \\end, \\vspace, \\label, \\includegraphics, etc.).
- Return that caption in plain, readable English — no LaTeX, no comments.

You MUST EXCLUDE figures that are primarily:
- Metrics, curves, score plots, RL training curves
- Performance comparisons, ablations, Best@N, validity rates, reward curves
- Qualitative examples, galleries of results, synthesized samples
- Task/environment illustrations, dataset examples, failure cases, case studies
- Editor UI screenshots, block catalogs, low-level component lists without a full system pipeline

Use ONLY the provided `caption` text (and, if helpful, filename hints like 'RL_Metrics').

Do not include explanations, comments, or extra fields.
"""},


{'role': 'user', 'content': f"""

Below is a JSON object containing figure data from a research paper.

{payload}


Output format:
Return **strictly valid JSON** of this exact format:
```json
{{
    
    "id": "clear caption text",
    "4": "clear caption text",
    "8": "clear caption text"
    
  
}}
```


"""}
]

response = api_call(model_name, filtering_prompt)

response_json = json.loads(response.model_dump_json())

output_json = response_json["choices"][0]["message"]["content"]
filtered_image_pairs = json.loads(output_json.replace("```json", "").replace("```", "").strip())

for key, caption in filtered_image_pairs.items():
    filtered_image_pairs[key] = {}

    filename = os.path.join(input_path, image_pairs[key]["image"])
    if filename.lower().endswith('.pdf'):
        png_output_path = Path(os.path.join(output_path, os.path.basename(filename))).with_suffix('.png')
        subprocess.run(['bash','pdf2png.sh', filename, png_output_path])
    elif filename.lower().endswith('.png'):
        png_output_path = Path(os.path.join(output_path, os.path.basename(filename)))
        shutil.copy(filename, png_output_path)
        print(f"💾 Saved: {png_output_path}")
    else:
        png_output_path = Path(os.path.join(output_path, os.path.basename(filename)))
        shutil.copy(filename, png_output_path)
        print(f"❌ Unsupported file type saved as: {png_output_path}")

    filtered_image_pairs[key] = {"image": image_pairs[key]["image"], \
        "caption": caption, \
        "b64": image_to_base64(png_output_path)
        }

image_list = []
for key, image_pair in tqdm(filtered_image_pairs.items(),
                            desc="Analyzing figures",
                            unit="fig"):
    caption = image_pair["caption"]
    image_b64 = image_pair["b64"]
    extraction_prompt = [{'role': 'system', 'content': '''
    You are an expert vision-language model specialized in understanding figures from computer science and machine learning papers.

    Given:
    1. A caption describing a figure.
    2. The figure image itself.

    Your tasks:

    1. Classify the figure into exactly ONE of the following categories:
    - "model_architecture": diagrams that show layers, blocks, components of a model or network.
    - "pipeline": diagrams that show a sequence of processing steps or modules for data or experiments (ingest → preprocess → train → eval, etc.).
    - "system_workflow": diagrams that show interactions between multiple components, agents, services, or tools in a larger system (e.g., multi-agent system, platform, or end-to-end toolchain).
    - "other": if none of the above clearly apply.

    2. Extract structured information from the figure that would help RECONSTRUCT the model, pipeline, or system in code.
    Be as concrete and implementation-oriented as possible:
    - List key components/blocks/modules with concise names and roles.
    - List edges / data flows (source → target, what is passed).
    - List stages or steps in order, if applicable.
    - Include inputs and outputs (data types, files, models, agents, APIs, etc.).
    - Include any loops, conditions, or control flow if visible.
    - Prefer information grounded in the VISUAL content; use the caption only as supporting context.
    - Do NOT invent components that are not clearly implied by the figure.

    3. Output MUST be valid JSON only. No extra text.

    JSON schema (strict):

    {
    "category": "model_architecture" # "pipeline" or "system_workflow" or "other",
    "extracted": {
        "components": [
        {
            "name": "string",
            "type": "string",
            "role": "string"
        }
        ],
        "flows": [
        {
            "source": "string",
            "target": "string",
            "data": "string"
        }
        ],
        "stages": [
        {
            "name": "string",
            "description": "string"
        }
        ],
        "inputs": ["string"],
        "outputs": ["string"]
    }
    }
    '''},{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Analyze the following figure and respond in the required JSON schema."
            },
            {
                "type": "text",
                "text": f"Caption: {caption}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                }
            }
        ]
    }]
    response = api_call(model_name, extraction_prompt)
    output_json = json.loads(response.model_dump_json())
    content = output_json["choices"][0]["message"]["content"]
    image_list.append(content)

with open(f'{output_path}/image_extracted_information.json', 'w') as f:
        json.dump(image_list, f)

print(f"✅ Figure analysis completed for {paper_name}")