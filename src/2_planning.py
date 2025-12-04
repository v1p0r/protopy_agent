import json
from transformers import AutoTokenizer
from openai import OpenAI
import os
from pathlib import Path
from collections import defaultdict
from utils import group_text_by_section
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--paper_name',type=str)
parser.add_argument('--input_path', type=str) 
parser.add_argument('--output_path',type=str)
parser.add_argument('--model_name',type=str)
parser.add_argument('--base_url',type=str)
args    = parser.parse_args()

paper_name = args.paper_name
input_path = args.input_path
output_path = args.output_path
model_name = args.model_name
base_url = args.base_url

image_information_path = os.path.join(output_path, "image_extracted_information.json")
paper_dir = Path(output_path) 
paper_md = next(paper_dir.glob("*/auto/*.md"), None)

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=base_url,
)

def api_call(model_name, prompt):
    completion = client.chat.completions.create(
            model=model_name, 
            messages=prompt
    )
    return completion

with open(f'{paper_md}') as f:
        processed_paper = f.read()

with open(image_information_path, "r", encoding="utf-8") as f:
    image_information = json.load(f)

# prompts 

############################# planning_prompt #################################
planning_prompt = [
    {'role': 'system', 'content': """
You are an expert in AI research with a strong understanding of the developing experiments and reproducing scientific papers.
You will receive a json format text which is obtained from a research paper.
Your task is to create a plan to reproduce the code base described in the paper following the details provided in the paper from methodology, experiments, and evaluations precisely.

Instructions:

1. The plan should precisely follow all the details provided in the paper from methodology, experiments, and evaluations.
2. Produce a Clear and Structured Plan: Generate the reproduction plan in a clean, logically organized format. Break all tasks into explicit, step-by-step actions that can be directly executed without interpretation.
3. Create the plan to be implementation-ready and concise. Eliminate unnecessary verbosity while ensuring every step remains fully faithful to the experimental procedures described in the paper.
4. If you believe the information provided in the additional information is conflicted with the paper, you should follow the paper.
"""},

    {'role':'user',
'content':f"""
## Paper Content
{processed_paper}

## Task
1. Your goal is to reproduce the method reproduce the code base strictly following the description from the paper.
2. The official code for this paper is unavailable, so you must infer and design the implementation from the paper alone.
3. Extracts the core elements of the Methodology.
4. Specifies all Experimental requirements, including datasets, preprocessing, experimental settings, hyperparameters, and evaluation metrics.
5. Make the plan highly detailed, technically precise, and directly actionable for later implementation.

## Instructions
1. Only the strategy and plan is required, no code is needed.
2. If you are uncertain a certain part of the implementation is not described in the paper, specify the part as "".
2. If any aspect of the implementation is uncertain, explicitly state the ambiguity instead of making assumptions.
"""}]

############################# file_structure_prompt #################################

file_structure_prompt = [
    {'role': 'user', 'content': f"""You are an expert in AI research with a strong understanding of the developing experiments and reproducing scientific papers.
You will receive a json format text which is obtained from a research paper and possibly another json format with a list of additional information each one of them is either model architecture, pipeline, or system workflow.
Based on the plan for reproducing the code base, you will generate a file structure for the code base. Make sure the design is complete and can be implemented.
Avoid unnecessary complexity, and use of public available libraries.

## Additional Information 
{image_information}

-----

## output format 
[EXAMPLE]
{{
    "file_list": [
        'main.py',
        'model.py',
        'train.py',
        'eval.py',
        'data_loader.py',
        'misc.py'
    ],
    'plan': 'Step by step plan for reproducing the code base.....',
    'Code structure': [
    {{
      "name": "data_loader",
      "classes": [
        {{
          "name": "DatasetLoader",
          "methods": [
            {{
              "name": "__init__",
              "args": ["config: dict"],
              "returns": "None"
            }},
            {{
              "name": "load_train_val_test",
              "args": [],
              "returns": "(Dataset, Dataset, Dataset)"
            }}
          ]
        }}
      ]
    }},
    {{
      "name": "model",
      "classes": [
        {{
          "name": "Model",
          "base_class": "nn.Module",
          "methods": [
            {{
              "name": "__init__",
              "args": ["config: dict"],
              "returns": "None"
            }},
            {{
              "name": "forward",
              "args": ["x: Tensor"],
              "returns": "Tensor"
            }}
          ]
        }}
      ]
    }}
  ],
  "relationships": [
    {{"from": "Trainer", "to": "Model", "type": "uses"}},
    {{"from": "Main", "to": "DatasetLoader", "type": "instantiates"}}
  ],
    'pipeline or workflow': 'The exact fields available in the parsed paper (e.g., how methods, datasets, and hyperparameters are structured) are not specified and may depend on an upstream pipeline. Hardware assumptions (GPU type, number of devices, distributed vs. single-node training) are not defined. It is also unclear how to handle missing hyperparameters from the paper (e.g., unspecified batch size, learning rate schedule, or optimizer details), whether we should use defaults, grid search, or prompt the user for manual input.'
}}
[/EXAMPLE]

## json keys ["key_name" : <key value type> # "instruction"]
* "file_list": <List[str]> list of file names in *python list* format
* "plan": <str> step by step plan for reproducing the code base, example: "Step 1: Implement the DatasetLoader class. Step 2: Implement the Model class. Step 3: Implement the Trainer class......."
* "Code structure": <List[dict]> code structure of the code base, details about the classes and methods that need to be implemented and tells the relationship between the classes and methods.
* "pipeline or workflow": <str> pipeline or workflow of the code base

## requirement
Output should follow the format strictly with in the [EXAMPLE][/EXAMPLE] and no other text.

## Action
Follow the instruction for *json keys* strictly.


"""}
]   

############################# task_list_prompt #################################
task_list_msg = [
        {'role': 'user', 'content': """Your goal is break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies. 
You will break down tasks, analyze dependencies.
             
You outline a clear PRD/technical design for reproducing the paper’s method and experiments. 

Now, let's break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist in writing the code needed to reproduce the paper.

-----

## Format Example
[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0"  
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "data_preprocessing.py",
            "DataPreprocessing class ........"
        ],
        [
            "trainer.py",
            "Trainer ....... "
        ],
        [
            "dataset_loader.py",
            "Handles loading and ........"
        ],
        [
            "model.py",
            "Defines the model ......."
        ],
        [
            "evaluation.py",
            "Evaluation class ........ "
        ],
        [
            "main.py",
            "Entry point  ......."
        ]
    ],
    "Task list": [
        "dataset_loader.py", 
        "model.py",  
        "trainer.py", 
        "evaluation.py",
        "main.py"  
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both data_preprocessing.py and trainer.py share ........",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}

[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format.(e.g., 'numpy==1.21.0').
- Required Other language third-party packages: typing.List[str]  # List down packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible.
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list.
- Full API spec: <class 'str'>  # Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end and back-end communication is not required, leave it blank.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common utility functions or configuration variables.
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions or clarifications needed from the paper or project scope.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example."""}]

# config
config_msg = [
        {'role': 'user', 'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, plan, design specified previously, follow the "Format Example" and generate the code. 
Extract the training details from the above paper (e.g., learning rate, batch size, epochs, etc.), follow the "Format example" and generate the code. 
DO NOT FABRICATE DETAILS — only use what the paper provides.

You must write `config.yaml`.

ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Your output format must follow the example below exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: ...
  batch_size: ...
  epochs: ...
...
```

-----

## Code: config.yaml
"""
    }]

responses = []
trajectories = []
total_accumulated_cost = 0

for idx, instruction_msg in enumerate([planning_prompt, file_structure_prompt, task_list_msg, config_msg]):
    current_stage = ""
    if idx == 0 :
        current_stage = f"[Planning] Overall plan"
    elif idx == 1:
        current_stage = f"[Planning] Architecture design"
    elif idx == 2:
        current_stage = f"[Planning] Logic design"
    elif idx == 3:
        current_stage = f"[Planning] Configuration file generation"
    print(current_stage)

    trajectories.extend(instruction_msg)

    completion = api_call(model_name, trajectories)
    
    # response
    completion_json = json.loads(completion.model_dump_json())
    responses.append(completion_json)

    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})


with open(f'{output_path}/planning_response.json', 'w') as f:
    json.dump(responses, f)

with open(f'{output_path}/planning_trajectories.json', 'w') as f:
    json.dump(trajectories, f)


print(f"✅ Planning completed for {paper_name}")