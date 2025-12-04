import argparse
import json
from utils import *
import subprocess


parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--input_path', type=str) 
parser.add_argument('--output_path',type=str)

args    = parser.parse_args()

paper_name = args.paper_name
input_path = args.input_path
output_path = args.output_path

###########################################################
# Image analysis
###########################################################
output_dir = Path(output_path)
content_list_path = next(output_dir.glob("*/auto/*content_list.json"))
content_list = json.loads(content_list_path.read_text())

image_pairs = {}

index = 0
for i in content_list:
    if i['type'] == 'image':

        image_pairs[index] = {
            "image": str(Path(next(output_dir.glob("*/auto/"))).joinpath(i['img_path'])),
            "caption": " ".join(i['image_caption']).strip()
        }
        index += 1

with open(output_path + '/' + 'image_pairs.json', 'w',encoding="utf-8") as f:
    json.dump(image_pairs, f)


# read the latex fileso
# for index, latexfile in enumerate(extract_raw_figures(input_path)):
    
#     image_path_line = [line for line in latexfile['snippet'].splitlines() if "\\includegraphics" in line][0]
#     filename = re.findall(pattern, image_path_line)
#     image_pairs[index] = {"image": filename[-1], "caption": latexfile['snippet']}

# with open(output_path + '/' + 'image_pairs.json', 'w',encoding="utf-8") as f:
#     json.dump(image_pairs, f)


###########################################################
# Planning
###########################################################
# json_path = os.path.join(output_path, os.path.basename(output_path)+".tar.json")
# image_information_path = os.path.join(output_path, "image_extracted_information.json")

# with open(image_information_path, "r", encoding="utf-8") as f:
#     image_information = json.load(f)

# data = load_s2orc_json(json_path)

# latex_parse = data['latex_parse']
# body_text = latex_parse['body_text']
# back_matter = latex_parse['back_matter']

# paper = {'abstract':data['title']}
# paper.update(group_text_by_section(body_text))
# paper.update(group_text_by_section(back_matter))

# with open(os.path.join(output_path, 'extracted_paper.json'), 'w', encoding="utf-8") as f:
#     json.dump(paper, f, ensure_ascii=False)




print(f"✅ Preprocessing completed for {paper_name}")


