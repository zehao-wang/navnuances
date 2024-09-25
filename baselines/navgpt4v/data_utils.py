import os
import json
import numpy as np

def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        filepath = os.path.join(anno_dir, f'R2R_{split}.json')
        with open(filepath) as f:
            new_data = json.load(f)

        data += new_data

    return data

def construct_instrs(anno_dir, dataset, splits):
    data = []
    if "instr" in splits[0]:
        return load_instr_datasets(anno_dir, dataset, splits)

    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        # Split multiple instructions into separate entries 
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            del new_item['instructions']
            data.append(new_item)
    return data