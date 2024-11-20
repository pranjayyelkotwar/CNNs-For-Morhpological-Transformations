import os
import numpy as np
import json
from scipy.ndimage import binary_dilation, binary_erosion

def generate_inp_out_catA_Simple(list_se_idx, **param):
    """
    Generates input and output image pairs for a given structuring element sequence.
    """
    base_img = np.zeros((param['img_size'], param['img_size']), dtype=np.int32)
    sz = np.random.randint(3, 6)
    idx1 = np.random.randint(0, param['img_size'], size=sz)
    idx2 = np.random.randint(0, param['img_size'], size=sz)
    base_img[idx1, idx2] = 1

    # Apply initial random dilations to give some structure to the base image
    for _ in range(2):
        idx = np.random.randint(0, 8)
        base_img = binary_dilation(base_img, param['list_se_3x3'][idx])

    inp_img = np.array(base_img, copy=True)
    out_img = np.array(base_img, copy=True)

    # Apply the specified sequence of dilations and erosions
    for idx in range(len(list_se_idx)):
        out_img = binary_dilation(out_img, param['list_se_3x3'][list_se_idx[idx]])
    for idx in range(len(list_se_idx)):
        out_img = binary_erosion(out_img, param['list_se_3x3'][list_se_idx[idx]])

    return inp_img, out_img

def generate_one_task_CatA_Simple(**param):
    """
    Generates one task with input and output images along with the SE sequence.
    """
    list_se_idx = np.random.randint(0, 8, param['seq_length'])
    data = []
    k = 0
    while k < param['no_examples_per_task']:
        inp_img, out_img = generate_inp_out_catA_Simple(list_se_idx, **param)

        # Check if both input and output images are non-trivial
        FLAG = np.all(inp_img == 1) or np.all(inp_img == 0) or np.all(out_img == 1) or np.all(out_img == 0)
        
        if FLAG:
            # Regenerate SE indices if trivial
            list_se_idx = np.random.randint(0, 8, param['seq_length'])
            data = []
            k = -1
        else:
            # Add valid input-output pair to data
            data.append((inp_img, out_img))
        k += 1

    return data, list_se_idx

def write_combined_json_CatA_Simple(data, list_se_idx, task_no, dataset):
    """
    Appends task data in specified format to dataset list for final JSON output.
    """
    sequence = [f"Dilation SE{idx+1}" for idx in list_se_idx] + [f"Erosion SE{idx+1}" for idx in list_se_idx]
    for inp_img, out_img in data:
        inp = [[int(y) for y in x] for x in inp_img]
        out = [[int(y) for y in x] for x in out_img]
        dataset.append({"input": inp, "output": out, "sequence": sequence})

def generate_1000_tasks_CatA_Simple(seed, **param):
    """
    Generates 1000 tasks and stores them in a single JSON file.
    """
    np.random.seed(seed)
    os.makedirs("./Dataset", exist_ok=True)
    dataset = []
    for task_no in range(1000):
        data, list_se_idx = generate_one_task_CatA_Simple(**param)
        write_combined_json_CatA_Simple(data, list_se_idx, task_no, dataset)

    # Write the combined dataset to a single JSON file
    with open('./Dataset/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    param = {
        'img_size': 10,
        'se_size': 5,
        'seq_length': 4,
        'no_examples_per_task': 10,
        'no_colors': 3,
        'list_se_3x3': [np.ones((3, 3), dtype=bool) for _ in range(8)]  # Define example SEs
    }

    generate_1000_tasks_CatA_Simple(32, **param)