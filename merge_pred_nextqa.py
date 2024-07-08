import os
import json
import argparse

parser = argparse.ArgumentParser(description='Main script for MST-MIXER')

args = parser.parse_args()

output_dir = 'output/nextqa'

file_paths = os.listdir(output_dir)
file_paths = list(filter(lambda f: 'part' in f , file_paths))
name = file_paths[0]
file_paths = list(map(lambda f: os.path.join(output_dir, f), file_paths))

results = {}
for pth in file_paths:
    with open(pth, 'r') as f:
        data = json.load(f)
    for video_id in data:
        if video_id not in results:
            results[video_id] = data[video_id]
        else:
            for qid in data[video_id]:
                if qid not in results[video_id]:
                    results[video_id][qid] = data[video_id][qid]
    os.remove(pth)

name = "".join(name.split('-')[:-1]) + '.json'
output_path = os.path.join(output_dir, name)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print('[INFO] Files merged and saved in {}'.format(output_path))
