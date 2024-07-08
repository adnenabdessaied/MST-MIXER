import os
import json
import argparse

parser = argparse.ArgumentParser(description='Main script for MST-MIXER')
parser.add_argument(
    '--dstc',
    type=int,
    default=8,
    choices=[7, 8, 10],
    help='DSTC challenge identifier')

args = parser.parse_args()

assert args.dstc in [7, 8, 10]
if args.dstc == 7:
    output_dir = 'output/dstc7'
    raw_data_path = 'raw_data/test_set4DSTC7-AVSD.json'

elif args.dstc == 8:
    output_dir = 'output/dstc8'
    raw_data_path = 'raw_data/test_set4DSTC8-AVSD.json'
else:
    output_dir = 'output/dstc10'
    raw_data_path = 'raw_data/test_set4DSTC10-AVSD.json'

with open(raw_data_path, 'r') as f:
    raw_dialogs = json.load(f)['dialogs']

file_paths = os.listdir(output_dir)
file_paths = list(filter(lambda f: 'part' in f , file_paths))
name = file_paths[0]
file_paths = list(map(lambda f: os.path.join(output_dir, f), file_paths))

dialogs = {}
for pth in file_paths:
    with open(pth, 'r') as f:
        data = json.load(f)
    
    for dialog in data['dialogs']:
        vid_id = dialog['image_id']
        dialogs[vid_id] = dialog
    # dialogs.extend(data['dialogs'])
    os.remove(pth)

# Now, re-establish the original order of the dialogs
res = []
for dialog in raw_dialogs:
    vid_id = dialog['image_id']
    res.append(dialogs[vid_id])

res = {
    'dialogs': res
}

name = "".join(name.split('-')[:-1]) + '.json'
output_path = os.path.join(output_dir, name)
with open(output_path, 'w') as f:
    json.dump(res, f, indent=4)

print('[INFO] Files merged and saved in {}'.format(output_path))
