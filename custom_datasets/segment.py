from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import cv2
import os
import torch
import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--sam_ckpt',
        type=str,
        help='SAM checkpoint to be used'
    )

    parser.add_argument(
        '--avsd_root',
        type=str,
        help='Directory where the individual AVSD frames are located'
    )

    parser.add_argument(
        '--crop_root',
        type=str,
        help='Directory where the individual crops (objects) will be saved'
    )

    parser.add_argument(
        '--embed_root',
        type=str,
        help='Directory where the individual embeddings will be saved'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['segment', 'embed'],
        help='segment: segment the image into regions | embed: embed the image crops detected during segmentation'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index of the partition'
    )

    parser.add_argument(
        '--end',
        type=int,
        default=1968,
        help='End index of the partition'
    )

    args = parser.parse_args()
    return args


def partition_ids(avsd_ids, start, end):
    avsd_ids.sort()
    assert start < end
    assert start >= 0 and end <= len(avsd_ids)
    avsd_ids_partition = avsd_ids[start:end]
    return avsd_ids_partition


def get_middle_frames(avsd_ids_partition, avsd_root):
    pbar = tqdm(avsd_ids_partition)
    pbar.set_description('[INFO] Preparing frames of {} videos'.format(len(avsd_ids_partition)))
    path_list = []
    for avsd_id in pbar:
        frames = os.listdir(os.path.join(avsd_root, avsd_id))
        if 'test' in avsd_root:
            frames.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
        else:
            frames.sort(key=lambda f: int(f.split('-')[-1].split('.')[0]))
        middle_frame = frames[int(len(frames)/2)]
        middle_frame = os.path.join(avsd_root, avsd_id, middle_frame)
        path_list.append(middle_frame)
    return path_list


def segment_images(sam, path_list, crop_root):
    mask_generator = SamAutomaticMaskGenerator(sam)
    pbar = tqdm(path_list)
    pbar.set_description('Detecting Objects')
    for pth in pbar:
        vid_id = pth.split('/')[-2]
        crop_dir = os.path.join(crop_root, vid_id)
        if not os.path.isdir(crop_dir):
            os.makedirs(crop_dir)

        image = cv2.imread(pth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        masks.sort(key=lambda e: e['stability_score'], reverse=True)
        if len(masks) > 36:
            masks = masks[:36]
        for i, mask in enumerate(masks):
            crop = image[
                int(mask['bbox'][1]):int(mask['bbox'][1] + mask['bbox'][3] + 1),
                int(mask['bbox'][0]):int(mask['bbox'][0] + mask['bbox'][2] + 1),
                :
            ]
            crop_flipped = cv2.flip(crop, 1)  # Horizontal flip
            cv2.imwrite(os.path.join(crop_dir, f'obj_{i}.jpg'), crop)
            cv2.imwrite(os.path.join(crop_dir, f'obj_{i}_flipped.jpg'), crop_flipped)

    print('[INFO] Done...')


def embed_objects(sam, crop_ids, crop_root, embed_root):
    predictor = SamPredictor(sam)
    pbar = tqdm(crop_ids)
    pbar.set_description('Embedding Objects')
    for vid_id in pbar:
        embeds = []
        crop_dir = os.path.join(crop_root, vid_id)
        crop_paths = list(map(lambda p: os.path.join(crop_dir, p), os.listdir(crop_dir)))
        crop_paths = list(filter(lambda p: 'flipped' not in p, crop_paths))
        crop_paths.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
        for cp in crop_paths:
            crop = cv2.imread(cp)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            predictor.set_image(crop)
            embed_crop = predictor.get_image_embedding()
            embed_crop = embed_crop.mean(-1).mean(-1)
            
            crop_flipped = cv2.flip(crop, 1)
            predictor.set_image(crop_flipped)
            embed_crop_flipped = predictor.get_image_embedding()
            embed_crop_flipped = embed_crop_flipped.mean(-1).mean(-1)
            
            embed = torch.cat((embed_crop, embed_crop_flipped), dim=-1)
            # embed = embed.copy().cpu()
            embeds.append(embed)

        embeds = torch.cat(embeds, 0).cpu().numpy()
        np.save(os.path.join(embed_root, f'{vid_id}.npy'), embeds)

    print('[INFO] Done...')


def segment(args, sam):
    avsd_ids = os.listdir(args.avsd_root)
    avsd_ids.sort()
    avsd_ids_partition = partition_ids(avsd_ids, args.start, args.end)
    path_list = get_middle_frames(avsd_ids_partition, args.avsd_root)

    if not os.path.isdir(args.crop_root):
        os.makedirs(args.crop_root)
    segment_images(sam, path_list, args.crop_root)


def embed(args, sam):
    crop_ids = os.listdir(args.crop_root)
    crop_ids.sort()
    crop_ids_partition = partition_ids(crop_ids, args.start, args.end)
    if not os.path.isdir(args.embed_root):
        os.makedirs(args.embed_root)
    embed_objects(sam, crop_ids_partition, args.crop_root, args.embed_root)


if __name__ == '__main__':
    args = parse_args()
    sam = sam_model_registry['vit_h'](
        checkpoint=args.sam_ckpt)
    device = 'cuda'
    sam.to(device=device)

    assert args.mode in ['segment', 'embed']
    if args.mode == 'segment':
        segment(args, sam)
    else:
        embed(args, sam)
