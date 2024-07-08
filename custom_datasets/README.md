1. Download the raw [Charades train/val](https://prior.allenai.org/projects/charades) data
2. Download the raw [Charades test](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_vu17_test_480.tar) data
3. Install [SAM](https://github.com/facebookresearch/segment-anything.git)
4. Segment the frames
   ```shell
   python segement.py --sam_ckpt path_to_sam_ckpt --avsd_root path_to_charades_trval_frames --crop_root path_to_save_the_trval_crops  --mode segment --start start_idx --end end_idx
   python segement.py --sam_ckpt path_to_sam_ckpt --avsd_root path_to_charades_test_frames --crop_root path_to_save_the_test_crops  --mode segment --start start_idx --end end_id
   ```
5. Embed the crops
   ```shell
   python segement.py --sam_ckpt path_to_sam_ckpt --crop_root path_to_save_the_trval_crops  --mode emebed --embed_root ../features/sam  --start start_idx --end end_idx
   python segement.py --sam_ckpt path_to_sam_ckpt --crop_root path_to_save_the_test_crops  --mode emebed --embed_root ../features/sam_testset  --start start_idx --end end_idx

   ```
6. Preprocess and log the data
   ```shell
   python dataset.py --split train
   python dataset.py --split val
   
   ```
