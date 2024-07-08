export MODEL=$1
export TAG=$2
export MODE=$3
export EVAL_DIR=$4
export DSTC=$5


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate mst_mixer

export CUDA_VISIBLE_DEVICES=0; python main.py --start_idx_gen 0000 --end_idx_gen 0285 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=0; python main.py --start_idx_gen 0285 --end_idx_gen 0570 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=0; python main.py --start_idx_gen 0570 --end_idx_gen 0855 --gen_subset_num 03 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=0; python main.py --start_idx_gen 0855 --end_idx_gen 1140 --gen_subset_num 04 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=1; python main.py --start_idx_gen 1140 --end_idx_gen 1425 --gen_subset_num 05 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main.py --start_idx_gen 1425 --end_idx_gen 1710 --gen_subset_num 06 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main.py --start_idx_gen 1710 --end_idx_gen 1995 --gen_subset_num 07 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main.py --start_idx_gen 1995 --end_idx_gen 2280 --gen_subset_num 08 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=2; python main.py --start_idx_gen 2280 --end_idx_gen 2565 --gen_subset_num 09 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main.py --start_idx_gen 2565 --end_idx_gen 2850 --gen_subset_num 10 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main.py --start_idx_gen 2850 --end_idx_gen 3135 --gen_subset_num 11 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main.py --start_idx_gen 3135 --end_idx_gen 3420 --gen_subset_num 12 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=3; python main.py --start_idx_gen 3420 --end_idx_gen 3705 --gen_subset_num 13 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=3; python main.py --start_idx_gen 3705 --end_idx_gen 3990 --gen_subset_num 14 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=3; python main.py --start_idx_gen 3990 --end_idx_gen 4275 --gen_subset_num 15 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=3; python main.py --start_idx_gen 4275 --end_idx_gen 4560 --gen_subset_num 16 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=4; python main.py --start_idx_gen 4560 --end_idx_gen 4845 --gen_subset_num 17 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main.py --start_idx_gen 4845 --end_idx_gen 5130 --gen_subset_num 18 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main.py --start_idx_gen 5130 --end_idx_gen 5415 --gen_subset_num 19 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main.py --start_idx_gen 5415 --end_idx_gen 5700 --gen_subset_num 20 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=5; python main.py --start_idx_gen 5700 --end_idx_gen 5985 --gen_subset_num 21 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=5; python main.py --start_idx_gen 5985 --end_idx_gen 6270 --gen_subset_num 22 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=5; python main.py --start_idx_gen 6270 --end_idx_gen 6555 --gen_subset_num 23 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=5; python main.py --start_idx_gen 6555 --end_idx_gen 6840 --gen_subset_num 24 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=6; python main.py --start_idx_gen 6840 --end_idx_gen 7125 --gen_subset_num 25 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=6; python main.py --start_idx_gen 7125 --end_idx_gen 7410 --gen_subset_num 26 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=6; python main.py --start_idx_gen 7410 --end_idx_gen 7695 --gen_subset_num 27 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=6; python main.py --start_idx_gen 7695 --end_idx_gen 7980 --gen_subset_num 28 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

export CUDA_VISIBLE_DEVICES=7; python main.py --start_idx_gen 7980 --end_idx_gen 8265 --gen_subset_num 29 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=7; python main.py --start_idx_gen 8265 --end_idx_gen 8550 --gen_subset_num 30 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=7; python main.py --start_idx_gen 8550 --end_idx_gen 8835 --gen_subset_num 31 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=7; python main.py --start_idx_gen 8835 --end_idx_gen 9178 --gen_subset_num 32 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

wait

python merge_pred_nextqa.py
