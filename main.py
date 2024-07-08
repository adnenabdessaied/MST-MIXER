from init_utils import (
    load_runner,
    load_dataset,
    set_random_seed,
    set_training_steps,
    initialize_from_env,
    set_log_file,
    copy_file_to_log
)
import os
import sys
import argparse
import pyhocon
import glog as log
import socket
import getpass

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from transformers import BartTokenizer

from custom_datasets.avsd import get_dataset as get_avsd_dataset
from custom_datasets.nextqa import get_dataset as get_nextqa_dataset


parser = argparse.ArgumentParser(description='Main script for MST-MIXER')
parser.add_argument(
    '--model',
    type=str,
    default='mst_mixer/mixer',
    help='model name to train or test')

parser.add_argument(
    '--mode',
    type=str,
    default='train',
    help='train, generate or debug'
    )

parser.add_argument(
    '--eval_dir',
    type=str,
    default='ckpt/avsd'
)

parser.add_argument(
    '--wandb_project',
    type=str,
    default='mst_mixer'
)

parser.add_argument(
    '--wandb_mode',
    type=str,
    default='offline',
    choices=['online', 'offline', 'disabled', 'run', 'dryrun']
)

parser.add_argument(
    '--tag',
    type=str,
    default='full_model',
    help="Tag to differentiate the models"
)

parser.add_argument(
    '--start_idx_gen',
    type=int,
    default=0,
    help="The start index for generation"
)

parser.add_argument(
    '--end_idx_gen',
    type=int,
    default=10,
    help="The end index for generation"
)

parser.add_argument(
    '--gen_subset_num',
    type=int,
    default=1,
    help="The index of the test split for generation"
)


parser.add_argument('--ssh', action='store_true',
                    help='whether or not we are executing command via ssh. '
                         'If set to True, we will not log.info anything to screen and only redirect them to log file')


def main(gpu, config, args):
    config['training'] = args.mode == 'train'
    config['debugging'] = args.mode == 'debug'
    config['generating'] = args.mode == 'generate'
    config['wandb_project'] = args.wandb_project
    config['wandb_mode'] = 'disabled'
    if config['training']:
        config['wandb_mode'] = args.wandb_mode

    # When generating, only use 1 GPU
    if config['generating']:
        assert config['num_gpus'] == 1, 'When generating, only use 1 GPU!'

    if config['parallel'] and config['dp_type'] != 'dp':
        config['rank'] = gpu
        dist.init_process_group(
            backend='nccl',
            # init_method='env://',
            world_size=config['num_gpus'],
            rank=gpu
        )
        config['display'] = gpu == 0
        torch.cuda.set_device(gpu)
    else:
        config['display'] = True
    if config['debugging'] or (config['parallel'] and config['dp_type'] != 'dp'):
        config['num_workers'] = 0

    # set logs
    if config['training']:
        log_file = os.path.join(config["log_dir"], f'{args.mode}.log')
        set_log_file(log_file, file_only=args.ssh)

    # print environment info
    if config['display']:
        log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
            socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
        log.info('Command line is: {}'.format(' '.join(sys.argv)))

        if config['parallel'] and config['dp_type'] != 'dp':
            log.info(f'World_size: {config["num_gpus"]}, cur rank: {config["rank"]}')
        log.info(f"Running experiment: {args.model}")
        log.info(f"Results saved to {config['log_dir']}")

    # initialization
    if config['display'] and config['training']:
        copy_file_to_log(config['log_dir'])
    set_random_seed(config['random_seed'])

    device = torch.device(f"cuda:{gpu}")
    if config["use_cpu"]:
        device = torch.device("cpu")
    config['device'] = device

    # prepare datasets (train and validation)
    dataset, dataset_eval = load_dataset(config)

    # set training steps
    if not config['generating'] or config['parallel']:
        config = set_training_steps(config, len(dataset))

    if config['display']:
        log.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    # load runner
    runner = load_runner(config, dataset.tokenizer, dataset.vocab_size)

    # parallel
    if config['parallel']:
        if config['dp_type'] == 'ddp':
            torch.cuda.set_device(gpu)
            runner.model = runner.model.to(gpu)
            runner.model = nn.parallel.DistributedDataParallel(
                runner.model,
                device_ids=[gpu],
                output_device=gpu,
                find_unused_parameters=True
            )
        else:
            raise ValueError(f'Unrecognized dp_type: {config["dp_type"]}')

    if config['training'] or config['debugging']:
        ckpt_path = config.get('start_path', None)
        runner.load_ckpt(ckpt_path=ckpt_path)
        runner.train(dataset, dataset_eval)

    elif config['generating']:
        if config['loads_start_path']:
            runner.load_ckpt(config['start_ckpt_for_generating'])
        else:
            runner.load_ckpt_best()
        assert args.gen_subset_num > 0
        # Load the data
        if config['task'] == 'avsd':
            # load the saved tokenizer
            tokenizer = BartTokenizer.from_pretrained(os.path.join(config['log_dir'], 'bart_tokenizer'))
            test_dataset, _ = get_avsd_dataset(config, 'test', tokenizer)
            assert args.start_idx_gen >= 0 and args.end_idx_gen <= len(test_dataset) and args.start_idx_gen < args.end_idx_gen
            test_dataset = test_dataset[args.start_idx_gen:args.end_idx_gen]
            runner.generate(
                test_dataset, args.tag, tokenizer, gen_subset_num=args.gen_subset_num
            )

        elif config['task'] == 'nextqa':
            # load the saved tokenizer
            tokenizer = BartTokenizer.from_pretrained(os.path.join(config['log_dir'], 'bart_tokenizer'))
            test_dataset, app_feats, mot_feats = get_nextqa_dataset(config, 'test')
            assert args.start_idx_gen >= 0 and args.end_idx_gen <= len(test_dataset) and args.start_idx_gen < args.end_idx_gen
            test_dataset = test_dataset[args.start_idx_gen:args.end_idx_gen]
            runner.generate(
                test_dataset, app_feats, mot_feats, args.tag, tokenizer, args.start_idx_gen, args.end_idx_gen, gen_subset_num=args.gen_subset_num
            )
        else:
            raise ValueError       

    if config['parallel']:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()

    # initialization
    model_type, model_name = args.model.split('/')
    config = initialize_from_env(model_name, args.mode, model_type, args.eval_dir, tag=args.tag)
    if config['num_gpus'] > 1:
        config['parallel'] = True
        mp.spawn(main, nprocs=config['num_gpus'], args=(config, args))
    else:
        config['parallel'] = False
        main(0, config, args)
