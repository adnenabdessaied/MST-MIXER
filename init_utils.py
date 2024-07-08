import os
import torch
import random
import pyhocon
import datetime
import json
import subprocess
import itertools
import glob
import glog as log
import sys
import re
from os import path as osp
import numpy as np

from custom_datasets.avsd import AVSDDataset
from custom_datasets.nextqa import NextQADataset

from runners.runner_avsd import AVSDRunner
from runners.runner_nextqa import NEXTQARunner


def load_runner(config, tokenizer, vocab_size):
    if config['task'] == 'avsd':
        return AVSDRunner(config, tokenizer, vocab_size)
    elif config['task'] == 'nextqa':
        return NEXTQARunner(config, tokenizer, vocab_size)
    else:
        raise ValueError


def load_dataset(config):
    if config['task'] == 'avsd':
        dataset = AVSDDataset(config, 'train')
        dataset_eval = AVSDDataset(config, 'val')
    elif config['task'] == 'nextqa':
        dataset = NextQADataset(config, 'train')
        dataset_eval = NextQADataset(config, 'val')
    else:
        raise ValueError
    return dataset, dataset_eval


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def copy_file_to_log(log_dir):
    dirs_to_cp = ['.', 'config', 'datasets', 'runners', 'models']
    files_to_cp = ['*.py', '*.json', '*.sh', '*.conf']
    for dir_name in dirs_to_cp:
        dir_name = osp.join(log_dir, 'code', dir_name)
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
    for dir_name, file_name in itertools.product(dirs_to_cp, files_to_cp):
        filename = osp.join(dir_name, file_name)
        if len(glob.glob(filename)) > 0:
            os.system(f'cp {filename} {osp.join(log_dir, "code", dir_name)}')
    log.info(f'Files copied to {osp.join(log_dir, "code")}')


def set_log_file(fname, file_only=False):
    # if fname already exists, find all log file under log dir,
    # and name the current log file with a new number
    if osp.exists(fname):
        prefix, suffix = osp.splitext(fname)
        log_files = glob.glob(prefix + '*' + suffix)
        count = 0
        for log_file in log_files:
            num = re.search(r'(\d+)', log_file)
            if num is not None:
                num = int(num.group(0))
                count = max(num, count)
        fname = fname.replace(suffix, str(count + 1) + suffix)
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = f = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def initialize_from_env(model, mode, model_type, eval_dir, tag=''):
    if "GPU" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['GPU']
    if mode in ['train', 'debug']:
        config = pyhocon.ConfigFactory.parse_file(f"config/{model_type}.conf")[model]
    else:
        path_config = os.path.join(eval_dir, 'code', f"config/{model_type}.conf")
        config = pyhocon.ConfigFactory.parse_file(path_config)[model]
        config['log_dir'] = eval_dir
        
    config['model_type'] = model_type
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        config['num_gpus'] = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        # multi-gpu setting
        if config['num_gpus'] > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ["MASTER_PORT"] = str(config['master_port'])

    if mode == 'debug':
        model += '_debug'

    if tag:
        model += '-' + tag
    if mode != 'generate':
        config["log_dir"] = os.path.join(config["log_dir"], model)
        if not os.path.exists(config["log_dir"]):
            os.makedirs(config["log_dir"])

    config['timestamp'] = datetime.datetime.now().strftime('%m%d-%H%M%S')

    # Choose the correct config file and add the BART json file to config
    if mode in ['train', 'debug']:
        config['bart_config'] = config['{}_bart_{}_config'.format(
            config['task'], config['bart_size'])]
    else:
        config['bart_config'] = os.path.join(eval_dir, 'code', 'config/{}_bart_{}.json'.format(
            config['task'], config['bart_size']))

    config['bart_config_json'] = json.load(open(config['bart_config'], 'r'))

    config['overfit'] = config['overfit_size'] > 0
    return config


def set_training_steps(config, num_samples):
    if config['parallel'] and config['dp_type'] == 'dp':
        config['num_iter_per_epoch'] = int(np.ceil(num_samples / config['batch_size']))
    else:
        config['num_iter_per_epoch'] = int(np.ceil(num_samples / (config['batch_size'] * config['num_gpus'])))
    if 'train_steps' not in config:
        config['train_steps'] = config['num_iter_per_epoch'] * config['num_epochs']
    if 'warmup_steps' not in config:
        config['warmup_steps'] = int(config['train_steps'] * config['warmup_ratio'])
    return config
