import wandb
import os
import os.path as osp
import json
from collections import deque, OrderedDict
import time
import re
import shutil
import glob
import pickle
import gc
import numpy as np
import glog as log

import torch
import torch.utils.data as tud
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_value_


class Runner:
    def __init__(self, config):
        self.config = config
        if 'rank' in config:
            self.gpu_rank = config['rank']
        else:
            self.gpu_rank = 0

        self.epoch_idx = 0
        self.min_gen_val_loss = float('inf')
        self.best_epoch_idx = 0

        if self.config["max_ckpt_to_keep"] > 0:
            self.checkpoint_queue = deque([], maxlen=config["max_ckpt_to_keep"])
            self.metrics_queue = deque([], maxlen=config["max_ckpt_to_keep"])

        self.setup_wandb()

    def setup_wandb(self):
        if self.gpu_rank == 0:
            print("[INFO] Set wandb logging on rank {}".format(0))
            run = wandb.init(
                project=self.config['wandb_project'], config=self.config, mode=self.config['wandb_mode'])
        else:
            run = None
        self.run = run

    def forward(self, batch, eval=False):
        return NotImplementedError

    def train(self, dataset, dataset_eval):
        batch_size = self.config['batch_size']
        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            sampler = tud.distributed.DistributedSampler(
                        dataset,
                        num_replicas=self.config['num_gpus'],
                        rank=self.gpu_rank
                    )
        else:
            sampler = None

        data_loader = tud.DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=self.config['training'] and not self.config['parallel'],
                    collate_fn=dataset.collate_fn,
                    num_workers=self.config['num_workers'],
                    sampler=sampler
                )

        start_epoch_idx = self.epoch_idx
        num_iter_epoch = self.config['num_iter_per_epoch']
        if self.config['display']:
            log.info(f'{num_iter_epoch} iter per epoch.')

        num_epochs = self.config['num_epochs']

        # Perform validation before training
        if self.config['eval_first']:
            _ = self.val(dataset_eval)

        for epoch_idx in range(start_epoch_idx, num_epochs):
            if self.config['parallel'] and self.config['dp_type'] != 'dp':
                sampler.set_epoch(epoch_idx)
            self.epoch_idx = epoch_idx

            if self.config['display']:
                log.info(f'starting epoch {epoch_idx}')
                log.info('training')

            self.model.train()

            num_batch = 0
            next_logging_pct = .1
            start_time = time.time()
            self.optimizer.zero_grad()

            for batch in data_loader:
                num_batch += 1
                pct = num_batch / num_iter_epoch * 100
                iter_now = num_iter_epoch * epoch_idx + num_batch

                output = self.forward(batch)
                
                losses = output['losses']

                # optimizer step
                losses['tot_loss'] /= self.config['batch_multiply']
                # debug
                if self.config['debugging']:
                    log.info('try backward')

                losses['tot_loss'].backward()
                if self.config['clip_grad_value'] > 0:
                    clip_grad_value_(self.model.parameters(), self.config['clip_grad_value'])
                if self.config['debugging']:
                    log.info('backward done')

                if iter_now % self.config['batch_multiply'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                self.scheduler.step()

                # display and eval
                if pct >= next_logging_pct:
                    if self.config['display']:
                        loss_to_print = ''
                        for key in losses:
                            if losses[key] is not None and isinstance(losses[key], torch.Tensor):
                                loss_to_print += f'[{key}: {losses[key].item():.4f}]'
                        print(
                            f'[{int(pct)}%][Epoch: {epoch_idx + 1}/{num_epochs}][Iter : {num_batch}/{len(data_loader)}] [time: {time.time() - start_time:.2f}] {loss_to_print}'
                        )
                    if self.config['print_output']:
                        print(10 * '-' + 'responses' + 10 * '-')
                        print(output['reponses'])
                        print(10 * '-' + 'gt' + 10 * '-')
                        print(output['gt'])

                    next_logging_pct += self.config["next_logging_pct"]

                    if self.config['debugging']:
                        break

                lr_bart, lr_other = self.scheduler.get_lr()[0], self.scheduler.get_lr()[-1]

                elbo_global_key = 'elbo_loss_global (x{})'.format(self.config['elbo_global_coeff'])
                elbo_local_key = 'elbo_loss_local (x{})'.format(self.config['elbo_local_coeff'])
                gen_key = 'gen_loss (x{})'.format(self.config['gen_coeff'])
                if self.run:
                    self.run.log(
                        {
                            f"Train/{gen_key}": losses[gen_key].item(),
                            f"Train/{elbo_global_key}": losses[elbo_global_key].item(),
                            f"Train/{elbo_local_key}": losses[elbo_local_key].item(),
                            "Train/total_loss": losses['tot_loss'].item(),
                        },
                        step=iter_now
                    )

                    self.run.log(
                        {"Train/lr_bart": lr_bart, "Train/lr_other": lr_other},
                        step=iter_now
                    )
                del losses
                del output

            if self.config['display']:
                log.info(
                    f'100%,\ttime:\t{time.time() - start_time:.2f}'
                )
                if not self.config['overfit'] and self.run:
                    self.save_ckpt()
            
            if not self.config['skip_eval']:

                iter_now = num_iter_epoch * (epoch_idx + 1)
                val_losses = self.val(dataset_eval)

                if self.config['display']:
                    log.info('#'*100)
                    for k in val_losses:
                        log.info('Average val {} (epoch {}) = {}'.format(k, self.epoch_idx, val_losses[k]))
                    log.info('#'*100)

                gen_val_loss = val_losses[gen_key]

                if gen_val_loss < self.min_gen_val_loss:
                    self.min_gen_val_loss = gen_val_loss
                    self.best_epoch_idx = epoch_idx
                    # Log the best model w.r.t. the validation data
                    if self.run and self.config['save_ckpt']:
                        self.save_ckpt_best()

                if self.run:

                    self.run.log(
                        {
                            f"Val/{gen_key}": val_losses[gen_key],
                            f"Val/{elbo_global_key}": val_losses[elbo_global_key],
                            f"Val/{elbo_local_key}": val_losses[elbo_local_key],
                            "Val/total_loss": val_losses['tot_loss'],
                            "Val/min_gen_loss": self.min_gen_val_loss
                        },
                        step=iter_now
                    )

            if self.config['parallel']:
                if self.config['dp_type'] == 'dp':
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    dist.barrier()
                    torch.cuda.empty_cache()

            if self.config['stop_epochs'] >= 0 and epoch_idx + 1 >= self.config['stop_epochs']:
                if self.config['display']:
                    log.info('Stop for reaching stop_epochs.')
                break
        if self.config['display']:
            log.info(f'Best validation loss was reached at epoch {self.best_epoch_idx}.')

    def val(self, dataset):
        total_loss_val = 0.0
        total_gen_loss_val = 0.0
        total_elbo_global_val = 0.0
        total_elbo_local_val = 0.0
        num_batch_val = 0
        next_logging_pct_val = 0.05

        elbo_global_key = 'elbo_loss_global (x{})'.format(self.config['elbo_global_coeff'])
        elbo_local_key = 'elbo_loss_local (x{})'.format(self.config['elbo_local_coeff'])
        gen_key = 'gen_loss (x{})'.format(self.config['gen_coeff'])

        # Prepare the dataloader
        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            sampler_val = tud.distributed.DistributedSampler(
                dataset,
                num_replicas=self.config['num_gpus'],
                rank=self.gpu_rank
            )
            sampler_val.set_epoch(self.epoch_idx)
        else:
            sampler_val = None
        
        data_loader_val = tud.DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.config['num_workers'],
            sampler=sampler_val
        )

        if self.config['parallel'] and self.config['dp_type'] == 'dp':
            num_iter_per_epoch_val = int(np.ceil(len(dataset) / self.config['batch_size']))
        else:
            num_iter_per_epoch_val = int(np.ceil(len(dataset) / (self.config['batch_size'] * self.config['num_gpus'])))


        self.model.eval()

        if self.gpu_rank == 0:
            start_time = time.time()
        
        for batch in data_loader_val:
            num_batch_val += 1

            pct = num_batch_val / num_iter_per_epoch_val * 100

            with torch.no_grad():
                output = self.forward(batch)

            losses = output['losses']

            losses['tot_loss'] /= self.config['batch_multiply']
            losses[elbo_global_key] /= self.config['batch_multiply']
            losses[elbo_local_key] /= self.config['batch_multiply']
            losses[gen_key] /= self.config['batch_multiply']

            total_loss_val += losses['tot_loss'].item()
            total_gen_loss_val += losses[gen_key].item()
            total_elbo_global_val += losses[elbo_global_key].item()
            total_elbo_local_val += losses[elbo_local_key].item()

            # display and eval
            if pct >= next_logging_pct_val:
                if self.config['display']:
                    loss_to_print = ''
                    for key in losses:
                        if losses[key] is not None and isinstance(losses[key], torch.Tensor):
                            loss_to_print += f'[{key}: {losses[key].item():.4f}]'
                    print(
                        f'[{int(pct)}%][Validating][Iter : {num_batch_val}/{num_iter_per_epoch_val}] [time: {time.time() - start_time:.2f}] {loss_to_print}'
                    )

                next_logging_pct_val += self.config["next_logging_pct"]
        loss_val = total_loss_val / num_batch_val
        gen_loss_val = total_gen_loss_val / num_batch_val
        elbo_global_val = total_elbo_global_val / num_batch_val
        elbo_local_val = total_elbo_local_val / num_batch_val

        losses_val = {
            'tot_loss': loss_val,
            elbo_global_key: elbo_global_val,
            elbo_local_key: elbo_local_val,
            gen_key: gen_loss_val
        }
        self.model.train()
        return losses_val


    def save_eval_results(self, split, epoch_idx, metrics_results):

        metrics_filename = osp.join(self.config['log_dir'], f'metrics_epoch_{epoch_idx}.json')
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_results, f)
        log.info(f'Results of metrics saved to {metrics_filename}')

        if self.config["max_ckpt_to_keep"] > 0:
            if len(self.metrics_queue) == self.metrics_queue.maxlen:
                todel = self.metrics_queue.popleft()
                os.remove(todel)
            self.metrics_queue.append(metrics_filename)

        if epoch_idx == 'best':
            self.copy_best_predictions(split)

    def copy_best_results(self, split, epoch_idx):
        to_print = 'Copy '

        if not self.config['skip_saving_ckpt']:
            ckpt_path = osp.join(self.config['log_dir'], f'epoch_{epoch_idx}.ckpt')
            best_ckpt_path = ckpt_path.replace(f'{epoch_idx}.ckpt', 'best.ckpt')
            shutil.copyfile(ckpt_path, best_ckpt_path)
            to_print += best_ckpt_path + ' '

        metrics_filename = osp.join(self.config['log_dir'], f'metrics_epoch_{epoch_idx}.json')
        best_metric_filename = metrics_filename.replace(f'{epoch_idx}.json', 'best.json')
        shutil.copyfile(metrics_filename, best_metric_filename)
        to_print += best_metric_filename + ' '

        log.info(to_print)


    def set_ckpt(self, ckpt_dict):
        if self.config['parallel']:
            model = self.model.module
        else:
            model = self.model
        
        model_state_dict = model.state_dict()
      
        former_dict = {k: v for k, v in ckpt_dict['model_state_dict'].items() if k in model_state_dict}

        if self.config['display']:
            log.info("number of keys transferred: %d" % len(former_dict))
        assert len(former_dict.keys()) > 0

        model_state_dict.update(former_dict)

        model.load_state_dict(model_state_dict)
        if self.config['display']:
            log.info('loaded model')
        del model_state_dict, former_dict

        # if not self.config['uses_new_optimizer']:
        if not self.config['generating'] and not (self.config['uses_new_optimizer'] or self.config['sets_new_lr']):
            if not self.config['restarts']:
                self.epoch_idx = ckpt_dict['epoch_idx'] + 1

            if not self.config['resets_min_val_loss']:
                self.min_gen_val_loss = ckpt_dict['min_gen_val_loss']

            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            if self.config['display']:
                log.info('loaded optimizer')
            if 'scheduler' in ckpt_dict:
                self.scheduler.last_epcoh = ckpt_dict['epoch_idx'] * self.config['num_iter_per_epoch']
                self.scheduler.load_state_dict(ckpt_dict['scheduler'])

        del ckpt_dict

        torch.cuda.empty_cache()


    def save_ckpt(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_{self.epoch_idx}.ckpt'
        log.info(f'saving checkpoint {ckpt_path}')
        ckpt = self.get_ckpt()
        if self.config['skip_saving_ckpt']:
            return ckpt_path
        torch_version = float(torch.__version__[:3])
        if torch_version - 1.4 > 1e-3:
            torch.save(ckpt, f=ckpt_path, _use_new_zipfile_serialization=False)
        else:
            torch.save(ckpt, f=ckpt_path)
        del ckpt
        if not self.config['parallel']:
            torch.cuda.empty_cache()

        if self.config["max_ckpt_to_keep"] > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                os.remove(todel)
            self.checkpoint_queue.append(ckpt_path)

    def save_ckpt_best(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
        log.info(f'saving checkpoint {ckpt_path}')
        ckpt = self.get_ckpt()
        torch.save(ckpt, f=ckpt_path)
        del ckpt
        return ckpt_path

    def load_ckpt_best(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
        if not osp.exists(ckpt_path):
            ckpt_paths = [path for path in os.listdir(f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
            if len(ckpt_paths) == 0:
                if self.config['display']:
                    log.info(f'No .ckpt found in {self.config["log_dir"]}')
                return
            sort_func = lambda x:int(re.search(r"(\d+)", x).groups()[0])
            ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func, reverse=True)[0]}'
        if self.config['display']:
            log.info(f'loading checkpoint {ckpt_path}')
        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        self.set_ckpt(torch.load(ckpt_path, map_location=map_location))

    def get_ckpt(self):
        ckpt = {
            'epoch_idx': self.epoch_idx,
            'min_gen_val_loss': self.min_gen_val_loss,
            'seed': self.config['random_seed'],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        ckpt['model_state_dict'] = self.model.module.state_dict()
        return ckpt

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if self.config['generating']:  # or self.config['start_ckpt_for_generating']:
                ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
            else:
                ckpt_paths = [path for path in os.listdir(f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
                if len(ckpt_paths) == 0:
                    if self.config['display']:
                        log.info(f'No .ckpt found in {self.config["log_dir"]}')
                    return
                sort_func = lambda x:int(re.search(r"(\d+)", x).groups()[0])
                ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func, reverse=True)[0]}'

        if self.config['display']:
            log.info(f'loading checkpoint {ckpt_path}')
            epoch_name = osp.split(ckpt_path)[1].split('.')[0]
            if re.search(r"(\d+)", epoch_name):
                self.checkpoint_queue.append(ckpt_path)
                metrics_filename = osp.join(self.config['log_dir'], f'metrics_{epoch_name}.json')
                if osp.exists(metrics_filename):
                    self.metrics_queue.append(metrics_filename)

        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        self.set_ckpt(torch.load(ckpt_path, map_location=map_location))

    def match_model_key(self, pretrained_dict, model_dict):
        matched_dict = dict()
        not_found = []
        for key in pretrained_dict:
            if key in model_dict:
                matched_key = key
            elif key.startswith('encoder.') and key[8:] in model_dict:
                matched_key = key[8:]
            elif key.startswith('module.') and key[7:] in model_dict:
                matched_key = key[7:]
            elif 'encoder.' + key in model_dict:
                matched_key = 'encoder.' + key
            elif 'module.' + key in model_dict:
                matched_key = 'module.' + key
            else:
                not_found.append(key)
                continue
            matched_dict[matched_key] = pretrained_dict[key]
        print("Keys from pretrained_dict that were not found in model_dict:\n", not_found)
        return matched_dict
