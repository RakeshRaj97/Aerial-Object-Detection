#!/usr/bin/env python
import datetime
import os
import sys
import logging
import time
import argparse
from math import isinf, isnan
from statistics import mean
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightnet as ln
import brambox as bb
from dataset import *

torch.set_num_threads(8)
log = logging.getLogger('lightnet.VOC.prune')
logging.logThreads = 0
logging.logProcesses = 0


def validate(params, dataloader, device, iou=0.5):
    """ Run testing dataset through model and compute mAP """
    params.network.eval()

    # Run through dataset
    with torch.no_grad():
        anno, det = [], []
        for data, target in tqdm(dataloader):
            data = data.to(device)
            output = params.network(data)
            output = params.post(output)

            output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
            anno.append(target)
            det.append(output)

    params.network.train()

    # Compute statistics
    anno = bb.util.concat(anno, ignore_index=True, sort=False)
    det = bb.util.concat(det, ignore_index=True, sort=False)

    aps = []
    for c in params.class_label_map:
        anno_c = anno[anno.class_label == c]
        det_c = det[det.class_label == c]

        # By default brambox considers ignored annos as regions -> we want to consider them as annos still
        matched_det = bb.stat.match_det(det_c, anno_c, iou, criteria=bb.stat.coordinates.iou, ignore=bb.stat.IgnoreMethod.SINGLE)
        pr = bb.stat.pr(matched_det, anno_c)
        aps.append(bb.stat.ap(pr))

    return mean(aps)


class PruneEngine(ln.engine.Engine):
    def start(self):
        self.params.reset()
        self.params.to(self.device)

        # Prune
        self.accuracy = 0
        self.prune_success = False
        self.min_accuracy = self.original_accuracy + self.min_accuracy_delta / 100
        self.max_accuracy = self.original_accuracy + self.max_accuracy_delta / 100
        self.pruner(*(p / 100 for p in self.prune_percentage))
        self.num_pruned = self.pruner.hard_pruned_channels
        log.info(f'[{self.filename}] Pruned {self.num_pruned} filters')
        self.validate()

        # Reset optimizer and scheduler
        self.scheduler.last_epoch = -1
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr']
        self.optimizer.zero_grad()

        # Setup training
        self.resize()
        self.train_loss = {'tot': [], 'coord': [], 'conf': []}

    def process_batch(self, data):
        data, target = data
        data = data.to(self.device)

        out = self.network(data)
        loss = self.loss(out, target) / self.batch_subdivisions
        loss.backward()

        self.train_loss['tot'].append(self.loss.loss.item())
        self.train_loss['coord'].append(self.loss.loss_coord.item())
        self.train_loss['conf'].append(self.loss.loss_conf.item())

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Get values from last batch
        tot = mean(self.train_loss['tot'][-self.batch_subdivisions:])
        coord = mean(self.train_loss['coord'][-self.batch_subdivisions:])
        conf = mean(self.train_loss['conf'][-self.batch_subdivisions:])
        self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f})')
        self.train_loss = {'tot': [], 'coord': [], 'conf': []}

        if isinf(tot) or isnan(tot):
            log.error('Infinite loss')
            self.sigint = True
            return

    @ln.engine.Engine.batch_end(10)
    def resize(self):
        if self.batch >= self.max_batches - 200:
            self.dataloader.change_input_dim(self.input_dimension, None)
        else:
            self.dataloader.change_input_dim(self.resize_factor, self.resize_range)

    @ln.engine.Engine.epoch_end()
    def validate(self):
        self.log('Start validation')
        self.accuracy = validate(self.params, self.testing_dataloader, self.device, self.iou)
        self.log(f'mAP={round(100*self.accuracy, 2)} [epoch={self.epoch}]')

    def quit(self):
        if self.num_pruned <= self.min_prune:
            return True
        elif self.accuracy >= self.max_accuracy:
            self.params.network.save(os.path.join(self.backup_folder, f'pruned-{self.filename}.pt'))
            self.prune_success = True
            return True
        elif self.batch >= self.max_batches:
            self.validate()
            if self.accuracy >= self.min_accuracy:
                self.params.network.save(os.path.join(self.backup_folder, f'pruned-{self.filename}.pt'))
                self.prune_success = True
            else:
                self.params.network.save(os.path.join(self.backup_folder, f'pruned-{self.filename}-FAILED.pt'))
            return True
        elif self.sigint:
            self.params.save(os.path.join(self.backup_folder, 'backup.state.pt'))
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder', default='./backup')
    parser.add_argument('-p', '--percentage', help='Pruning percentage stepsize', type=int, nargs='+')
    parser.add_argument('-m', '--minprune', help='Stop when pruning percentage results in less than X number of filters', type=int, default=1)
    parser.add_argument('-i', '--iou', help='IoU threshold for PR', type=float, default=0.5)
    parser.add_argument('-l', '--log', help='Log file', default=None)
    args = parser.parse_args()

    # Parse arguments
    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            log.warning('Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            raise ValueError('Backup path is not a folder')

    if args.log is not None:
        ln.logger.setLogFile(args.log, filemode='w')
        log.metadata(datetime.datetime.today())
        log.metadata('Terminal command:' + ' '.join(sys.argv))

    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')
    
    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)
    else:
        log.warn('No weights were given, starting with random weights!')
    log.metadata(repr(params))
    log.metadata('Network init: ' + str(params.network))

    # Dataloaders
    training_loader = ln.data.DataLoader(
        VOCDataset(params.train_set, params, True, remove_empty_images=True),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    data = params.val_set if hasattr(params, 'val_set') else params.test_set
    testing_loader = ln.data.DataLoader(
        VOCDataset(data, params, False),
        batch_size = params.mini_batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Original accuracy
    params.network.to(device)
    original_accuracy = validate(params, testing_loader, device, args.iou)
    log.info(f'Original network accuracy: mAP={round(100*original_accuracy, 2)}%')

    # Engine
    engine = PruneEngine(
        params, training_loader,
        prune_percentage=args.percentage, min_prune=args.minprune,
        testing_dataloader=testing_loader, original_accuracy=original_accuracy,
        device=device, backup_folder=args.backup,
        iou=args.iou,
    )

    # Pruning
    prune_step = 0
    prunable_channels_start = engine.pruner.prunable_channels
    tt1 = time.time()

    while True:
        prune_step += 1
        engine.filename = f'{prune_step:02d}'

        t1 = time.time()
        engine()
        t2 = time.time()

        log.info(f'Pruning step {prune_step:02d} [percentage={engine.pruner.prunable_channels/prunable_channels_start*100:.02f}, status={engine.prune_success}, time={t2-t1:.2f}s]')
        if not engine.prune_success:
            break

    tt2 = time.time()
    log.info(f'{prune_step} prunings took {tt2-tt1:.2f} seconds [{(tt2-tt1)/(prune_step):.3f} sec/pruning]')
