#!/usr/bin/env python
import datetime
import sys
import os
import logging
import time
import argparse
from math import isinf, isnan
from statistics import mean
import torch
import numpy as np
import lightnet as ln
from dataset import *

torch.set_num_threads(8)
log = logging.getLogger('lightnet.VOC.train')


class TrainEngine(ln.engine.Engine):
    def start(self):
        self.params.to(self.device)
        self.resize()
        self.optimizer.zero_grad()

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
        self.train_loss = {'tot': [], 'coord': [], 'conf': []}
        self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f})')

        if isinf(tot) or isnan(tot):
            log.error('Infinite loss')
            self.sigint = True
            return

    @ln.engine.Engine.batch_end(10000)
    def backup(self):
        self.params.save(os.path.join(self.backup_folder, f'weights_{self.batch}.state.pt'))
        log.info(f'Saved backup')

    @ln.engine.Engine.batch_end(10)
    def resize(self):
        if self.batch >= self.max_batches - 200:
        	self.dataloader.change_input_dim(self.input_dimension, None)
        else:
        	self.dataloader.change_input_dim(self.resize_factor, self.resize_range)

    def quit(self):
        if self.batch >= self.max_batches:
            self.params.network.save(os.path.join(self.backup_folder, 'final.pt'))
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
    parser.add_argument('weight', help='Path to weight file', default=None, nargs='?')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder', default='./backup')
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
            params.network.load(args.weight, strict=False)  # Disable strict mode for loading partial weights
    log.metadata(repr(params))
    log.metadata('Network init: ' + str(params.network))

    # Dataloader
    training_loader = ln.data.DataLoader(
        VOCDataset(params.train_set, params, True, remove_empty_images=True),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 0,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Start training
    eng = TrainEngine(
        params, training_loader,
        device=device, backup_folder=args.backup
    )
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch
    log.info(f'Training {b2-b1} batches took {t2-t1:.2f} seconds [{(t2-t1)/(b2-b1):.3f} sec/batch]')
