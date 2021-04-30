#!/usr/bin/env python
import os
import argparse
import logging
import time
import statistics
import torch
from PIL import Image
import pandas as pd
import lightnet as ln
import brambox as bb
from dataset import *

log = logging.getLogger('lightnet.VOC.benchmark')
SEC_CONVERT = 1e3


def benchmark(params, dataset, device):
    network = params.network.to(device).eval()
    post = params.post

    # Run network
    log.info('Running Network')
    time_network, time_post, time_total = [], [], []
    anno, det = [], []
    with torch.no_grad():
        for data, target in dataset:
            data = data.unsqueeze(0).to(device)

            t0 = time.perf_counter()
            output = network(data)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            output = post(output)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            time_network.append(SEC_CONVERT * (t1 - t0))
            time_post.append(SEC_CONVERT * (t2 - t1))
            time_total.append(SEC_CONVERT * (t2 - t0))

            output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
            anno.append(target)
            det.append(output)

    log.info('Computing statistics')

    # mAP
    anno = bb.util.concat(anno, ignore_index=True, sort=False)
    det = bb.util.concat(det, ignore_index=True, sort=False)
    aps = []
    for c in params.class_label_map:
        anno_c = anno[anno.class_label == c]
        det_c = det[det.class_label == c]

        # By default brambox considers ignored annos as regions -> we want to consider them as annos still
        matched_det = bb.stat.match_det(det_c, anno_c, 0.5, criteria=bb.stat.coordinates.iou, ignore=bb.stat.IgnoreMethod.SINGLE)
        pr = bb.stat.pr(matched_det, anno_c)

        aps.append(bb.stat.ap(pr))

    m_ap = 100 * statistics.mean(aps)
    print(f'mAP: {round(m_ap,2):.2f}%')

    # Timing
    network_min = min(time_network)
    network_mean = statistics.mean(time_network)
    network_max = max(time_network)
    post_min = min(time_post)
    post_mean = statistics.mean(time_post)
    post_max = max(time_post)
    total_min = min(time_total)
    total_mean = statistics.mean(time_total)
    total_max = max(time_total)

    print(f'Network time (ms):     Min={network_min:6.2f}\tMax={network_max:6.2f}\tMean={network_mean:6.2f}')
    print(f'Postprocess time (ms): Min={post_min:6.2f}\tMax={post_max:6.2f}\tMean={post_mean:6.2f}')
    print(f'Total time (ms):       Min={total_min:6.2f}\tMax={total_max:6.2f}\tMean={total_mean:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark trained network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-t', '--thresh', help='Detection Threshold', type=float, default=None)
    parser.add_argument('-p', '--pruned', help='Whether weight file is a pruned file', action='store_true')
    args = parser.parse_args()

    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight.endswith('.state.pt'):
        if args.pruned:
            raise NotImplementedError('Cannot read pruned weights from `.state.pt` file')
        params.load(args.weight)
    elif args.pruned:
        params.network.load_pruned(args.weight)
    else:
        params.network.load(args.weight)

    if args.thresh is not None: # Overwrite threshold
        params.post[0].conf_thresh = args.thresh

    # Dataloader
    dataset = VOCDataset(params.test_set, params, augment=False)

    # Start benchmark
    benchmark(params, dataset, device)
