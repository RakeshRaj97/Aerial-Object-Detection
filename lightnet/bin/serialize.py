#!/usr/bin/env python
import argparse
import logging
import torch
import lightnet as ln

log = logging.getLogger('lightnet.FLIR.train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Serialize network for usage in Photonnet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-p', '--pruned', help='Whether weight file is a pruned file', action='store_true')
    parser.add_argument('-o', '--output', help='Where to save traced model')
    args = parser.parse_args()


    # Parse arguments
    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight.endswith('.state.pt'):
        if args.pruned:
            raise NotImplementedError('Cannot read pruned weights from `.state.pt` file')
        params.load(args.weight)
    elif args.pruned:
        params.network.load_pruned(args.weight)
    else:
        params.network.load(args.weight)
    
    # Create random input
    log.info(f'Input Dimension: {params.input_dimension}')
    input_tensor = torch.rand(1, 3, *params.input_dimension[::-1])

    # Serialize model
    params.network.eval()
    traced_network = torch.jit.trace(params.network, input_tensor)

    # Save trace
    if args.output:
        traced_network.save(args.output)
        log.info(f'Saved the traced model to: {args.output}')
