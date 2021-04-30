#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Transform annotations for VOCdevkit to the brambox pickle format
#
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import brambox as bb


DATA_SPLITS = {
    'trainvaltest': (
        [
            ('2012', 'train'),
        ],
        [
            ('2012', 'val'),
        ],
        [
            ('2012', 'test'),
        ]
    )
}


def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = os.path.splitext(root.find('filename').text)[0]
    return f'{folder}/JPEGImages/{filename}'


def get_data(root_folder, split, rmdifficult, verbose=False):
    data = []
    year = '2012'
    for (year, img_set) in split:
        with open(f'{root_folder}/VOCdevkit/VOC{year}/{img_set}.txt', 'r') as f:
            ids = f.read().strip().split()
        data += [f'{root_folder}/VOCdevkit/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids] #os.listdir('/home/rakesh/aerial/data/yolo_data/data/VOCdevkit/VOC2012/Annotations')]

    if verbose:
        print(f'\t{len(data)} xml files')

    print('\tParsing annotation files')
    annos = bb.io.load('anno_pascalvoc', data, identify)

    if verbose:
        print(f'\t{len(annos)} annotations in {len(annos.image.cat.categories)} images')

    if rmdifficult:
        if verbose:
            print('\tRemoving difficult annotations')
        annos = annos[~annos.difficult].reset_index(drop=True)
        if verbose:
            print(f'\t{len(annos)} annotations in {len(annos.image.cat.categories)} images')

    return annos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert annotations and split them in train/val/test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('root', help='Root data folder (Which contains VOCDevkit)')
    parser.add_argument('split', metavar='split', help=f'How to split the data {list(DATA_SPLITS.keys())}', choices=DATA_SPLITS.keys())
    parser.add_argument('-v', '--verbose', help='Print debug messages', action='store_true')
    parser.add_argument('-d', '--rmdifficult', help='Remove difficult training annotations', action='store_true')
    parser.add_argument('-x', '--extension', metavar='.EXT', help='Pandas extension to store data (See PandasParser)', default='.pkl')
    parser.add_argument('-o', '--output', help="Output directory for annotations (default: `root/split`)", default=None)
    args = parser.parse_args()

    # Check and create directory
    out_dir = args.output if args.output is not None else f'{args.root}/{args.split}'
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or len(os.listdir(out_dir)) != 0):
        raise ValueError(f'Output path is not a directory or is not empty! [{out_dir}]')
    os.makedirs(out_dir, exist_ok=True)
    
    # Parse annotations
    train, val, test = DATA_SPLITS[args.split]
    
    if len(train) != 0:
        print('Getting training annotations')
        train = get_data(args.root, train, args.rmdifficult, args.verbose)

        print('Generating training annotation file')
        bb.io.save(train, 'pandas', f'{out_dir}/train{args.extension}')

        print()

    if len(val) != 0:
        print('Getting validation annotations')
        train = get_data(args.root, val, args.rmdifficult, args.verbose)

        print('Generating validation annotation file')
        bb.io.save(train, 'pandas', f'{out_dir}/val{args.extension}')

        print()

    if len(test) != 0:
        print('Getting test annotations')
        train = get_data(args.root, test, args.rmdifficult, args.verbose)

        print('Generating test annotation file')
        bb.io.save(train, 'pandas', f'{out_dir}/test{args.extension}')

        print()

