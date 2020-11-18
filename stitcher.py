from pathlib import Path
import re
import argparse
from typing import List, Tuple, Union
import json

import numpy as np
import tifffile as tif
import pandas as pd
import dask

from stitcher_core import stitch_plane
from border_remap import get_border_map, remap_values
Image = np.ndarray


def alpha_num_order(string: str) -> str:
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return ''.join([format(int(x), '05d') if x.isdigit()
                    else x for x in re.split(r'(\d+)', string)])


def get_img_listing(in_dir: Path) -> List[Path]:
    allowed_extensions = ('.tif', '.tiff')
    listing = list(in_dir.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    img_listing = sorted(img_listing, key=lambda x: alpha_num_order(x.name))
    return img_listing


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def path_to_dict(path: Path):
    """
    Extract region, x position, y position and put into the dictionary
    {X: position, Y: position, Z: plane, CH: channel, path: path}
    """
    file_name_parts = re.split(r'(\d+)(?:_?)', path.name)[:-1]
    keys = [part for i, part in enumerate(file_name_parts) if i % 2 == 0]
    values = [int(part) for i, part in enumerate(file_name_parts) if i % 2 != 0]
    d = dict(zip(keys, values))
    d['path'] = path
    return d


def read_tile_mask(path: Path):
    return tif.imread(path_to_str(path)).astype(np.uint32)


def read_tile(path: Path):
    return tif.imread(path_to_str(path))


def load_tiles(path_list: List[Path], is_mask: False):
    if is_mask:
        task = []
        for path in path_list:
            task.append(dask.delayed(read_tile_mask)(path))
        tiles = dask.compute(*task, scheduler='threads')
    else:
        task = []
        for path in path_list:
            task.append(dask.delayed(read_tile)(path))
        tiles = dask.compute(*task, scheduler='threads')

    return tiles


def load_slicer_info_from_file(slicer_info_path: Path):
    with open(slicer_info_path, 'r') as s:
        slicer_info = json.load(s)
    padding = slicer_info['padding']
    overlap = slicer_info['overlap']
    return padding, overlap


def get_slicer_info(img_dir: Path, slicer_info_path: Path, padding_str: str, overlap: int):
    img_dir_slicer_path = img_dir.joinpath('slicer_info.json')
    if slicer_info_path is not None:
        padding, overlap = load_slicer_info_from_file(slicer_info_path)
    elif img_dir_slicer_path.exists():
        padding, overlap = load_slicer_info_from_file(img_dir_slicer_path)
    else:
        overlap = overlap
        padding_list = [int(i) for i in padding_str.split(',')]
        padding = {'left': padding_list[0], 'right': padding_list[1], 'top': padding_list[2], 'bottom': padding_list[3]}

    return padding, overlap


def get_dataset_info(img_dir: Path):
    img_paths = get_img_listing(img_dir)
    positions = [path_to_dict(p) for p in img_paths]
    df = pd.DataFrame(positions)
    df.sort_values(['CH', 'Y', 'X', 'Z'], inplace=True)
    df.reset_index(inplace=True)
    print(df)

    #n_channels = df['CH'].max()
    channel_ids = list(df['CH'].unique())
    zplane_ids = list(df['Z'].unique())
    x_ntiles = df['X'].max()
    y_ntiles = df['Y'].max()
    #n_zplanes = df['Z'].max()

    path_list_per_channel_per_tile = []

    for ch in channel_ids:
        ch_selection = df[df['CH'] == ch].index
        path_list_per_zplane = []
        for zplane in zplane_ids:

            z_selection = df[df.loc[ch_selection, 'Z'] == zplane].index
            path_list = list(df.loc[z_selection, 'path'])

            path_list_per_zplane.append(path_list)
        path_list_per_channel_per_tile.append(path_list_per_zplane)

    return path_list_per_channel_per_tile, x_ntiles, y_ntiles


def main(img_dir: Path, out_path: Path, overlap: int, padding_str: str, is_mask: False, slicer_info_path: Path):
    padding, overlap = get_slicer_info(img_dir, slicer_info_path, padding_str, overlap)
    path_list_per_channel_per_tile, x_ntiles, y_ntiles = get_dataset_info(img_dir)

    with tif.TiffFile(path_to_str(path_list_per_channel_per_tile[0][0][0])) as TF:
        tile_shape = list(TF.series[0].shape)
        # npages = len(TF.pages)
        dtype = TF.series[0].dtype
        ome_meta = '' if TF.ome_metadata is None else TF.ome_metadata

    big_image_x_size = (x_ntiles * (tile_shape[-1] - overlap * 2)) - padding['left'] - padding['right']
    big_image_y_size = (y_ntiles * (tile_shape[-2] - overlap * 2)) - padding['top'] - padding['bottom']

    if is_mask:
        print('getting values for remapping')
        border_map_per_channel_per_zplane = []
        for path_list_per_channel in path_list_per_channel_per_tile:
            for zplane_path_list in path_list_per_channel:
                border_map_per_zplane = []

                tile_list = load_tiles(zplane_path_list, is_mask)
                border_map = get_border_map(tile_list, x_ntiles, y_ntiles, overlap)

                border_map_per_zplane.append(border_map)
            border_map_per_channel_per_zplane.append(border_map_per_zplane)

        dtype = np.uint32
        ome_meta = ''
    else:
        border_map = None
        ome_meta = re.sub(r'\sSizeX="\d+"', ' SizeX="' + str(big_image_x_size) + '"', ome_meta)
        ome_meta = re.sub(r'\sSizeY="\d+"', ' SizeY="' + str(big_image_y_size) + '"', ome_meta)

    with tif.TiffWriter(path_to_str(out_path), bigtiff=True) as TW:
        print('stitching')
        for c, path_list_per_channel in enumerate(path_list_per_channel_per_tile):
            for z, zplane_path_list in enumerate(path_list_per_channel):
                if is_mask:
                    border_map = border_map_per_channel_per_zplane[c][z]

                tile_list = load_tiles(zplane_path_list, is_mask)
                plane, tile_additions = stitch_plane(tile_list, x_ntiles, y_ntiles, tile_shape, dtype, overlap,
                                                     padding, border_map)
                if is_mask:
                    plane = remap_values(plane, border_map, tile_additions, tile_shape, overlap, x_ntiles, y_ntiles)

                TW.save(plane, photometric='minisblack', description=ome_meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=Path, required=True, help='path to directory with images')
    parser.add_argument('-o', type=Path, required=True, help='path to output file')
    parser.add_argument('-v', type=int, default=0, help='overlap size in pixels, default 0')
    parser.add_argument('-p', type=str, default='0,0,0,0',
                        help='image padding that should be removed, 4 comma separated numbers: left, right, top, bottom.' +
                             'Default: 0,0,0,0')
    parser.add_argument('--mask', action='store_true', help='use this flag if image is a binary mask')
    parser.add_argument('--slicer_info', default=None, help='path to information from slicer.' +
                                                            'By default will check for it in input image directory')
    args = parser.parse_args()

    main(args.i, args.o, args.v, args.p, args.mask, args.slicer_info)
