from pathlib import Path
import argparse
import json

import tifffile as tif
import dask

from slicer_core import split_by_size, split_by_ntiles


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def split_plane(in_path, page, zplane, channel, tile_size, ntiles, overlap):
    if ntiles == 0:
        plane_split, plane_tile_names, slicer_info = split_by_size(tif.imread(path_to_str(in_path), key=page),
                                                                   zplane, channel, tile_size, tile_size, overlap)
    elif tile_size == 0:
        plane_split, plane_tile_names, slicer_info = split_by_ntiles(tif.imread(path_to_str(in_path), key=page),
                                                                     zplane, channel, ntiles, ntiles, overlap)
    return plane_split, plane_tile_names, slicer_info


def save_slicer_info(out_dir: Path, slicer_info: dict):
    with open(out_dir.joinpath('slicer_info.json'), 'w') as s:
        json.dump(slicer_info, s, sort_keys=False, indent=4)


def split_tiff(in_path: Path, out_dir: Path, tile_size: int, ntiles: int, overlap: int,
               nzplanes: int, nchannels: int, selected_channels: list):
    with tif.TiffFile(in_path) as TF:
        npages = len(TF.pages)

    for c in selected_channels:
        for z in range(0, nzplanes):
            page = c * nzplanes + z
            print('page', page + 1, '/', npages)
            this_plane_tiles, this_plane_tile_names, slicer_info = split_plane(in_path, page, z, c, tile_size, ntiles, overlap)
            task = []
            for i, tile in enumerate(this_plane_tiles):
                out_path = path_to_str(out_dir.joinpath(this_plane_tile_names[i]))
                task.append(dask.delayed(tif.imwrite)(out_path, tile, photometric='minisblack'))

            dask.compute(*task, scheduler='threads')
    save_slicer_info(out_dir, slicer_info)


def main(in_path: Path, out_dir: Path, tile_size: int, ntiles: int, overlap: int,
         nzplanes: int, nchannels: int, selected_channels: list):

    if in_path.suffix not in ('.tif', '.tiff'):
        raise ValueError('Only tif, tiff input files are accepted')

    if ntiles != 0 and tile_size != 0:
        raise ValueError('One of the parameters -s or -n must be zero')

    if ntiles == 0 and tile_size == 0:
        raise ValueError('One of the parameters -s or -n must be non zero')

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    if selected_channels is None:
        selected_channels = list(range(0, nchannels))
    else:
        selected_channels = [ch_id for ch_id in selected_channels if ch_id < nchannels]

    split_tiff(in_path, out_dir, tile_size, ntiles, overlap, nzplanes, nchannels, selected_channels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split image into number of tiles')
    parser.add_argument('-i', type=Path, help='path to image file')
    parser.add_argument('-o', type=Path, help='path to output dir')
    parser.add_argument('-s', type=int, default=1000,
                        help='size of tile, default 1000x1000, if set to 0, then -n parameter used instead')
    parser.add_argument('-n', type=int, default=0,
                        help='number of tiles, default 0, if set to 0, then -s parameter used instead')
    parser.add_argument('-v', type=int, default=0, help='size of overlap, default 0 (no overlap)')
    parser.add_argument('--nzplanes', type=int, default=1, help='number of z-planes, default 1')
    parser.add_argument('--nchannels', type=int, default=1, help='number of channels, default 1')
    parser.add_argument('--selected_channels', type=int, nargs='+', default=None,
                        help="space separated ids of channels you want to slice, e.g. 0 1 3, default all")

    args = parser.parse_args()
    main(args.i, args.o, args.s, args.n, args.v,
         args.nzplanes, args.nchannels, args.selected_channels)
