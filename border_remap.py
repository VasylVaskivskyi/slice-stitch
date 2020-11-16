from typing import List, Tuple, Union
import numpy as np
import dask

Image = np.ndarray


def _find_overlapping_border_labels(img1: Image, img2: Image, overlap: int, mode: str) -> dict:
    if mode == 'horizontal':
        img1_ov = img1[:, -overlap * 2: -overlap]
        img2_ov = img2[:, :overlap]
    elif mode == 'vertical':
        img1_ov = img1[-overlap * 2: -overlap, :]
        img2_ov = img2[:overlap, :]

    nrows, ncols = img2_ov.shape

    border_map = dict()

    for i in range(0, nrows):
        for j in range(0, ncols):
            old_value = img2_ov[i, j]
            if old_value in border_map:
                continue
            else:
                new_value = img1_ov[i, j]
                if old_value > 0 and new_value > 0:
                    border_map[old_value] = img1_ov[i, j]

    return border_map


def _get_map_of_overlapping_labels(img1: Image, img2: Image, img1_id: int, img2_id: int, overlap: int, mode: str):
    remapping = _find_overlapping_border_labels(img1, img2, overlap, mode=mode)
    return img2_id, remapping


def get_border_map(img_list: List[Image], x_ntiles: int, y_ntiles: int, overlap: int) -> dict:
    border_map = dict()
    htask = []
    # initialize border_map for all img ids
    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles):
            img_id = i * x_ntiles + j
            border_map[img_id] = {'horizontal': {}, 'vertical': {}}

    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles - 1):
            img1_id = i * x_ntiles + j
            img2h_id = i * x_ntiles + (j + 1)
            img1 = img_list[img1_id]
            img2 = img_list[img2h_id]
            htask.append(
                dask.delayed(_get_map_of_overlapping_labels)(img1, img2, img1_id, img2h_id, overlap, 'horizontal'))

    hor_values = dask.compute(*htask, scheduler='processes')
    hor_values = list(hor_values)

    for id_and_map in hor_values:
        border_map[id_and_map[0]]['horizontal'] = id_and_map[1]
    del hor_values

    vtask = []
    for i in range(0, y_ntiles - 1):
        for j in range(0, x_ntiles):
            img1_id = i * x_ntiles + j
            img2v_id = (i + 1) * x_ntiles + j
            img1 = img_list[img1_id]
            img2 = img_list[img2v_id]
            vtask.append(
                dask.delayed(_get_map_of_overlapping_labels)(img1, img2, img1_id, img2v_id, overlap, 'vertical'))

    ver_values = dask.compute(*vtask, scheduler='processes')
    ver_values = list(ver_values)

    for id_and_map in ver_values:
        border_map[id_and_map[0]]['vertical'] = id_and_map[1]

    return border_map


def remap_values(big_image: Image, border_map: dict,
                 tile_additions: np.ndarray, tile_shape: list,
                 overlap: int, x_ntiles: int, y_ntiles: int) -> Image:
    print('remapping values')
    x_axis = -1
    y_axis = -2
    x_tile_size = tile_shape[x_axis] - overlap * 2
    y_tile_size = tile_shape[y_axis] - overlap * 2

    this_tile_slice = [slice(None), slice(None)]

    n = 0
    for i in range(0, y_ntiles):
        yf = i * y_tile_size
        yt = yf + y_tile_size

        this_tile_slice[y_axis] = slice(yf, yt)

        for j in range(0, x_ntiles):
            xf = j * x_tile_size
            xt = xf + x_tile_size

            this_tile_slice[x_axis] = slice(xf, xt)

            this_tile = big_image[tuple(this_tile_slice)]
            try:
                hor_remap = border_map[n]['horizontal']
            except KeyError:
                hor_remap = {}
            try:
                ver_remap = border_map[n]['vertical']
            except KeyError:
                ver_remap = {}

            modified_x = False
            modified_y = False

            if hor_remap != {}:
                left_tile_addition = tile_additions[i, j - 1]
                this_tile_addition = tile_additions[i, j]
                for old_value, new_value in hor_remap.items():
                    this_tile[this_tile == old_value + this_tile_addition] = new_value + left_tile_addition
                modified_x = True

            if ver_remap != {}:
                top_tile_addition = tile_additions[i - 1, j]
                this_tile_addition = tile_additions[i, j]
                for old_value, new_value in ver_remap.items():
                    this_tile[this_tile == old_value + this_tile_addition] = new_value + top_tile_addition
                modified_y = True

            if modified_x or modified_y:
                big_image[tuple(this_tile_slice)] = this_tile

            n += 1
    return big_image
