from typing import List, Tuple, Union
import numpy as np


Image = np.ndarray


def get_slices(big_image: Image, hor_f: int, hor_t: int, ver_f: int, ver_t: int, padding: dict, overlap=0):
    # check if tile is over image boundary
    left_check = hor_f - padding['left']
    top_check = ver_f - padding['top']
    right_check = hor_t - big_image.shape[-1]
    bot_check = ver_t - big_image.shape[-2]

    left_pad_size = 0
    top_pad_size = 0
    right_pad_size = 0
    bot_pad_size = 0

    if left_check < 0:
        left_pad_size = abs(left_check)
        hor_f = 0
    if top_check < 0:
        top_pad_size = abs(top_check)
        ver_f = 0
    if right_check > 0:
        right_pad_size = right_check
        hor_t = big_image.shape[-1]
    if bot_check > 0:
        bot_pad_size = bot_check
        ver_t = big_image.shape[-2]

    big_image_slice = (slice(ver_f, ver_t), slice(hor_f, hor_t))
    tile_shape = (ver_t - ver_f, hor_t - hor_f)
    tile_slice = (slice(top_pad_size + overlap, tile_shape[-2] + overlap),
                  slice(left_pad_size + overlap, tile_shape[-1] + overlap))

    return big_image_slice, tile_slice


def stitch_plane(img_list: List[Image],
                 x_ntiles: int, y_ntiles: int,
                 tile_shape: list, dtype,
                 overlap: int, padding: dict, border_map: dict = None) -> Tuple[Image, Union[np.ndarray, None]]:

    x_axis = -1
    y_axis = -2

    tile_x_size = tile_shape[x_axis] - overlap * 2
    tile_y_size = tile_shape[y_axis] - overlap * 2

    big_image_x_size = (x_ntiles * tile_x_size) - padding["left"] - padding["right"]
    big_image_y_size = (y_ntiles * tile_y_size) - padding["top"] - padding["bottom"]

    big_image_shape = (big_image_y_size, big_image_x_size)
    big_image = np.zeros(big_image_shape, dtype=dtype)

    previous_tile_max = 0
    tile_additions = np.zeros((y_ntiles, x_ntiles), dtype=dtype)
    print('n tiles x,y:', (x_ntiles, y_ntiles))
    print('plane shape x,y:', big_image_shape[::-1])
    n = 0
    for i in range(0, y_ntiles):
        ver_f = i * tile_y_size
        ver_t = ver_f + tile_y_size

        for j in range(0, x_ntiles):
            hor_f = j * tile_x_size
            hor_t = hor_f + tile_x_size

            big_image_slice, tile_slice = get_slices(big_image, hor_f, hor_t, ver_f, ver_t, padding, overlap)
            tile = img_list[n]

            if border_map is not None:
                tile[np.nonzero(tile)] += previous_tile_max

            big_image[tuple(big_image_slice)] = tile[tuple(tile_slice)]

            if border_map is not None:
                tile_additions[i, j] = previous_tile_max

                # update previous tile max
                non_zero_selection = tile[np.nonzero(tile)]
                if len(non_zero_selection) > 0:
                    previous_tile_max = non_zero_selection.max()

            n += 1
    if border_map is None:
        tile_additions = None
    return big_image, tile_additions
