## Slicer and stitcher combination
These two scripts can be integrated into a pipeline to work in succession. 
The `slicer` divides image into tiles of the same size and saves metadata file `slicer_info.json` that can be used to 
stitch tiles back into the big image by `stitcher`. Both scripts works only with TIFF files.



### Slicer 

This script can divide image into tiles by size or by number of tiles. 
The `-s` parameter defines the size of the tile. For example, `-s 1000` will produces tiles with the shape 1000 x 1000.
The `-n` parameter tells script to divide image into given number of tiles both in horizontal and vertical direction. 
For example, `-n 4` will divide image into 16 tiles, 4 in horizontal direction and 4 in vertical.
Either `-s` or `-n` can be used, but one of them must be set to 0.


#### Command line arguments

**`-i`**    path to image file

**`-o`**    path to output directory

**`-s`**    size of a tile, default 1000, if set to 0, then `-n` parameter used instead

**`-n`**    number of tiles, default 0, if left 0, then `-s` parameter used instead

**`-v`**    size of overlap on one side of image, default 0 (no overlap). 
            e.g. if `-v 50`, then each tile has 50px overlap on each side: top, bottom, left, right.  

**`--nzplanes`**    number of z-planes, default 1

**`--nchannels`**   number of channels, default 1

**`--selected_channels`**   space separated ids of channels you want to slice, e.g. 0 1 3, default all


#### Example usage

`python slicer.py -i /path/to/img.tif -o /path/to/out/dir -s 1000 -v 50`

<br/>

### Stitcher

This script allows to stitch tiles with constant size and overlap (provided by `slicer`).
It can pick up stitching parameters (overlap and padding) from slicer metadata or they can be provided manually.
By default it will try to load slicer metadata from `slicer_info.json` inside directory with tiles.
Otherwise you can provide overlap and padding using `-v` and `-p` parameters.
If you want to stitch binary mask, you have to use flag `--mask`. 
It will ensure that border values in tiles are correctly stitched. 

#### Command line arguments


**`-i`** path to directory with images

**`-o`** path to output file

**`-v`** overlap size in pixels, default 0

**`-p`** image padding that should be removed, 4 comma separated numbers: left, right, top, bottom. Default: 0,0,0,0.

**`--mask`** use this flag if image is a binary mask

**`--slicer_info`**  path to metadata from slicer that contains padding and overlap. 
                     By default will check for it is a file `slicer_info.json` in the directory with tiles.



#### Example usage

`python stitcher.py -i /path/to/dir/with/tiles -o /path/to/out/stitched_img.tif -v 50 -p 50,100,30,0`
<br/>
<br/>

### Requirements

`numpy pandas tifffile dask`
