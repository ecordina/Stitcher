import logging
import sys
from skimage.exposure import rescale_intensity,equalize_hist
import argparse
from os import path
from skimage.registration import phase_cross_correlation
import os
import numpy as np
import xml.etree.ElementTree as ET
from tifffile import imread, imwrite, TiffWriter
from skimage.transform import downscale_local_mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import zarr
import xarray as xr
from typing import Optional,Tuple
from imaris_ims_file_reader.ims import ims
import traceback
import ast
import datetime


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a formatter to define the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Environment:
# mamba install matplotlib tqdm scikit-image python zarr xarray
# pip install imaris-ims-file-reader

def build_ome_xml(stitched, voxel_x, voxel_y, voxel_z, significantBits=16, base_name="Image0"):

    # Full OME dimensions in required order
    ome_dims = ["T","C","Z","Y","X"]
    default_dim_sizes = {"T": 1,"C": 1,"Z": 1,"Y": 1,"X": 1}

    # Extract dimension sizes from stitched DataArray
    for dim in stitched.dims:
        default_dim_sizes[dim] = stitched.sizes[dim]

    # Determine OME-XML pixel type
    dtype_map = {
        np.dtype('uint8'): 'uint8',
        np.dtype('uint16'): 'uint16',
        np.dtype('float32'): 'float',
        np.dtype('float64'): 'double'
    }
    dtype_str = dtype_map.get(stitched.dtype, 'uint16')

    # Compose <Channel> tags if 'C' is present
    channel_elems = ""
    num_channels = default_dim_sizes["C"]
    for c in range(num_channels):
        channel_elems += f'''
   <Channel ID="Channel:0:{c}" SamplesPerPixel="1"><LightPath/></Channel>'''

    # Total number of planes = T * C * Z
    plane_count = default_dim_sizes["T"] * default_dim_sizes["C"] * default_dim_sizes["Z"]

    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
                         http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
 <Image ID="Image:0" Name="{base_name}">
  <Pixels ID="Pixels:0"
          DimensionOrder="TCZYX"
          Type="{dtype_str}"
          SizeX="{default_dim_sizes['X']}"
          SizeY="{default_dim_sizes['Y']}"
          SizeC="{default_dim_sizes['C']}"
          SizeZ="{default_dim_sizes['Z']}"
          SizeT="{default_dim_sizes['T']}"
          PhysicalSizeX="{voxel_x}" PhysicalSizeXUnit="um"
          PhysicalSizeY="{voxel_y}" PhysicalSizeYUnit="um"
          PhysicalSizeZ="{voxel_z}" PhysicalSizeZUnit="um"
          SignificantBits="{significantBits}">
{channel_elems}
   <TiffData IFD="0" PlaneCount="{plane_count}"/>
  </Pixels>
 </Image>
</OME>'''
    return ome_xml


def find_translation(ref: np.ndarray, src: np.ndarray) -> Tuple[float, float]:
    """
    Compute the translation needed to align `src` with `ref` using phase cross-correlation.

    Args:
        ref (np.ndarray): Reference image.
        src (np.ndarray): Source image to align.

    Returns:
        Tuple[float, float]: Translation offsets in the y and x directions.
    """
    shift_values, _, _ = phase_cross_correlation(
        ref, src, space='real', disambiguate=True, normalization="phase"
    )
    return shift_values



def crop_zero_borders(image: xr.DataArray) -> xr.DataArray:
    """
    Crop an xarray image to exclude all-zero borders along 'X' and 'Y' dimensions.

    Parameters:
    - image (xr.DataArray): Input image with dims including 'X' and 'Y'.

    Returns:
    - xr.DataArray: Cropped image.
    """
    if 'X' not in image.dims or 'Y' not in image.dims:
        raise ValueError("Input DataArray must have 'X' and 'Y' dimensions.")

    # Identify non-zero regions along Y and X
    reduced_y = image.any(dim=[d for d in image.dims if d != 'Y'], keepdims=False)
    reduced_x = image.any(dim=[d for d in image.dims if d != 'X'], keepdims=False)

    non_zero_y = reduced_y.values.nonzero()[0]
    non_zero_x = reduced_x.values.nonzero()[0]

    if len(non_zero_y) == 0 or len(non_zero_x) == 0:
        return image.isel({ 'Y': slice(0, 0), 'X': slice(0, 0) })  # Return empty slice if all zero

    y_start, y_end = non_zero_y[0], non_zero_y[-1] + 1
    x_start, x_end = non_zero_x[0], non_zero_x[-1] + 1

    return image.isel(X=slice(x_start, x_end), Y=slice(y_start, y_end))


    


def stitch_dragonfly_tiles(
    folder: Optional[str] = None,
    xml_file: Optional[str] = None,
    output_path: str = "./",
    max_project: bool = True,
    background_subtraction: bool = True,
    flat_field_path: str = '',
    dims: Optional[dict] = {"T":0,"C":1,"Z":2,"Y":3,"X":4},
    save_format: str = "tif",
    tile_size: Optional[int] = None,
    image_prefix: Optional[str] = "",
    overlap: Optional[int|str] = None,
    refine_overlap: Optional[bool] = None,
    max_overlap: bool = False,
    crop_overlap: bool = False,
    offset_channel: Optional[np.ndarray] = None,
    disable_logger: Optional[bool] = False,
    attrs: Optional[dict] = None,
) -> None:
    """
    Stitch tiled TIFF images from Dragonfly output using positions defined in an XML file.

    Parameters
    ----------
    folder : str
        Path to the directory containing TIFF tiles.
    xml_file : str
        Path to the Dragonfly XML metadata file describing tile positions.
    output_path : str
        Path to save the output stitched image and thumbnail.
    max_project : bool, optional
        Whether to perform max projection along Z-axis, by default True.
    background_subtraction : bool, optional
        Whether to subtract the minimum across time/channel for background removal, by default True.
    flat_field_path : str, optional
        Path to the Flat-Field image, if specified will perform Flat-field correction, if not, won't perform it
        Can be tiff files or ims, needs to be the full path image
    dims_order : str, optional
        Dimensional order of the image (e.g., "TCZXY"), by default "TCZXY".
    save_format : str, optional
        Output format, either "tif" or "zarr", by default "tif".
    tile_size : int, optional
        Size of each tile. If None, inferred from the first tile.
    image_prefix : str, optional
        Name of the Prefix of the tiles, to use instead of looking into the metadata
    overlap : int, optional
        Overlap in pixels between adjacent tiles. If None, inferred from metadata.
    refine_overlap : bool, optional
        Whether to refine the overlap using correlation, by default None.
    max_overlap : bool, optional
        Use max intensity blending in the overlap regions, by default True.
    crop_overlap : bool, optional
        Use hard cropping to remove overlap regions, by default False.
    offset_channel : np.ndarray, optional
        Placeholder for per-channel offset corrections. Not implemented yet.
    disable_logger: bool, optional
        Set True to disable Logging

    Raises
    ------
    ValueError
        If `save_format` is not "tif" or "zarr".
    FileNotFoundError
        If no TIFF files are found in the specified folder.
    AssertionError
        If both `max_overlap` and `crop_overlap` are True or False.
    """

    if disable_logger:
        logger.info("Disabling Logger")
        logger.disabled=True
    else:
        # Create a file handler to write logs to a file
        if xml_file is not None and xml_file!='':
            name = os.path.basename(xml_file)
        elif folder is not None and folder!='':
            name = folder
        else:
            now = datetime.datetime.now()
            name  = now.strftime("Stitching_date_%Y_%m_time_%d_%H_%M_%S")
        os.makedirs(output_path,exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_path,f'{name}.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Create a stream handler to print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    if (xml_file is None or xml_file=="") and (folder is None or folder==""):
        raise ValueError("At least one of xml_file or folder must be given")
    if (folder is None or folder==""):
        logger.info("Folder is None and XML file is given, assuming the tiles are in the same folder as the xml")
        folder = os.path.dirname(xml_file)+"/"
    if (xml_file is None or xml_file==""):
        logger.warning(r"XML file is None and Folder is given, assuming the xml is in the same folder as the tiles and\n /!\ /!\ assuming that there is only one xml file in it (One Experiment) /!\/!\ ")
        xml_file = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith("xml")][0]
        logger.info(f"Using {os.path.basename(xml_file)}")
    if save_format not in ("tif", "zarr"):
        raise ValueError("save_format must be 'tif' or 'zarr'")

    list_tiff = sorted([f for f in os.listdir(folder) if f.endswith("tiff") or f.endswith("tif") or f.endswith("ims")])
    if not list_tiff:
        raise FileNotFoundError("No TIFF or IMS files found in the folder.")
    if not ((max_overlap and not crop_overlap) or (not max_overlap and crop_overlap)):
        if (not max_overlap and not crop_overlap):
            logger.info("No Stitching method given, defaulting to Crop Overlap")
        else:
            logger.error("Both Stitching Method were chosen, only one is accepted!")
            raise valueError("Both Stitching Method were chosen, only one is accepted!")
    logger.info("Using Crop Overlap" if crop_overlap else "Using Max Projection Overlap" )

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')
        # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__

    # Parse positions from XML
    root = ET.parse(xml_file).getroot()
    voxel_elem = root.find("voxel_dims")
    voxel_z, voxel_y, voxel_x = abs(float(voxel_elem.attrib["D"])),abs(float(voxel_elem.attrib["V"])),abs(float(voxel_elem.attrib["H"]))
    n_max = max(int(root.find("dimensions").attrib.get("stack_columns")),int(root.find("dimensions").attrib.get("stack_rows"))) 
    list_tiff = list()
    dict_pos = {}
    for stack in root.findall(".//Stack"):
        file_name = stack.attrib.get("IMG_REGEX")
        if image_prefix!="":
            file_name = image_prefix+file_name[-9:]
        y = n_max  - int(stack.attrib.get("ROW"))
        x = n_max  - int(stack.attrib.get("COL"))
        dict_pos[file_name] = [y, x]
        list_tiff.append(file_name)
    logger.info(f"Found {len(list_tiff)} tiles for experiment {os.path.basename(xml_file)[:-4]}")
    #######################################
    # Open All Images 
    # #######################################

    list_images = []
    f_name = f"Loading {os.path.basename(xml_file).split('.')[0]} Images"
    for file in tqdm(list_tiff,desc=f_name):
        try:
            if file.endswith(".tiff") or file.endswith(".tif"):
                img = imread(os.path.join(folder, file))
                while len(img.shape)<len(dims):
                    img = np.expand_dims(img,axis=0)
            elif file.endswith(".ims"):
                blockPrint()
                img = ims(os.path.join(folder, file),squeeze_output=False)[:]
                enablePrint()
            else:
                logger.error(f"Error opening file {file}, at position {dict_pos[file]}\nSupported format are TIFF and IMS, file ends in {file.split('.')[-1]}")
            img = xr.DataArray(img,dims=dims)
            z_stack = len(img.Z)
            if max_project:
                img = img.max("Z")
            img = img.squeeze()
            list_images.append(img)
        except Exception as e:
            enablePrint()
            logger.error(f"Error opening file {file}, at position {dict_pos[file]}:", e)
            traceback.print_exc(file=sys.stdout)
    significantBits = 16 if img.dtype==np.uint16 else 8
    #######################################
    # Get Sizes, overlap
    #######################################
    if tile_size is None:
        tile_size = len(list_images[0].X)
        logger.info(f"Tile size is: {tile_size}")
    # Getting Overlap from XML
    if overlap is None or overlap=='':
        for stack in root.findall(".//Stack"):
            if stack.attrib.get("COL")=="1":
                actual_size = int(stack.attrib.get("ABS_H"))
                logger.info(f"Actual size is: {actual_size}")
                break
        overlap = tile_size-actual_size
        logger.info(f"Overlap Found in XML to be {overlap} px")
    else:
        if type(overlap)==str:
            overlap = int(overlap)
        logger.info(f"Using Given Overlap {overlap} px")
        actual_size = tile_size - overlap
    # Refining Overlap
    list_images = xr.concat(list_images,'stack_images')
    if refine_overlap:
        logger.info("Refining Overlap")
        one = two = None
        tile_variances = list_images.var(dim=[dim for dim in list_images.dims if dim!="stack_images"])
        # Get the index of the tile with the highest variance
        max_var_index = tile_variances.argmax(axis=0).item()
        name_max_variance = list_tiff[max_var_index]
        row,col = dict_pos[list_tiff[max_var_index]]
        row = n_max  - row 
        col = n_max  - col
        row_2 = row + 1
        if col >= n_max:
            col = col - 1
        if row_2 > n_max-1:
            row_2 = n_max - 1
            row = n_max - 2
        col = str(col)
        row = str(row)
        row_2 = str(row_2)
        for stack in root.findall(".//Stack"):
            if stack.attrib.get("COL")==col and stack.attrib.get("ROW")==row:
                one = stack.attrib.get("IMG_REGEX")
            if stack.attrib.get("COL")==col and stack.attrib.get("ROW")==row_2:
                two = stack.attrib.get("IMG_REGEX")
        test_images = [one,two]
        for i,file in enumerate(test_images):
            try:
                if file.endswith(".tiff") or file.endswith(".tif"):
                    test_images[i] = imread(os.path.join(folder, file))
                    while len(test_images[i].shape)<len(dims):
                        test_images[i] = np.expand_dims(test_images[i],axis=0)
                elif file.endswith(".ims"):
                    blockPrint()
                    test_images[i] = ims(os.path.join(folder, file),squeeze_output=False)[:]
                    enablePrint()
                else:
                    logger.error(f"Error opening file {file}, at position {dict_pos[file]}\nSupported format are TIFF and IMS, file ends in {file.split('.')[-1]}")
                test_images[i] = xr.DataArray(test_images[i],dims=dims)
                if max_project:
                    test_images[i] = test_images[i].max("Z")
                test_images[i] = test_images[i].squeeze()
            except Exception as e:
                enablePrint()
                logger.error(f"Error opening file {file}, at position {dict_pos[file]}:", e)
                traceback.print_exc(file=sys.stdout)
        for i in range(int(overlap//100)+1,10):
            test = find_translation(test_images[0].sel(X=slice(0,i*100)).data,test_images[1].sel(X=slice(-i*100,tile_size)).data)
            if (test!=0).any():
                if int(i*100-np.max(np.abs(test)))>0:
                    overlap = int(i*100-np.max(np.abs(test)))
                    break
        actual_size = tile_size - overlap
        logger.info(f"Using An Overlap of {overlap} px")
    #######################################
    # Background Substraction
    #######################################
    if background_subtraction:
        logger.info("Substracting Background")
        D_image = list_images.min("stack_images") # Background
        list_images = list_images - D_image
    #######################################
    # Flat-Field Correction
    #######################################
    # if flat_field_path is not None and flat_field_path!="":
    #     logger.info("Performing Flat-Field Correction")
    #     if flat_field_path.endswith("tiff") or flat_field_path.endswith("tif"):
    #         flat_field_image = imread(flat_field_path)
    #         while len(flat_field_image.shape)<len(dims):
    #             flat_field_image = np.expand_dims(flat_field_image,axis=0)
    #     elif file.endswith(".ims"):
    #         blockPrint()
    #         flat_field_image = ims(flat_field_path,squeeze_output=False)[:]
    #         enablePrint()
    if False:
        list_median = list_images.median(("X","Y"))
        len(np.where(list_median > list_median.max()//3)[0])/len(list_images.stack_images)
        flat_field_image = list_images.isel(stack_images = np.where(list_median > list_median.max()//1.5)[0]).max(dim = ('stack_images'))
        #flat_field_image = xr.DataArray(flat_field_image,dims=dims)
        # if max_project:
        #     flat_field_image = flat_field_image.max("Z")
        flat_field_image = flat_field_image.squeeze()
        D_image = list_images.min("stack_images") # Background, D Image in Flat-Field Correction
        m = (flat_field_image - D_image).mean()
        G_image = m / (flat_field_image - D_image)
        list_images = (list_images - D_image) * G_image
    #######################################
    # Create Empty Tiled Image
    #######################################
    dims_final = dict()
    for dim in list_images.dims:
        if dim not in ["X","Y","stack_images"]:
            dims_final[dim] = len(list_images[dim])
    dims_final["X"] = tile_size * (n_max+1)
    dims_final["Y"] = tile_size * (n_max+1)
    stitched = xr.DataArray(np.zeros(list(dims_final.values())),dims=list(dims_final.keys()))
    #######################################
    # Cropping Stitch
    #######################################

    if crop_overlap:
        f_name = f"Stitching {os.path.basename(xml_file).split('.')[0]} Images"
        for i in tqdm(range(len(list_images.stack_images)),desc=f_name):
            x_pos, y_pos = dict_pos[list_tiff[i]]
            begin_tile_x = actual_size * x_pos
            begin_tile_y = actual_size * y_pos
            end_tile_x = begin_tile_x+actual_size
            end_tile_y = begin_tile_y+actual_size
            stitched.loc[dict(X=slice(begin_tile_x, end_tile_x), Y= slice(begin_tile_y, end_tile_y))] = list_images.sel(stack_images=i,X=slice(0,actual_size),Y=slice(0,actual_size))
        
    #######################################
    # Max Overlap Stitch
    #######################################
    else:
        f_name = f"Stitching {os.path.basename(xml_file).split('.')[0]} Images"
        for i in tqdm(range(len(list_images.stack_images)),desc=f_name):
            x_pos, y_pos = dict_pos[list_tiff[i]]
            begin_tile_x = actual_size * x_pos
            begin_tile_y = actual_size * y_pos
            end_tile_x = begin_tile_x+tile_size
            end_tile_y = begin_tile_y+tile_size
            # Center [overlap:-overlap]
            stitched.loc[dict(X=slice(begin_tile_x+overlap, end_tile_x-overlap), Y=slice(begin_tile_y+overlap, end_tile_y-overlap))] = list_images.sel(stack_images=i,X=slice(overlap,-overlap),Y=slice(overlap,-overlap))


            # Left [:overlap,:]
            stitched.loc[dict(X=slice(begin_tile_x, begin_tile_x+overlap), Y=slice(begin_tile_y,end_tile_y))]  = xr.concat(([list_images.sel(stack_images=i,X=slice(0,overlap))          , stitched.sel(X=slice(begin_tile_x ,begin_tile_x+overlap), Y=slice(begin_tile_y,end_tile_y))]),dim="max").max("max")
            # Right [:,-overlap:]
            stitched.loc[dict(X=slice(end_tile_x-overlap,end_tile_x ),     Y=slice(begin_tile_y,end_tile_y))]  = xr.concat(([list_images.sel(stack_images=i,X=slice(-overlap,tile_size)) , stitched.sel(X=slice(end_tile_x-overlap,end_tile_x),      Y=slice(begin_tile_y,end_tile_y))]),dim="max").max("max")
            # Top [:,:overlap]
            stitched.loc[dict(Y=slice(begin_tile_y ,begin_tile_y+overlap), X=slice(begin_tile_x,end_tile_x))]  = xr.concat(([list_images.sel(stack_images=i,Y=slice(0,overlap))          , stitched.sel(Y=slice(begin_tile_y ,begin_tile_y+overlap), X=slice(begin_tile_x,end_tile_x))]),dim="max").max("max")
            # Bottom [:,-overlap:]
            stitched.loc[dict(Y=slice(end_tile_y-overlap,end_tile_y),      X=slice(begin_tile_x,end_tile_x))]  = xr.concat(([list_images.sel(stack_images=i,Y=slice(-overlap,tile_size)) , stitched.sel(Y=slice(end_tile_y-overlap,end_tile_y),      X=slice(begin_tile_x,end_tile_x))]),dim="max").max("max")

    logger.info("Stitching Done!")
    # Channel offset placeholder
    if offset_channel is not None and 'C' in stitched.dims:
        logger.info("Offsetting Channels")
        try:
            if isinstance(offset_channel, (int, float)):
                # Shift all channels by the same amount along the Y and X dims
                for ch in stitched.C.values:
                    stitched.loc[dict(Channel=ch)] = stitched.sel(Channel=ch).shift(X=offset_channel, Y=offset_channel, fill_value=0)
            elif hasattr(offset_channel, '__len__') and len(offset_channel) == stitched.sizes['C']:
                for ch, shift_val in zip(stitched.C.values, offset_channel):
                    stitched.loc[dict(Channel=ch)] = stitched.sel(Channel=ch).shift(X=shift_val, Y=shift_val, fill_value=0)
            else:
                logger.error(
                    f"Error: `offset_channel` is {offset_channel}, expected a scalar or a list of length {len(stitched.C)}"
                )
        except Exception as e:
            logger.error(f"Exception while shifting channels: {e}")

    logger.info("Cropping Empty tiles")
    stitched = crop_zero_borders(stitched)
    # Reordering Axes to Match OME format
    res = ''
    for i,letter in enumerate(list("TCZYXS")):
        if letter in stitched.dims:
            res+=letter
    stitched = stitched.transpose(*tuple(res))

    logger.info(f"Final array size is: {stitched.shape}")
    logger.info(f"Saving to {output_path}")
    # Save stitched image and thumbnail
    base_name = os.path.commonprefix(list_tiff)[:-2]
    thumbnail = stitched.copy()
    for dim in thumbnail.dims:
        if dim not in ["X","Y"]:
            thumbnail =  thumbnail.max(dim)
    thumbnail = downscale_local_mean(thumbnail.to_numpy(),5)
    thumbnail = equalize_hist(thumbnail,mask=thumbnail>np.percentile(thumbnail,40))
    thumbnail = rescale_intensity(thumbnail, in_range=(np.percentile(thumbnail,1),np.percentile(thumbnail,98)),out_range=np.uint8)
    thumbnail_name = os.path.basename(xml_file).split('.')[0]
    plt.imsave(os.path.join(output_path, f"thumbnail_{thumbnail_name}.png"), thumbnail , cmap="gray")
    if save_format == "tif":
        # Prepare stitched data
        axes = stitched.dims  # e.g. 'CYX'
        stitched = stitched.astype(np.uint16)
        data = stitched.transpose(*axes).to_numpy()
        logger.info(f"{data.max()}")
        logger.info(f"{data.shape}")
        axes = "".join(axes)

        logger.info(f"Image axes {axes}")
        logger.info(f"Image shape {stitched.shape}")
        logger.info(f"Image dtype {stitched.dtype}")
        pixelsize = np.abs(voxel_x)  # assuming isotropic XY for simplicity

        metadata = {
            'axes': axes,
            'SignificantBits': significantBits,
            'PhysicalSizeX': np.abs(voxel_x),
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeY': np.abs(voxel_y),
            'PhysicalSizeYUnit': 'um',
            'PhysicalSizeZ': np.abs(voxel_z),
            'PhysicalSizeZUnit': 'um',
            'Description': 'Stitched multi-tile image with pyramidal levels',
            'MapAnnotation': {  # for OMERO
                'Namespace': 'openmicroscopy.org/PyramidResolution',
                '1': f'{data.shape[0]//2} {data.shape[1]//2}',
                '2': f'{data.shape[0]//4} {data.shape[1]//4}',
                },
        }
        
        ome_xml = build_ome_xml(
            stitched=stitched,
            voxel_x=np.abs(voxel_x),
            voxel_y=np.abs(voxel_y),
            voxel_z=np.abs(voxel_z),
            significantBits=significantBits,
            base_name=base_name
        )

        logger.info(f"Metadata is {metadata}")
        # Pyramid parameters
        subresolutions = 2
        # options = dict(
        #     tile=(tile_size//2, tile_size//2),
        #     compression='jpeg',
        #     resolutionunit='MICROMETER',
        #     photometric='minisblack',
        #     maxworkers=2
        # )

        ome_path = os.path.join(output_path, f"{base_name}.ome.tif")
        with TiffWriter(ome_path, bigtiff=True) as tif:
            tif.write(
                data,
                description=ome_xml,
                subifds=subresolutions,
                resolution=(1 / pixelsize, 1 / pixelsize),
                metadata=metadata,
            )

            for level in range(subresolutions):
                mag = 2 ** (level + 1)
                down = data[..., ::mag, ::mag]
                tif.write(
                    down,
                    description=ome_xml,
                    subfiletype=1,
                    resolution=(1 / mag / pixelsize, 1 / mag / pixelsize),
                )

            # Write thumbnail as a separate series
            thumb_dims = tuple(i for i, ax in enumerate(stitched.dims) if ax not in ["X", "Y"])
            thumbnail = (stitched.max(thumb_dims).to_numpy()[::8, ::8]>>2).astype('uint8')
            tif.write(thumbnail, metadata={'Name': 'thumbnail'})
    elif save_format == "zarr":
        if attrs is not None:
            stitched.attrs.update(attrs)
        stitched.to_zarr(os.path.join(output_path, f"{base_name}.zarr"))
    logger.info("All Done Successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch Dragonfly TIFF tiles using metadata from an XML file."
    )
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Folder containing the TIFF tile images."
    )
    parser.add_argument(
        "--xml_file", type=str, required=True,
        help="Path to the Dragonfly XML metadata file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Directory to save the stitched image and thumbnail."
    )
    parser.add_argument(
        "--max_project", action="store_true",
        help="Apply max projection along Z axis."
    )
    parser.add_argument(
        "--no_background_subtraction", action="store_true",
        help="Disable background subtraction."
    )
    parser.add_argument(
        "--flat_field_path", type=str, required=False,
        help="Full Path to the Flat Field Image to perform Flat Field Correction, leave empty to not perform it"
    )
    parser.add_argument(
        "--dims", type=ast.literal_eval, default={"T":0,"C":1,"Z":2,"X":3,"Y":4},
        help='Dimension order of the input images (default: {"T":0,"C":1,"Z":2,"X":3,"Y":4}).'
    )
    parser.add_argument(
        "--attrs", type=ast.literal_eval, default=None,
        help='Attributes to attach to the zarr file. Only used when save_format is zarr. (Ex: {"Lateral Resolution":0.118},"Units":um})'
    )
    parser.add_argument(
        "--image_prefix",type=str,default='',required=False,
        help="Use this as prefix for the tiles instead of the name from the metadata"
    )
    parser.add_argument(
        "--save_format", type=str, choices=["tif", "zarr"], default="tif",
        help="Output format (tif or zarr)."
    )
    parser.add_argument(
        "--tile_size", type=int, default=None,
        help="Size of each tile (optional, inferred if not provided)."
    )
    parser.add_argument(
        "--overlap", default=None,
        help="Overlap between tiles in pixels (optional, inferred if not provided)."
    )
    parser.add_argument(
        "--refine_overlap", action="store_true",
        help="Refine overlap using image correlation (slow)."
    )
    parser.add_argument(
        "--crop_overlap", action="store_true",
        help="Stitch using hard cropping (instead of max blending)."
    )
    parser.add_argument("--disable_logger", action="store_true",
    help="Whether or not to use the logger, useful to deactivate when multiple images are stitched in parallel")
    

    args = parser.parse_args()
    stitch_dragonfly_tiles(
        folder=args.folder,
        xml_file=args.xml_file,
        output_path=args.output_path,
        max_project=args.max_project,
        background_subtraction=not args.no_background_subtraction,
        flat_field_path=args.flat_field_path,
        image_prefix=args.image_prefix,
        dims=args.dims,
        save_format=args.save_format,
        tile_size=args.tile_size,
        overlap=args.overlap,
        refine_overlap=args.refine_overlap,
        max_overlap=not args.crop_overlap,
        crop_overlap=args.crop_overlap,
        disable_logger=args.disable_logger,
        offset_channel=None,  # Future implementation
        attrs = args.attrs,
    )

if __name__ == "__main__":
    main()
