import logging
import sys
import re
from skimage.exposure import rescale_intensity,equalize_hist
from skimage.filters import gaussian
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
import numpy as np
import re
from scipy.spatial import cKDTree

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


def extract_scene_id(filename):
    match = re.search(r"s(\d+)", filename)
    return match.group(1) if match else "0"

def parse_xy(filename):
    """
    Extract x/y grid coordinates from filenames like:
    Fibres_b0s51z2c1x64222-1392y4914-1040m25.tiff
    """
    m = re.search(r"x(\d+)-\d+y(\d+)-\d+m(\d+)", filename)
    if not m:
        return None
    x = int(m.group(1))
    y = int(m.group(2))
    return x, y

def get_frame_step(xml_file):
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_file).getroot()
    text = root.find('Tags').find('DetectorState').text
    text = text[text.find('<Frame>')+len("<Frame>"):text.find('</Frame>')].split(',')
    _, _, w, h = map(int, text)

    step_x_px = int(0.90 * w)
    step_y_px = int(0.90 * h)

    return w, h, step_x_px, step_y_px

def group_by_spacing(files,xml_file):
    coords = {}

    for f in files:
        xy = parse_xy(f)
        if xy is not None:
            coords[f] = xy

    if len(coords) == 0:
        return []

    _,_, dx_ref, dy_ref = get_frame_step(xml_file)
    print('REF: ',dx_ref,dy_ref)
    tx = dx_ref
    ty = dy_ref

    def is_neighbor(p1, p2, dx, dy):
        x1, y1 = p1
        x2, y2 = p2

        return (
            abs(x1 - x2) <= 1.1 * dx and
            abs(y1 - y2) <= 1.1 * dy
        )

    visited = set()
    groups = []

    items = list(coords.keys())

    for f in items:
        if f in visited:
            continue

        stack = [f]
        group = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)

            for f2 in items:
                if f2 not in visited and is_neighbor(coords[cur], coords[f2], tx, ty):
                    stack.append(f2)

        groups.append(group)

    return groups

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
    overlap_mode: str = "blend",
    offset_channel: Optional[np.ndarray] = None,
    disable_logger: Optional[bool] = False,
    attrs: Optional[dict] = None,
    metadata_format: str = "dragonfly",
    gaussian_sigma: float = 0.0,
    normalize_intensity: bool = False,
    split_by_scene: bool = False,              
    selected_files: Optional[list] = None,     
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
    overlap_mode : str, optional
        Stitching mode for overlap regions ('max', 'mean', 'crop', or 'blend'), by default 'blend'.
    offset_channel : np.ndarray, optional
        Placeholder for per-channel offset corrections. Not implemented yet.
    disable_logger: bool, optional
        Set True to disable Logging
    attrs : dict, optional
        Attributes to save with the image.
    metadata_format : str, optional
        Format of the metadata, either "dragonfly" or "zeiss_zen".
    gaussian_sigma : float, optional
        Sigma for Gaussian filtering at the end of stitching, by default 0.0 (disabled).
    normalize_intensity : bool, optional
        Whether to normalize intensity level across tiles, by default False.
    split_by_scene : bool, optional
        Whether to split the stitching process by scene, by default False.
    selected_files : list, optional
        List of files to stitch, if None, all files will be stitched.

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

    all_files = sorted([f for f in os.listdir(folder) if f.endswith(("tiff","tif","ims"))])

    if selected_files is not None:
        list_tiff = [f for f in all_files if f in selected_files]
    else:
        list_tiff = all_files


    if not list_tiff:
        raise FileNotFoundError("No TIFF or IMS files found in the folder.")


    if overlap_mode not in ("max", "mean", "crop", "blend"):
        logger.error(f"Invalid overlap_mode '{overlap_mode}'. Choosing 'blend' instead.")
        overlap_mode = "blend"
    logger.info(f"Using Overlap Mode: {overlap_mode}")

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')
        # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__

    # Parse positions from XML
    root = ET.parse(xml_file).getroot()
    
    if metadata_format == "zeiss_zen":
        logger.info("Using Zeiss Zen Metadata Format")
        # Automatic folder detection for Zen
        if folder is not None:
            # Normalize folder path for comparison
            norm_folder = os.path.normpath(folder)
            norm_xml_dir = os.path.normpath(os.path.dirname(xml_file))
            if norm_folder == norm_xml_dir:
                potential_folder = xml_file.replace("_metadata.xml", "_files")
                if os.path.isdir(potential_folder):
                    folder = potential_folder
                    logger.info(f"Detected Zen folder: {folder}")

        # Extract voxel dimensions from Zen Scaling
        scaling = root.find(".//Scaling/Items")
        if scaling is not None:
            try:
                vx_node = scaling.find("Distance[@Id='X']/Value")
                vy_node = scaling.find("Distance[@Id='Y']/Value")
                voxel_x = float(vx_node.text) * 1e6 if vx_node is not None else 1.0
                voxel_y = float(vy_node.text) * 1e6 if vy_node is not None else 1.0
                voxel_z = 1.0 # Default
                logger.info(f"Voxel size found in Zen XML: {voxel_x}x{voxel_y}x{voxel_z} um")
            except Exception as e:
                logger.warning(f"Could not parse voxel sizes from Zen XML: {e}. Using defaults.")
                voxel_z, voxel_y, voxel_x = 1.0, 1.0, 1.0
        else:
            voxel_z, voxel_y, voxel_x = 1.0, 1.0, 1.0
            
        list_tiff_all = sorted([f for f in os.listdir(folder) if (f.endswith("tiff") or f.endswith("tif")) and "metadata" not in f])
        if selected_files is not None:
            list_tiff_all = [f for f in list_tiff_all if f in selected_files]
        list_tiff = []
        dict_pos = {}
        # Pattern: ..._b<B>s<S>[z<Z>]c<C>x<X>-<W>y<Y>-<H>m<M>.tiff
        zen_pattern = re.compile(r".*_b\d+s\d+(?:z(\d+))?c(\d+)x(\d+)-(\d+)y(\d+)-(\d+)m(\d+)\.tiff")
        
        # Try to extract target Z from XML filename if it follows the pattern
        target_z = None
        xml_name = os.path.basename(xml_file)
        xml_match = zen_pattern.search(xml_name)
        if xml_match and xml_match.group(1) is not None:
            target_z = xml_match.group(1)
            logger.info(f"Target Z-plane from XML: {target_z}")

        max_px_x = 0
        max_px_y = 0
        channels = set()
        for file_name in list_tiff_all:
            match = zen_pattern.search(file_name)
            if match:
                z_val = match.group(1)
                # If we have a target Z, filter out other planes
                if target_z is not None and z_val != target_z:
                    continue
                
                # Group indices shift if Z is present (z=1, c=2, x=3, w=4, y=5, h=6)
                c = int(match.group(2))
                x_start = int(match.group(3))
                width = int(match.group(4))
                y_start = int(match.group(5))
                height = int(match.group(6))
                
                dict_pos[file_name] = [y_start, x_start, c, height, width] # Absolute pixel positions + channel + extent
                list_tiff.append(file_name)
                channels.add(c)
                max_px_x = max(max_px_x, x_start + width)
                max_px_y = max(max_px_y, y_start + height)
        
        num_channels = len(channels)
        n_max = 0 # Not used for absolute positioning calculation of total size
        # We will use max_px_x, max_px_y later
        actual_size = 1 
        # overlap = 0
        logger.info(f"Found {len(list_tiff)} tiles ({num_channels} channels) for experiment {os.path.basename(xml_file)}")
    else:
        # Dragonfly Metadata
        voxel_elem = root.find("voxel_dims")
        voxel_z, voxel_y, voxel_x = abs(float(voxel_elem.attrib["D"])),abs(float(voxel_elem.attrib["V"])),abs(float(voxel_elem.attrib["H"]))
        n_max = max(int(root.find("dimensions").attrib.get("stack_columns")),int(root.find("dimensions").attrib.get("stack_rows"))) 
        list_tiff = list()
        dict_pos = {}
        # We'll infer tile_size later, but for now we need a placeholder if we use consistent indexing
        for stack in root.findall(".//Stack"):
            file_name = stack.attrib.get("IMG_REGEX")
            if image_prefix!="":
                file_name = image_prefix+file_name[-9:]
            y = n_max  - int(stack.attrib.get("ROW"))
            x = n_max  - int(stack.attrib.get("COL"))
            dict_pos[file_name] = [y, x, 0, None, None] # Extent None for now
            list_tiff.append(file_name)
        logger.info(f"Found {len(list_tiff)} tiles for experiment {os.path.basename(xml_file)[:-4]}")

    if split_by_scene:
        logger.info("Grouping tiles using dx/dy tolerance clustering")

        groups = group_by_spacing(list_tiff,xml_file)

        logger.info(f"Found {len(groups)} spatial groups")

        for i, files in enumerate(groups):
            logger.info(f"Processing group {i} with {len(files)} tiles")

            stitch_dragonfly_tiles(
                folder=folder,
                xml_file=xml_file,
                output_path=os.path.join(output_path, f"group_{i}"),
                max_project=max_project,
                background_subtraction=background_subtraction,
                flat_field_path=flat_field_path,
                dims=dims,
                save_format=save_format,
                tile_size=tile_size,
                image_prefix=image_prefix,
                overlap=overlap,
                refine_overlap=refine_overlap,
                overlap_mode=overlap_mode,
                offset_channel=offset_channel,
                disable_logger=disable_logger,
                attrs=attrs,
                metadata_format=metadata_format,
                gaussian_sigma=gaussian_sigma,
                normalize_intensity=normalize_intensity,
                split_by_scene=False,
                selected_files=files,
            )

        return
    #######################################
    # Open All Images 
    # #######################################

    list_images = []
    f_name = f"Loading {os.path.basename(xml_file).split('.')[0]} Images"
    for file in tqdm(list_tiff,desc=f_name):
        try:
            path_img = os.path.join(folder, file)
            if file.endswith(".tiff") or file.endswith(".tif"):
                img_raw = imread(path_img)
            elif file.endswith(".ims"):
                blockPrint()
                img_raw = ims(path_img,squeeze_output=False)[:]
                enablePrint()
            else:
                logger.error(f"Error opening file {file}, at position {dict_pos[file]}\nSupported format are TIFF and IMS, file ends in {file.split('.')[-1]}")
                continue
            
            if metadata_format == "zeiss_zen":
                # Zeiss Zen Exported tiles are typically 2D (Y, X) or 3D (C, Y, X)
                # We squeeze to remove singleton dimensions
                img = np.squeeze(img_raw)
                if len(img.shape) == 3:
                    # Max project if it has a channel/singleton dim
                    # We assume Y and X are the large ones
                    if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                        img = np.max(img, axis=0)
                    elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
                        img = np.max(img, axis=2)
                img = xr.DataArray(img, dims=["Y", "X"])
            else:
                img = img_raw
                # Dragonfly: Expand dims to match dims parameter
                while len(img.shape)<len(dims):
                    img = np.expand_dims(img,axis=0)
                img = xr.DataArray(img, dims=dims)
                if max_project:
                    img = img.max("Z")
                img = img.squeeze()
            
            list_images.append(img)
        except Exception as e:
            enablePrint()
            logger.error(f"Error opening file {file}, at position {dict_pos[file]}:", e)
            traceback.print_exc(file=sys.stdout)
    if not list_images:
        logger.error(f"No images loaded for {xml_file}. Please check the folder and regex.")
        return
    img = list_images[0]
    significantBits = img.to_numpy().itemsize*8
    logger.info(f"{significantBits} Significant Bits")
    #######################################
    # Get Sizes, overlap
    #######################################
    if tile_size is None:
        tile_size_x = int(list_images[0].sizes["X"])
        tile_size_y = int(list_images[0].sizes["Y"])
    else:
        tile_size_x = tile_size_y = tile_size
    logger.info(f"Tile size is: {tile_size_x}x{tile_size_y}")

    # Getting Overlap
    overlap_x = overlap_y = 0
    if overlap is not None and isinstance(overlap, str) and overlap.endswith("%"):
        try:
            percentage = float(overlap.strip('%'))
            overlap_x = int(tile_size_x * percentage / 100)
            overlap_y = int(tile_size_y * percentage / 100)
            logger.info(f"Calculated overlap from percentage: X={overlap_x} px, Y={overlap_y} px ({percentage}%)")
        except ValueError:
            logger.warning(f"Could not parse overlap percentage '{overlap}'. Using default.")
    elif overlap is not None and overlap != '':
        overlap_x = overlap_y = int(overlap)

    if metadata_format == "zeiss_zen":
        actual_size_x = actual_size_y = 1
    elif overlap is None or overlap=='':
        actual_size_val = tile_size_x # Default
        for stack in root.findall(".//Stack"):
            if stack.attrib.get("COL")=="1":
                actual_size_val = int(stack.attrib.get("ABS_H"))
                break
        actual_size_x = actual_size_y = actual_size_val
        overlap_x = tile_size_x - actual_size_x
        overlap_y = tile_size_y - actual_size_y
        logger.info(f"Overlap Found in XML: X={overlap_x} px, Y={overlap_y} px")
    else:
        actual_size_x = tile_size_x - overlap_x
        actual_size_y = tile_size_y - overlap_y
        logger.info(f"Using Overlap: X={overlap_x} px, Y={overlap_y} px")
    
    # Fill in extents for Dragonfly if they were None
    for key in dict_pos:
        if dict_pos[key][3] is None:
            dict_pos[key][3] = tile_size_y
            dict_pos[key][4] = tile_size_x

    # Refining Overlap
    list_images = xr.concat(list_images,'stack_images')
    if refine_overlap and metadata_format != "zeiss_zen":
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
        for i in range(int(overlap_x//100)+1,10):
            test = find_translation(test_images[0].sel(X=slice(0,i*100)).data,test_images[1].sel(X=slice(-i*100,tile_size_x)).data)
            if (test!=0).any():
                if int(i*100-np.max(np.abs(test)))>0:
                    overlap_x = overlap_y = int(i*100-np.max(np.abs(test)))
                    break
        actual_size_x = tile_size_x - overlap_x
        actual_size_y = tile_size_y - overlap_y
        logger.info(f"Using Refined Overlap: X={overlap_x} px, Y={overlap_y} px")
    elif refine_overlap and metadata_format == "zeiss_zen":
        logger.warning("Refine overlap is not currently supported for Zeiss Zen format. Skipping.")
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
    if flat_field_path is not None and flat_field_path!="":
        logger.info("Performing Flat-Field Correction")
        try:
            if flat_field_path.endswith("tiff") or flat_field_path.endswith("tif"):
                ff_raw = imread(flat_field_path)
            elif flat_field_path.endswith("ims"):
                blockPrint()
                ff_raw = ims(flat_field_path, squeeze_output=False)[:]
                enablePrint()
            else:
                logger.error(f"Flat-field format not supported: {flat_field_path}")
                ff_raw = None

            if ff_raw is not None:
                if metadata_format == "zeiss_zen":
                    ff_img = np.squeeze(ff_raw)
                    if len(ff_img.shape) == 3:
                        if ff_img.shape[0] < ff_img.shape[1] and ff_img.shape[0] < ff_img.shape[2]:
                            ff_img = np.max(ff_img, axis=0)
                        elif ff_img.shape[2] < ff_img.shape[0] and ff_img.shape[2] < ff_img.shape[1]:
                            ff_img = np.max(ff_img, axis=2)
                    flat_field_image = xr.DataArray(ff_img, dims=["Y", "X"])
                else:
                    ff_img = ff_raw
                    while len(ff_img.shape) < len(dims):
                        ff_img = np.expand_dims(ff_img, axis=0)
                    flat_field_image = xr.DataArray(ff_img, dims=dims)
                    if max_project:
                        flat_field_image = flat_field_image.max("Z")
                    flat_field_image = flat_field_image.squeeze()

                # Perform Correcton: Corrected = (Raw - Dark) * Mean(Flat - Dark) / (Flat - Dark)
                D_image = list_images.min("stack_images") # Background/Dark Image
                
                # Ensure we don't divide by zero or negative values
                illumination = (flat_field_image - D_image).clip(min=1)
                m = illumination.mean()
                G_image = m / illumination
                
                logger.info("Applying Flat-Field Correction to tiles")
                list_images = (list_images - D_image) * G_image
                # Ensure no negative values or overflows before casting back to uint16 later
                list_images = list_images.clip(min=0)
                
        except Exception as e:
            enablePrint()
            logger.error(f"Error during Flat-Field Correction: {e}")
            traceback.print_exc(file=sys.stdout)

    if normalize_intensity:
        logger.info("Normalizing intensity across tiles")
        try:
            # list_images has dims ('stack_images', 'C', 'Y', 'X') or similar
            if "C" in list_images.dims:
                for c in list_images.C:
                    # Calculate median for each tile in this channel
                    tile_medians = list_images.sel(C=c).median(dim=("X", "Y"))
                    # Filter out very dark/empty tiles to avoid extreme scaling
                    valid_medians = tile_medians.values[tile_medians.values > 5] 
                    if len(valid_medians) > 0:
                        target_median = np.mean(valid_medians)
                        logger.debug(f"Channel {c.item()} medians range: {tile_medians.min().item():.1f} - {tile_medians.max().item():.1f}, target: {target_median:.1f}")
                        # Scale factor: target / current
                        # Use clip on tile_medians to avoid div by zero
                        scale_factors = target_median / tile_medians.where(tile_medians > 1, 1)
                        # Multiply tile-wise
                        list_images.loc[dict(C=c)] = (list_images.sel(C=c) * scale_factors).clip(min=0, max=65535)
            else:
                tile_medians = list_images.median(dim=("X", "Y"))
                valid_medians = tile_medians.values[tile_medians.values > 5]
                if len(valid_medians) > 0:
                    target_median = np.mean(valid_medians)
                    logger.debug(f"Medians range: {tile_medians.min().item():.1f} - {tile_medians.max().item():.1f}, target: {target_median:.1f}")
                    scale_factors = target_median / tile_medians.where(tile_medians > 1, 1)
                    list_images = (list_images * scale_factors).clip(min=0, max=65535)
        except Exception as e:
            logger.error(f"Error during intensity normalization: {e}")
            traceback.print_exc(file=sys.stdout)
    # Create Empty Tiled Image
    dims_final = dict()
    if metadata_format == "zeiss_zen":
        # Force OME order C, Y, X
        dims_final = {"C": num_channels, "Y": max_px_y, "X": max_px_x}
    else:
        for dim in list_images.dims:
            if dim not in ["X","Y","stack_images"]:
                dims_final[dim] = len(list_images[dim])
        dims_final["Y"] = tile_size_y * (n_max + 1)
        dims_final["X"] = tile_size_x * (n_max + 1)
    
    # Use float32 for blend mode to avoid rounding errors during accumulation
    if overlap_mode == "blend":
        stitched = xr.DataArray(np.zeros(list(dims_final.values()), dtype=np.float32), dims=list(dims_final.keys()), name="stitched")
        weights = xr.DataArray(np.zeros(list(dims_final.values()), dtype=np.float32), dims=list(dims_final.keys()), name="weights")
    else:
        # Use uint16 to save memory and avoid float64 default
        stitched = xr.DataArray(np.zeros(list(dims_final.values()), dtype=np.uint16), dims=list(dims_final.keys()), name="stitched")
    
    # Mapping dims to indices for stitched.values positional updates
    dim_to_idx = {dim: i for i, dim in enumerate(stitched.dims)}
    
    def update_stitched_pos(target_isel_dict, source_isel_dict, tile_idx, mode="set", weight_mask=None):
        """Position-based update of the stitched array using raw numpy access."""
        indices = [slice(None)] * stitched.ndim
        for d, s in target_isel_dict.items():
            if d in dim_to_idx:
                indices[dim_to_idx[d]] = s
        indices = tuple(indices)
        
        # Get source tile and ensure it's squeezed to matching 2D/3D shape
        tile_data = list_images.isel(stack_images=tile_idx, **source_isel_dict).values
        
        if mode == "set":
            stitched.values[indices] = tile_data
        elif mode == "max":
            if tile_data.size > 0:
                stitched.values[indices] = np.maximum(stitched.values[indices], tile_data)
        elif mode == "mean":
            if tile_data.size > 0:
                stitched.values[indices] = np.mean(np.stack([stitched.values[indices], tile_data]), axis=0)
        elif mode == "blend":
            if tile_data.size > 0 and weight_mask is not None:
                # Add weights for broadcasting: mask is (Y, X), tile_data might be (C, Y, X)
                if tile_data.ndim > weight_mask.ndim:
                    weight_mask = np.expand_dims(weight_mask, axis=0)
                
                stitched.values[indices] += (tile_data * weight_mask).astype(np.float32)
                weights.values[indices] += weight_mask.astype(np.float32)

    #######################################
    # Stitching 
    #######################################
    f_name = f"Stitching {os.path.basename(xml_file).split('.')[0]} Images"
    
    for i in tqdm(range(len(list_images.stack_images)), desc=f_name):
        y_pos, x_pos, c_pos, h_tile, w_tile = dict_pos[list_tiff[i]]
        begin_tile_x = actual_size_x * x_pos
        begin_tile_y = actual_size_y * y_pos
        
        if overlap_mode == "crop":
            # Zen uses absolute positions, Dragonfly uses index positions
            step_x = w_tile if metadata_format=="zeiss_zen" else actual_size_x
            step_y = h_tile if metadata_format=="zeiss_zen" else actual_size_y
            
            target_isel = dict(X=slice(begin_tile_x, begin_tile_x + step_x), 
                               Y=slice(begin_tile_y, begin_tile_y + step_y))
            if "C" in stitched.dims:
                target_isel["C"] = c_pos
            
            source_isel = dict(X=slice(0, step_x), Y=slice(0, step_y))
            update_stitched_pos(target_isel, source_isel, i, mode="set")
        
        elif overlap_mode == "blend":
            # Feathered blending with linear ramps
            end_tile_x = begin_tile_x + w_tile
            end_tile_y = begin_tile_y + h_tile
            
            # Create weight mask for this tile
            mask_y = np.ones(h_tile, dtype=np.float32)
            if overlap_y > 0:
                ramp_y = np.linspace(0, 1, overlap_y, dtype=np.float32)
                mask_y[:overlap_y] = ramp_y
                mask_y[-overlap_y:] = ramp_y[::-1]
            
            mask_x = np.ones(w_tile, dtype=np.float32)
            if overlap_x > 0:
                ramp_x = np.linspace(0, 1, overlap_x, dtype=np.float32)
                mask_x[:overlap_x] = ramp_x
                mask_x[-overlap_x:] = ramp_x[::-1]
                
            weight_mask = mask_y[:, np.newaxis] * mask_x[np.newaxis, :]
            
            target_isel = dict(X=slice(begin_tile_x, end_tile_x), 
                               Y=slice(begin_tile_y, end_tile_y))
            if "C" in stitched.dims:
                target_isel["C"] = c_pos
            
            source_isel = dict(X=slice(0, w_tile), Y=slice(0, h_tile))
            update_stitched_pos(target_isel, source_isel, i, mode="blend", weight_mask=weight_mask)
            
        else:
            # Max/Mean Blending
            end_tile_x = begin_tile_x + w_tile
            end_tile_y = begin_tile_y + h_tile
            
            # Center [overlap: -overlap]
            center_x = slice(begin_tile_x + overlap_x, end_tile_x - overlap_x)
            center_y = slice(begin_tile_y + overlap_y, end_tile_y - overlap_y)
            
            if center_x.start < center_x.stop and center_y.start < center_y.stop:
                target_isel = dict(X=center_x, Y=center_y)
                if "C" in stitched.dims:
                    target_isel["C"] = c_pos
                
                source_isel = dict(X=slice(overlap_x, w_tile - overlap_x), 
                                   Y=slice(overlap_y, h_tile - overlap_y))
                update_stitched_pos(target_isel, source_isel, i, mode=overlap_mode)

            # blending for edges
            if overlap_x > 0 or overlap_y > 0:
                # Left
                if overlap_x > 0:
                    update_stitched_pos(dict(X=slice(begin_tile_x, begin_tile_x + overlap_x), Y=slice(begin_tile_y, end_tile_y), C=c_pos if "C" in stitched.dims else None), 
                                        dict(X=slice(0, overlap_x)), i, mode=overlap_mode)
                # Right
                if overlap_x > 0:
                    update_stitched_pos(dict(X=slice(end_tile_x - overlap_x, end_tile_x),     Y=slice(begin_tile_y, end_tile_y), C=c_pos if "C" in stitched.dims else None), 
                                        dict(X=slice(w_tile - overlap_x, w_tile)), i, mode=overlap_mode)
                # Top
                if overlap_y > 0:
                    update_stitched_pos(dict(Y=slice(begin_tile_y, begin_tile_y + overlap_y), X=slice(begin_tile_x, end_tile_x), C=c_pos if "C" in stitched.dims else None), 
                                        dict(Y=slice(0, overlap_y)), i, mode=overlap_mode)
                # Bottom
                if overlap_y > 0:
                    update_stitched_pos(dict(Y=slice(end_tile_y - overlap_y, end_tile_y),      X=slice(begin_tile_x, end_tile_x), C=c_pos if "C" in stitched.dims else None), 
                                        dict(Y=slice(h_tile - overlap_y, h_tile)), i, mode=overlap_mode)

    if overlap_mode == "blend":
        logger.info("Normalizing blended image by weights")
        # Avoid division by zero
        stitched.values = stitched.values / np.maximum(weights.values, 1e-6)
        # Cast back to uint16
        stitched = stitched.clip(min=0, max=65535).astype(np.uint16)

    logger.info("Stitching Done!")
    logger.info(f"Stitched array value range: {stitched.min().item()} to {stitched.max().item()}")
    
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

    if gaussian_sigma > 0:
        logger.info(f"Applying Gaussian filter with sigma={gaussian_sigma}")
        # Apply Gaussian filtering to each channel if present
        if "C" in stitched.dims:
            for c in stitched.C:
                chan_data = stitched.sel(C=c).values
                filtered = gaussian(chan_data, sigma=gaussian_sigma, preserve_range=True)
                stitched.loc[dict(C=c)] = filtered.astype(np.uint16)
        else:
            filtered = gaussian(stitched.values, sigma=gaussian_sigma, preserve_range=True)
            stitched.values = filtered.astype(np.uint16)

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
                photometric='minisblack',
            )

            for level in range(subresolutions):
                mag = 2 ** (level + 1)
                down = data[..., ::mag, ::mag]
                tif.write(
                    down,
                    description=ome_xml,
                    subfiletype=1,
                    resolution=(1 / mag / pixelsize, 1 / mag / pixelsize),
                    photometric='minisblack',
                )

            # Write thumbnail as a separate series
            thumb_dims = tuple(ax for i, ax in enumerate(stitched.dims) if ax not in ["X", "Y"])
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
        "--overlap_mode", type=str, choices=["max", "mean", "crop", "blend"], default="blend",
        help="Stitch using hard cropping (instead of max blending)."
    )
    parser.add_argument("--disable_logger", action="store_true",
        help="Whether or not to use the logger, useful to deactivate when multiple images are stitched in parallel")
    parser.add_argument("--metadata_format", type=str, choices=["dragonfly", "zeiss_zen"], default="dragonfly",
        help="Format of the metadata and tiles (dragonfly or zeiss_zen)")
    parser.add_argument("--gaussian_sigma", type=float, default=0.0,
        help="Sigma for Gaussian filtering at the end of stitching (0 to disable)")
    parser.add_argument("--normalize", action="store_true",
        help="Normalize intensity between tiles")
    parser.add_argument("--split_by_scene", action="store_true",
        help="Stitch tiles separately for each scene (s index in filename)."
    )

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
        overlap_mode=args.overlap_mode,
        disable_logger=args.disable_logger,
        offset_channel=None,  # Future implementation
        attrs = args.attrs,
        metadata_format=args.metadata_format,
        gaussian_sigma=args.gaussian_sigma,
        normalize_intensity=args.normalize,
        split_by_scene=args.split_by_scene,
    )

if __name__ == "__main__":
    main()
