import numpy as np
import pytest
import sys
import os
import tempfile
from pathlib import Path
from tifffile import imwrite
from xml.etree.ElementTree import Element, SubElement, ElementTree
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from stitching import stitch_dragonfly_tiles

# ---------- Utility Function ----------

def create_tiles_and_xml(
    tile_shape=(20, 20),
    overlap=(10, 10),
    grid_shape=(2, 2),
    output_dir=None,
    prefix="tile",
    pixel_size=(0.6, 0.6),  # (H, V) in microns
    mechanical_displacement=(1111.9, -1111.9)  # (H, V)
):
    """
    Creates synthetic tiles and a TeraStitcher-style XML metadata file.
    
    Returns:
        tuple: (tile_dir: Path, xml_path: Path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_dir = output_dir / "tiles"
    tile_dir.mkdir(exist_ok=True)

    root = Element("TeraStitcher", volume_format="TiledXY|3Dseries")
    SubElement(root, "stacks_dir", value=".")
    SubElement(root, "ref_sys", ref1="-2", ref2="1", ref3="3")
    SubElement(root, "voxel_dims", D="1", H=str(pixel_size[0]), V=str(-pixel_size[1]))
    SubElement(root, "origin", D="0.0", H="0.0", V="0.0")
    SubElement(root, "mechanical_displacements", H=str(mechanical_displacement[0]), V=str(mechanical_displacement[1]))
    SubElement(root, "dimensions",
               stack_columns=str(grid_shape[1]),
               stack_rows=str(grid_shape[0]),
               stack_slices="-1")
    SubElement(root, "subimage", resolution="0", timepoint="0")

    stacks_elem = SubElement(root, "STACKS")
    tile_id = 0
    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            value = (tile_id + 1) * 50
            tile = np.full(tile_shape, value, dtype=np.uint16)
            filename = f"{prefix}_{tile_id:03d}.tif"
            filepath = tile_dir / filename
            imwrite(filepath, tile)

            # Compute absolute position in microns
            abs_H = col * (tile_shape[-1] - overlap[1])
            abs_V = row * (tile_shape[-2] - overlap[0])

            stack_elem = SubElement(
                stacks_elem,
                "Stack",
                ABS_D="0",
                ABS_H=str(int(abs_H)),
                ABS_V=str(int(abs_V)),
                COL=str(col),
                ROW=str(row),
                DIR_NAME=".",
                IMG_REGEX=filename,
                N_BYTESxCHAN="1",
                N_CHANS="1",
                STITCHABLE="no"
            )
            SubElement(stack_elem, "NORTH_displacements")
            SubElement(stack_elem, "EAST_displacements")
            SubElement(stack_elem, "SOUTH_displacements")
            SubElement(stack_elem, "WEST_displacements")

            tile_id += 1

    xml_path = output_dir / "tiles.xml"
    ElementTree(root).write(xml_path, encoding="UTF-8", xml_declaration=True)

    return tile_dir, xml_path


# ---------- Test ----------
# tile_shape,max_project,offset_channel = (10,512,512),False,None
# background_subtraction = True
# refine_overlap = True
# save_format = "tif"
# crop_overlap = True
# max_overlap = False
@pytest.mark.parametrize("tile_shape,max_project,offset_channel", [ ((64,64),False,None),   #{"X":3,"Y":4}),
                                                                    ((10,64,64),False,None),     #{"Z":2,"X":3,"Y":4}),
                                                                    ((10,64,64),True,None),      #{"Z":2,"X":3,"Y":4}),
                                                                    ((2,10,64,64),True,None),    #{"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((2,10,64,64),False,[2,2]),  #{"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((2,10,64,64),True,2),       #{"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((2,10,64,64),False,None),   #{"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,2,10,64,64),False,None), #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,2,10,64,64),True,None),  #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,2,10,64,64),True,[2,2]), #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,2,10,64,64),True,2),     #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,2,10,64,64),False,[2,2]),#{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,2,10,64,64),False,2),    #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,1,10,64,64),False,None), #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,1,10,64,64),False,[2,2]),#{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,1,10,64,64),False,2),    #{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ((3,1,10,64,64),True,None),#{"Cycle":0,"Channel":1,"Z":2,"X":3,"Y":4}),
                                                                    ])
@pytest.mark.parametrize("background_subtraction", [True, False])
@pytest.mark.parametrize("refine_overlap", [True, False])
@pytest.mark.parametrize("save_format", ["tif",'zarr'])  # Add more formats if needed
@pytest.mark.parametrize("crop_overlap,max_overlap", [(True,False), (False,True)])
def test_stitch_args_combinations(
    tile_shape, max_project,
    background_subtraction, save_format,
    max_overlap, crop_overlap,refine_overlap,offset_channel
):
    with tempfile.TemporaryDirectory() as tmpdir:
        attrs = {'Instrument':'Dragonfly'}
        tmp_path = Path(tmpdir)
        tile_dir, xml_path = create_tiles_and_xml(
            tile_shape=tile_shape,
            overlap=(10, 10),
            grid_shape=(3, 3),
            output_dir=tmp_path
        )

        stitch_dragonfly_tiles(
            folder=str(tile_dir),
            xml_file=str(xml_path),
            output_path=str(tmp_path),
            overlap = "",
            attrs = attrs,
            image_prefix = '',
            flat_field_path='',
            max_project=max_project,
            background_subtraction=background_subtraction,
            save_format=save_format,
            max_overlap=max_overlap,
            crop_overlap=crop_overlap,
            disable_logger=True,
            dims = {"T":0,"C":1,"Z":2,"X":3,"Y":4},
            tile_size = None,
            refine_overlap = refine_overlap,
            offset_channel = offset_channel,
        )

        assert list(tmp_path.glob("thumbnail*.png")), "Expected thumbnail not created"
        assert list(tmp_path.glob(f"*.{save_format}")), f"Expected {save_format} not created"
# test_stitch_args_combinations(
#     tile_shape, max_project,
#     background_subtraction, save_format,
#     max_overlap, crop_overlap,refine_overlap,offset_channel
# )