import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from stitching import find_translation, crop_zero_borders, stitch_dragonfly_tiles
import xarray as xr
import os

# ---------- Utils Function ----------
def generate_tile(base_value, shape=(20, 20), overlap=(10, 10)):
    """Create a synthetic image tile with a unique pattern in the center."""
    img = np.zeros(shape, dtype=np.uint8)
    img[overlap[0]:-overlap[0], overlap[1]:-overlap[1]] = base_value
    return img

def generate_dragonfly_xml(tile_shape=(20, 20), overlap=(10, 10), grid_shape=(2, 2)):
    """
    Generates XML metadata mimicking Dragonfly tile positions.
    Each tile overlaps its right and bottom neighbor by `overlap` pixels.
    """
    from xml.etree.ElementTree import Element, SubElement, tostring
    import xml.dom.minidom

    tile_width, tile_height = tile_shape[1], tile_shape[0]
    x_overlap, y_overlap = overlap[1], overlap[0]
    
    root = Element("ImageTiles")

    tile_id = 0
    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            x_pos = col * (tile_width - x_overlap)
            y_pos = row * (tile_height - y_overlap)
            tile = SubElement(root, "ImageTile", ID=str(tile_id))
            SubElement(tile, "PositionX").text = str(float(x_pos))
            SubElement(tile, "PositionY").text = str(float(y_pos))
            SubElement(tile, "PositionZ").text = "0.0"
            tile_id += 1

    xml_str = xml.dom.minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    return xml_str


# ---------- Test: find_translation ----------
def test_find_translation_identity():
    image = np.random.rand(128, 128)
    shift = find_translation(image, image)
    np.testing.assert_allclose(shift, (0.0, 0.0), atol=1e-3)

def test_find_translation_shift():
    image = np.zeros((128, 128))
    image[32:96, 32:96] = 1.0
    shifted_image = np.roll(np.roll(image, 5, axis=0), -7, axis=1)
    shift = find_translation(image, shifted_image)
    np.testing.assert_allclose(shift, (-5.0, 7.0), atol=1.0)

# ---------- Test: crop_zero_borders ----------
def test_crop_zero_borders_2d():
    image = np.zeros((10, 10))
    image[2:8, 3:9] = 1
    cropped = crop_zero_borders(image)
    assert cropped.shape == (6, 6)
    assert np.all(cropped == 1)

def test_crop_zero_borders_3d():
    image = np.zeros((3, 10, 10))
    image[:, 2:8, 3:9] = 1
    cropped = crop_zero_borders(image,col_axis=1,row_axis=2)
    assert cropped.shape == (3, 6, 6)
    assert np.all(cropped == 1)

def test_crop_zero_borders_invalid():
    with pytest.raises(ValueError):
        crop_zero_borders(np.zeros((1, 1, 1, 1)))

# # ---------- Test: stitch_dragonfly_tiles ----------
# def test_stitch_four_tiles_with_overlap_and_xml():
#     tile_shape = (20, 20)
#     overlap = (10, 10)
    
#     tiles = [
#         generate_tile(50, tile_shape, overlap),
#         generate_tile(100, tile_shape, overlap),
#         generate_tile(150, tile_shape, overlap),
#         generate_tile(200, tile_shape, overlap),
#     ]
    
#     data = np.stack(tiles).reshape(2, 2, *tile_shape)
#     arr = xr.DataArray(data, dims=("row", "col", "y", "x"))

#     xml = generate_dragonfly_xml(tile_shape=tile_shape, overlap=overlap, grid_shape=(2, 2))

#     # Save XML to disk or parse directly if stitch_dragonfly_tiles takes XML input
#     stitched = stitch_dragonfly_tiles(arr, xml_metadata=xml)

#     expected_shape = (
#         tile_shape[0] * 2 - overlap[0],
#         tile_shape[1] * 2 - overlap[1],
#     )
#     assert stitched.shape == expected_shape

# def test_stitch_dragonfly_tiles_minimal(tmp_path):
#     # This is a minimal mock test to ensure function runs with basic args
#     xml_path = tmp_path / "metadata.xml"
#     image_path = tmp_path / "tile_001.tiff"
    
#     # Create mock XML and image
#     xml_path.write_text('''
#         <Experiment>
#             <dimensions stack_rows="1" stack_columns="1" />
#             <Stack ROW="1" COL="1" IMG_REGEX="tile_001.tiff" />
#         </Experiment>
#     ''')
#     from tifffile import imwrite
#     dummy_image = np.ones((1, 1, 64, 64), dtype=np.uint16)
#     imwrite(image_path, dummy_image)

#     # Run stitching with mocked input
#     stitch_dragonfly_tiles(
#         folder=str(tmp_path),
#         xml_file=str(xml_path),
#         output_path=str(tmp_path),
#         max_project=True,
#         background_subtraction=False,
#         flat_field_path='',
#         max_overlap=True,
#         crop_overlap=False,
#         disable_logger=True
#     )

#     # Check that log or result exists
#     log_files = list(tmp_path.glob("*.log"))
#     assert log_files, "Expected log file not created"
