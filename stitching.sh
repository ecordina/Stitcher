#!/bin/bash
#SBATCH --job-name=Stitching # Specify the number of GPUs needed per job
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --mem=50g
# mamba activate stitcher
FOLDER="C:\Users\Emman\Desktop\work\These\Data\Camille\090426\Fibres2"
XML_FILE=""
# Default Arguments
OUTPUT_PATH="C:/Users/Emman/Desktop/work/These/Code/Stitcher/data/output/04092026"
SAVE_FORMAT="tif"

# Optional flags
IMAGE_PREFIX="" 		   # In case the image prefix is different than the one in the metadata /!\, has to be complete as the code will look for image_prefix+"^_F\d+$"
          # ie: For MP_20250411-C1_14.38.46_F000.ims, IMAGE_PREFIX = "MP_20250411-C1_14.38.46" 
MAX_PROJECT="--max_project"        # Add --max_project to Z-Max Projection
OVERLAP="10%" 			   # Leave Empty to scrap overlap from the xml, else specify the overlap
REFINE_OVERLAP=""  # Take it off if you want to force either the given overlap value or the overlap from the xml file
OVERLAP_MODE="max"      # Leave empty to use max projection on the overlap, add crop if you want to use crop overlap, max if you want to max project it, mean if you want to mean project it, blend if you want feathered blending, default is blend
flat_field_path=""		   # Flat Field Image if you want to perform Flat-Field Correction, leave empty if not. Be careful that the Flat Field Tile has to be the same dimensions as the image tiles
NO_BACKGROUND="--no_background_subtraction"                   # Leave empty to enable background subtraction, else add --no_background_subtraction
attrs="{'Instrument':'5x'}" # Attributes to attach to the zarr, only used when save_format == zarr
METADATA_FORMAT="zeiss_zen"    # dragonfly or zeiss_zen
gaussian_sigma='0'
logger="" 			   # Add --disable_logger to disable logger
NORMALIZE=""        # Add --normalize to normalize intensity between tiles
SPLIT_BY_SCENE='' # Add --split_by_scene to stitch scene separately
# Run the Python script
python ./src/stitching.py \
  --folder "$FOLDER" \
  --xml_file "$XML_FILE" \
  --output_path "$OUTPUT_PATH" \
  --overlap "$OVERLAP" \
  --attrs "$attrs" \
  --image_prefix "$IMAGE_PREFIX" \
  --flat_field_path "$flat_field_path" \
  --save_format "$SAVE_FORMAT" \
  --metadata_format "$METADATA_FORMAT" \
  --gaussian_sigma $gaussian_sigma \
  --overlap_mode "$OVERLAP_MODE" \
  $MAX_PROJECT \
  $REFINE_OVERLAP \
  $NO_BACKGROUND \
  $NORMALIZE \
  $logger \
  $SPLIT_BY_SCENE
