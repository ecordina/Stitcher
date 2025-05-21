#!/bin/bash
#SBATCH --job-name=Stitching # Specify the number of GPUs needed per job
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --mem=50g

# Arguments
FOLDER=""
XML_FILE=""

# Default Arguments
OUTPUT_PATH="./" #Save in folder where the bash script is
SAVE_FORMAT="tiff" # default to 'tiff' if not specified

# Optional flags
MAX_PROJECT="--max_project"  # add --max_project to Z-Max Projection
OVERLAP="" #Leave Empty to scrap overlap from the xml
REFINE_OVERLAP="--refine_overlap" # take it off if you want to force either the given overlap value or the overlap from the xml file
CROP_OVERLAP="--crop_overlap" # leave empty to use max projection on the overlap, add --crop_overlap if you want to use crop overlap
NO_BACKGROUND=""          # leave empty to enable background subtraction
attrs="{'Instrument':'Dragonfly'}" # Attributes to attach to the zarr, only used when save_format == zarr
logger="--disable_logger" # Add --disable_logger to disable logger

# Run the Python script
python ./src/stitching_final.py \
  --folder "$FOLDER" \
  --xml_file "$XML_FILE" \
  --output_path "$OUTPUT_PATH" \
  --overlap "$OVERLAP" \
  --attrs "$attrs" \
  $MAX_PROJECT \
  $REFINE_OVERLAP \
  $CROP_OVERLAP \
  $NO_BACKGROUND \
  $logger \
  --save_format "$SAVE_FORMAT"
