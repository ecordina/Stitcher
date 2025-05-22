#!/bin/bash
#SBATCH --job-name=Stitching
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH --mem=50g

# Required input folder (must be passed as a parameter to sbatch)
FOLDER="./"
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
logger="--disable_logger" # Take off to activate Logger
IMAGE_PREFIX="" # In case the image prefix is different than the one in the metadata /!\, has to be complete as the code will look for image_prefix+"_F\d+"
		# ie: For MP_20250411-C1_14.38.46_F000.ims, IMAGE_PREFIX = "MP_20250411-C1_14.38.46" 
# Run the Python script
# Loop over all XML files in the folder
for XML_FILE in "$FOLDER"/*.xml; do
  if [[ -f "$XML_FILE" ]]; then
    echo "Processing XML: $XML_FILE"

    python ./src/stitching_final.py \
      --folder "$FOLDER" \
      --xml_file "$XML_FILE" \
      --output_path "$OUTPUT_PATH" \
      --overlap "$OVERLAP" \
      --attrs "$attrs" \
      --image_prefix "$IMAGE_PREFIX" \
      $MAX_PROJECT \
      $REFINE_OVERLAP \
      $CROP_OVERLAP \
      $NO_BACKGROUND \
      $logger \
      --save_format "$SAVE_FORMAT" &  ## Take Off the "&" if you don't want to run in paralell
    else
      echo "No XML files found in $FOLDER."
    fi
done
wait