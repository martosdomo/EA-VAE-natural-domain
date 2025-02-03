#!/bin/sh

# Downloads training and test datasets.
# invoke as `sh download_datasets.sh`
# Needs GNU/Linux to work.

# download to the directory of this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

DATASET_DIR="$SCRIPTPATH/../datasets/textures"

# makedir if not exists
mkdir -p "$DATASET_DIR"

# download fakelabeled_natural_commonfiltered_640000_20px.pkl
#wget -c -O "$DATASET_DIR/fakelabeled_natural_commonfiltered_640000_20px.pkl" 'https://zenodo.org/record/6644895/files/fakelabeled_natural_commonfiltered_640000_20px.pkl?download=1'

# download fakelabeled_natural_commonfiltered_640000_40px.pkl
wget -c -O "$DATASET_DIR/fakelabeled_natural_commonfiltered_640000_40px.pkl" 'https://zenodo.org/records/10245168/files/fakelabeled_natural_commonfiltered_640000_40px.pkl?download=1'

# download labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_20px.pkl
#wget -c -O "$DATASET_DIR/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_20px.pkl" 'https://zenodo.org/record/6644895/files/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_20px.pkl?download=1'

# download labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_40px.pkl
#wget -c -O "$DATASET_DIR/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_40px.pkl" 'https://zenodo.org/record/6644895/files/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_40px.pkl?download=1'

