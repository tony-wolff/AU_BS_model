# Script to extract OpenFace features in order to create a dataset, that takes AU as inputs and outputs blend shapes
# This script executes FeatureExtraction from a docker image, it extracts  3D landmarks in mm, heapose, gaze, and action units

# Gather all participants from the json file using jq
USERS=$(cat user_data_valid.json | jq -r ".uuid")
# Get all the uuid, the folders containing the data are named after them
UUID=$(echo $USERS | jq -r 'to_entries | map(.value)[]')
# Shortcut for the openface docker script
FeatureExtraction=/home/openface-build/build/bin/FeatureExtraction
outdir=/myDataFolder/

# Create an array, if the folder exists, add to the array
# Some uuids have no folders attached to them, see the json file
FOLDER=()
for user in $UUID; do
	if  [ -d "$user" ]; then
                FOLDER+=($user)
	fi
done

# Iterate over all folders that contains mp4 videos
# for all videos, apply openface featureExtraction script (get 3D landmarks in mm, heapose, gaze, action units)
# store result inside the outdir folder with the name of the folder in it
for f in "${FOLDER[@]}"; do
	videos=$(find "${f}" -maxdepth 1 -type f -name "*.mp4")
	for v in $videos; do
		$FeatureExtraction -out_dir $outdir$f -f $v -3Dfp -pose -gaze -aus
	done
done
