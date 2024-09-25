ANNTROOT=""   # The path to the directory of where all the R2R_{split}.json files located
SUBMITROOT="" # The root directory of the submission files generated by each method
OUTROOT=""    # The directory where the results.json will be dumped
SCANDIR=""    # The directory to scans of matterport3d

python eval.py \
--annotation_root $ANNTROOT \
--submission_root $SUBMITROOT \
--out_root $OUTROOT \
--scans_dir $SCANDIR
