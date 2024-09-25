ANNTROOT="../baselines/VLN-DUET/datasets/R2R/annotations"
SUBMITROOT="../baselines/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-vitbase-seed.0-init.aug.45k/preds"
OUTROOT="./outs/duet"
SCANDIR= [fill in your dir]

python eval.py \
--annotation_root $ANNTROOT \
--submission_root $SUBMITROOT \
--out_root $OUTROOT \
--scans_dir $SCANDIR
