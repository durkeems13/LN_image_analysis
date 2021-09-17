##Pipeline 1

Pipeline 1 generates predictions on test set of images and saves to .pkl files

Pipeline 1 will also generate image overlays of predictions and restructure .pkl files for downstream analysis

If desired, binary masks can also be generated for predictions

## Changes that need to be made to analysis code

# for pipeline 1:
- inputs are: train log name; test set image folder
- predict.py --- line 8, 552
- predict.sbatch --- line 23
- make_final_overlays_and_stage_features.py --- line 23
- make_final_overlays_and_stage_features.sbatch --- line 17
- make_rgb_seg_images.py --- lines 86,89,82 set to 0 to predict single class; line 63 to go between 2 ch and 3 ch
 


