#!/bin/sh

export trainlogdir=$1
export pred_ims=$2
export rootdir=../../$trainlogdir/$pred_ims
export model_load=../../../train_logs/$trainlogdir/checkpoint
export predict_dir=../../../data/$pred_ims #Lupus_512_tiffstacks_separate_sort/$pred_ims
export inference_output_dir=../../$trainlogdir/$pred_ims/inference/untiled_biopsy_predictions
export overlay_dir=../../$trainlogdir/$pred_ims/processing/final_overlays
export feature_pkl_dir=../../$trainlogdir/$pred_ims/processing/feature_stage_pkls
export rgb_dir=../../$trainlogdir/$pred_ims/processing/final_overlays_rgb
SECONDS=0

if [ ! -d "$trainlogdir" ]; then
    mkdir "$trainlogdir"
fi

if [ ! -d "$trainlogdir"/"$pred_ims" ]; then
    mkdir "$trainlogdir"/"$pred_ims"
    mkdir "$trainlogdir"/"$pred_ims"/{inference,analysis,processing}
fi

if [ ! -d "$trainlogdir"/"$pred_ims"/inference/untiled_biopsy_predictions ]; then
    mkdir "$trainlogdir"/"$pred_ims"/inference/untiled_biopsy_predictions 
fi

if [ ! -d "$trainlogdir"/"$pred_ims"/processing/feature_stage_pkls ]; then
    mkdir "$trainlogdir"/"$pred_ims"/processing/feature_stage_pkls
fi

if [ ! -d "$trainlogdir"/"$pred_ims"/processing/final_overlays ]; then
    mkdir "$trainlogdir"/"$pred_ims"/processing/final_overlays
fi

if [ ! -d "$trainlogdir"/"$pred_ims"/processing/final_overlays_rgb ]; then
    mkdir "$trainlogdir"/"$pred_ims"/processing/final_overlays_rgb 
fi

cd Tcell_code/inference

echo "Predictions running"
#assign-gpu -n 1 python3 predict.py --index=0 --load=$model_load --predict=$predict_dir --output_dir=$inference_output_dir --tasks=1
assign-gpu -n 1 python3 predict_512.py --index=0 --load=$model_load --predict=$predict_dir --output_dir=$inference_output_dir --tasks=1
echo "Predictions done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""

cd ../processing

echo "Final overlays running"
python3 make_final_overlays_and_stage_features.py --pkls_read=$inference_output_dir --ims_read=$predict_dir --overlays_write=$overlay_dir --pkls_write=$feature_pkl_dir
python3 indices_tofile.py --folder=$feature_pkl_dir > $rootdir/processing/indices.txt
echo "Final overlays done"
echo ""

echo "RGBs running"
python3 make_rgb_seg_images.py --read=$overlay_dir --write=$rgb_dir
echo "RGBs done"
echo ""

cd ../../
echo "Pipeline 1 done"
