#!/bin/sh

#SBATCH -p broadwl
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --output=MRhybrid_on_MR_testset.out
#SBATCH --error=MRhybrid_on_MR_testset.err
#SBATCH --job-name=test_set

module load use.deprecated horovod

export GT_PKLS='../data/manual_pkls/MR_GT'
export PRED_PKLS='../analysis/MRhybrid_on_MRtestset/processing/combined_inference_pkls'
export CSVNAME='cdm3_analysis_pkls/MRhybrid_on_MRtestset.csv'

python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.0
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.1
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.2
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.3
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.4
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.5
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.6
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.7
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.8
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
python get_prediction_features_joint.py --gt_pkls=$GT_PKLS --pkls_read=$PRED_PKLS --csv_name=$CSVNAME --threshold=0.9
python seg_metrics_paraffin_ss_joint.py --csv_name=$CSVNAME
