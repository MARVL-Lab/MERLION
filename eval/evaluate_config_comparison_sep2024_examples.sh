

# High visibility dataset (BK_clear)

## MERLION

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/automatic_summary/BK/Bonaire-kralen-gopro_456_256_70_image_features_frame_number.txt" --benchmark_dataset="BK_clear" --labels="bkclear_v15_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

## ROST (take average of 5) - trial 3 was invalid (did not finish)

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_BKClear_1_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-10_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="BK_clear" --labels="bkclear_v15_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_BKClear_2_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-10_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="BK_clear" --labels="bkclear_v15_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_BKClear_4_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-10_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="BK_clear" --labels="bkclear_v15_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_BKClear_5_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-10_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="BK_clear" --labels="bkclear_v15_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_BKClear_6_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-10_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="BK_clear" --labels="bkclear_v15_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

echo Done
echo .

## Moderate visibility (GOPRO)


# MERLION

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/automatic_summary/gopro/50_merlion/Gopro_456_256_50_image_features_frame_number.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments

# MERLION-E

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/automatic_summary/gopro/7070/DM_Gopro_456_256_70_70_image_features_frame_number.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments

## ROST (take average of 5)

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_murkGoPro_1_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_murkGoPro_2_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_murkGoPro_3_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_murkGoPro_4_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_murkGoPro_5_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments

## ROST pre-enhanced (take average of 5)

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_DMGoPro_1_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_DMGoPro_2_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_DMGoPro_3_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_DMGoPro_4_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/Set2_ICRA/ROST_DMGoPro_5_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-10_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv" && \ # some comments


echo Done
echo .



# Low visibility dataset (Powells)

## Low visibility MERLION

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/automatic_summary/lowvis/40/frame_number.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

## Low visibility MERLION-E

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/automatic_summary/lowvis/40_70/frame_number.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

## Low visibility ROST (take the average from 5 trials)

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_Sep13-A_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_Sep13-B_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_Sep13-C_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_Sep13-D_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_Sep13-E_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

## Low visibility ROST pre-enhanced (take the average from 5 trials)

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/DM-PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_DM_Sep13-A_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/DM-PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_DM_Sep13-B_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/DM-PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_DM_Sep13-C_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/DM-PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_DM_Sep13-D_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/DM-PowellsCay-Trimmed-GreenRemoved/PowellsCay_ROST_DM_Sep13-E_ntpcs-100_sumsze-6_scl-1.000000_sub-1_batchFq-1_rszeH-_rszW-/frames-1based.txt" --benchmark_dataset="Powells_murky" --labels="powells_sep2024" --output_csv="eval_results/Sep_14_icra.csv"

echo Done
echo .

### ICRA Video

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/ICRA_Video/ROST_GoPro_Enhanced_for_video_456256_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-1_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra_2.csv" && \ # some comments

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/ICRA_Video/ROST_GoPro_unenhanced_for_video_456256_ntpcs-100_sumsze-6_scl-1.000000_sub-2_batchFq-1_rszeH-256_rszW-456/frames-1based.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra_2.csv" && \ # some comments

python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/fps5-v02-aug28/fps5-v02-aug28-frames.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra_2.csv" && \ # some comments

echo Done
echo .

