# 📌 MERLION : Marine ExploRation with Language guIded Online iNformative Visual Sampling and Enhancement

MERLION, a novel framework that provides semantically aligned and visually enhanced summaries for murky underwater marine environment monitoring and exploration.


* Paper: S. V. Thengane, M. B. Prasetyo, Y. X. Tan, and M. Meghjani. "MERLION: Marine ExploRation with Language guIded Online iNformative Visual Sampling and Enhancement," _2025 IEEE International Conference on Robotics and Automation (ICRA), Atlanta, Georgia, 2025_. Preprint: [https://arxiv.org/abs/2503.06953](https://arxiv.org/abs/2503.06953)
* Video: [https://www.youtube.com/watch?v=KdlEqcLH9rY](https://www.youtube.com/watch?v=KdlEqcLH9rY) and [https://www.youtube.com/watch?v=o1hDcecdX5g](https://www.youtube.com/watch?v=o1hDcecdX5g)

![MERLION architecture](./assets/MERLION_3Mar.png?raw=true)

This branch is the desktop version of MERLION. For the version for AUVs (Autonomous Underwater Vehicles), please check the _ros_ branch, which will be available at a later date.
 
## Installing core modules

Create a conda environment or a python environment otherwise.

Clone this repository.

```sh
git clone https://github.com/MARVL-Lab/MERLION
```

__Install DM_underwater for Visual enhancement module:__

Clone the repository within the MERLION folder

```sh
git clone https://github.com/piggy2009/DM_underwater.git
cd MERLION
```

Install the repository
- Install necessary Python packages from requirement.txt
- Download a [model checkpoint](https://drive.google.com/file/d/1As3Pd8W6XmQBU__83iYtBT5vssoZHSqn/view?usp=sharing).
- To enhance any folder of images, add the path to the folder into the config file (check instructions on the repository)
- Visual enhancement can be run via the script `infer.py`


__Install CLIP:__

Conda installation:

```sh
conda activate {your_conda_environment}
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Python venv installation is similar to the above.

Your first use of the CLIP model in this project will require an internet connection as CLIP needs to download a checkpoint. You may modify the chosen model under `model, preprocess = clip.load("ViT-B/32", device=device)` in `merlion.py`. Apart from this, you will not require an internet connection when running MERLION.

__Install missing packages__

Run the scripts merlion.py and summarizer.py and see if you still have any missing packages and install them:

```sh
pip install json
```

## Preparing video datasets

Prepare the videos you would want to use as input as images (we converted to 256x456 resolution)

```sh
ffmpeg -i "/path/to/output/video.mp4" -vf "scale=256:456" -pix_fmt yuv420p -c:v libx264 "/path/to/frames/%5d.png"
```

Adjust the parameters as needed ("-pix_fmt yuv420p -c:v libx264" may not be needed)

You should have a folder with images named in order

The following publicly available videos are the sources we used for our dataset. You may use any other video sources for MERLION as MERLION is generalizable.

__Low visibility dataset:__

Diving Powells Cay reefs on another murky bumpy day. (January 4, 2022). Reflections DC Private Cruises and Sailing Classes. URL: [https://www.youtube.com/watch?v=5PiZJwWxIJw](https://www.youtube.com/watch?v=5PiZJwWxIJw)


__Moderate visibility dataset:__

GoPro Hero 5 - Bonaire-Buddy Reef Solo Dive Part 2 - RAW. (December 9,2016). Convergence Precision Props. URL: [https://www.youtube.com/watch?v=aejouVv8n](https://www.youtube.com/watch?v=aejouVv8n)


__High visibility dataset:__

Bonaire - Kralendijk - GoPro Hero 4 Silver - Best Snorkeling and Scuba Diving. (February 15, 2015). louis edwar rodriguez benavides. URL: [https://www.youtube.com/watch?v=SiNj0Av9Zqk](https://www.youtube.com/watch?v=SiNj0Av9Zqk)


For ready-made exemplars of the above data (unenhanced and enhanced) you may download from [OneDrive](https://sutdapac-my.sharepoint.com/:f:/g/personal/malika_meghjani_sutd_edu_sg/EuXGZj9K0MdIv69IxOwj87IBhwG_LcRG2SbPgjSL4_iACg?e=ZoCmg1): 

Use the password provided below (without spaces)
```
marvl merlion 2025
```

## Running MERLION

### MERLION

For plain MERLION without visual enhancement run the following command in your conda / python environment:

```sh
python merlion.py --video_frame_path="/path/to/frames" --summary_size=6 --save_summary_to="/path/to/save" --threshold=0.5
```

`threshold` must be between 0 and 1.0 (exclusive), and 0.5 refers to 50% threshold as detailed in the paper. Check out the script for other arguments such as custom text prompts.

### MERLION-E

Run the below command for presampling and save the presampled frames (first phase)

```sh
python presampling.py --video_frame_path="/path/to/frames" --save_selected_to="/path/to/save" --threshold=0.5
```

Please note that the special parameter name is "save_selected_to" and not "save_summary_to".

Presampled frames will be saved under the folder specified by the argument "save_selected_to". After presampling, use the DM_underwater visual enhancement model to enhance the presampled frames.

Once you are done with enhancement, run this command to obtain the final summary.

```sh
python merlion.py --video_frame_path="/path/to/enhanced/presampled/frames" --summary_size=6 --save_summary_to="/path/to/save" --threshold=0.7
```

Following the paper, the threshold for final sampling can be higher such as 0.7.

## Evaluation

To evaluate the results obtained, run the file `eval/evaluate_config_comparison.py` (you may rename it thus). For examples of running it, see `eval/evaluate_config_comparison_sep2024_examples.sh` . Follow the steps below to evaluate properly.

### Human benchmark

You will need a human benchmark for each dataset, that is, a list of lists containing the frame indices of frames chosen by human evaluators as benchmark. We have provided and coded the human benchmark for the aforementioned example datasets in the evaluation script.

### Numerical frame index format summary files

To supply the input summary files to the evaluation script, we will need to prepare them.
- First create a summary text file for each result which you would like to evaluate, containing the filepaths or filenames corresponding to the chosen frames. 
- Convert these summary text files from filepath format (list of filepaths) to frame index format (list of frame indices) corresponding to the order of images in the video starting from 1 (the first frame), and stripped of 0's (do not pad with zeroes). 
  - This facilitates the calculation of time representativeness score with respect to a given benchmark ("ground truth"). 
  - One way to do this is to create a correspondence file which pairs each filepath with its frame index and use this for conversion.

### Labels

You will also need semantic labels if you are using the semantic evaluation method. Check the given labels under eval/labels/ for the format. It is a dictionary which contains a list of semantic labels of relevant objects (e.g. fish species) as a value for each frame index (the key). All possible video frame indices need to have labels, and there are ways to generate this dictionary with optimal effort.

We have provided the labels for the example datasets.

Without these labels, you can only use the `eval_one2many` or `eval` method which is based on frame index / time difference.

### Running

Once you have both the human benchmark lists and the summary files in the appropriate format, you may run the evaluation script. An example is:

```sh
python evaluate_config_comparison.py --eval_mode "eval_semantic_not_remove" --fps=30 --threshold=2 --weight=0.5 --second_path="$PWD/experiments/September2024-ICRA-1/automatic_summary/gopro/50_merlion/Gopro_456_256_50_image_features_frame_number.txt" --benchmark_dataset="gopro_sep24_enhanced" --labels="gopro_sep2024" --output_csv="eval_results/Sep_14_icra.csv"
```

The time representativeness decay threshold we use is 2 seconds which is standard. This may be adjusted depending on the speed of the video.

### Interpretation

The (absolute) scores output by the evaluation script are between 0 and 1.0 and correspond to that summary's semantic and (time) representativeness correlation with all the human-chosen summaries in the chosen benchmark. 

Relative scores are percentages relative to the average human score of the human benchmark (each human benchmark is pitted against the rest and then averaged). 
- Not even the human benchmarks will ever obtain a perfect score of 1.0, this is why the relative scores are more meaningful.

## Citing MERLION

If you use MERLION, please consider getting in touch and letting us know. It will put a smile on our face :) . It would also be great to cite us via the following BibTeX entry or otherwise :D


```bibtex
@INPROCEEDINGS{thenganeprasetyo2025merlionmarineexplorationlanguage,
      author={Thengane, Shrutika Vishal and Prasetyo, Marcel Bartholomeus and Tan, Yu Xiang and Meghjani, Malika},
      booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
      title={{MERLION: Marine ExploRation with Language guIded Online iNformative Visual Sampling and Enhancement}}, 
      year={2025},
      volume={},
      number={},
      pages={},
      keywords={},
      doi={}
}
```

We are grateful for being able to contribute this repository for the advancement of marine science and the greater good.