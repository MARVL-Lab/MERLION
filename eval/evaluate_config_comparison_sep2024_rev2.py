import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
import cv2 as cv
import pickle
import statistics
import copy
import ast
import argparse

parser = argparse.ArgumentParser(description='Evaluate score of summaries')
parser.add_argument('--eval_mode', type=str, help='Evaluation algorithm chosen')
parser.add_argument('--fps', type=int, help='Framerate of the source video')
parser.add_argument('--threshold', type=float, help='Threshold for the spot-on score of time score')
parser.add_argument('--weight', type=float, help='Weightage (alpha) for semantic score compared to time representative score')
parser.add_argument('--second_path', type=str, help='Path to second batch of frame number file')
parser.add_argument('--benchmark_dataset', required=True, type=str, help='Choose the human benchmark dataset. e.g. gopro_sep24_unenh , gopro_sep24_enhanced , BK_clear , Powells_murky')
parser.add_argument('--labels', type=str, required=True, help='Choose the appropriate labels for the chosen dataset. e.g. gopro_feb2024 , gopro_sep2024 ,  bkclear_v15_sep2024 , powells_sep2024')
parser.add_argument('--output_csv', type=str, help='Write the resultant output into a CSV')

args = parser.parse_args()

# do not change the below values
set_fps = 30
set_weight = 0.5
set_threshold = 2

### Gopro bonaire: unenh sep 2024

bon_unen_011011 = [329, 4446, 5371, 6466, 6515, 12127]
bon_unen_012012 = [88, 2045, 2368, 3007, 4443, 6491]
bon_unen_014014 = [162, 2910, 5348, 6521, 6716, 12074]
bon_unen_002002 = [275, 629, 2818, 6522, 9452, 12096]
bon_unen_013013 = [287, 2991, 5504, 6448, 6697, 7245]
bon_unen_007007 = [261, 638, 2991, 6403, 10949, 12075]
bon_unen_004004 = [261, 348, 1894, 2692, 2968, 3028]
bon_unen_015015 = [279, 615, 2985, 4445, 6416, 6492]
bon_unen_001001 = [1593, 1842, 2104, 3154, 6509, 11515]
bon_unen_006006 = [310, 619, 1868, 3003, 5278, 12081]
bon_unen_003003 = [70, 337, 618, 2063, 5331, 6343]
bon_unen_010010 = [236, 312, 3026, 6460, 6491, 12079]
bon_unen_000000 = [326, 1779, 2225, 2998, 6518, 12084]
bon_unen_005005 = [320, 625, 3234, 4442, 6514, 12065]
bon_unen_008008 = [1794, 2973, 5565, 6543, 6706, 9462]
bon_unen_009009 = [1811, 3001, 5538, 6515, 6714, 7207]

bonaire_unenh_sep2024 = [bon_unen_011011, bon_unen_012012, bon_unen_014014, bon_unen_002002, bon_unen_013013, bon_unen_007007, bon_unen_004004, bon_unen_015015, bon_unen_001001, bon_unen_006006, bon_unen_003003, bon_unen_010010, bon_unen_000000, bon_unen_005005, bon_unen_008008, bon_unen_009009]

bonaire_unenh_sep2024_14count = [bon_unen_011011, bon_unen_012012, bon_unen_002002, bon_unen_013013, bon_unen_007007, bon_unen_004004, bon_unen_001001, bon_unen_006006, bon_unen_003003, bon_unen_010010, bon_unen_000000, bon_unen_005005, bon_unen_008008, bon_unen_009009]

gopro_sep2024_v2_unen = bonaire_unenh_sep2024_14count

### Gopro bonaire: enhanced sep 2024

bon_enh_011011 = [326, 4444, 6550, 6633, 6714, 7810]
bon_enh_012012 = [160, 2033, 2221, 4444, 6564, 6705]
bon_enh_014014 = [1949, 6350, 6481, 7008, 9535, 12081]
bon_enh_002002 = [300, 634, 1908, 4288, 6524, 12094]
bon_enh_013013 = [268, 1983, 2808, 5482, 6437, 11548]
bon_enh_007007 = [265, 1902, 3007, 4444, 6470, 12069]
bon_enh_004004 = [336, 1862, 2974, 3035, 4450, 6457]
bon_enh_015015 = [325, 2975, 3313, 4443, 6436, 6488]
bon_enh_001001 = [1833, 2053, 2550, 3028, 6742, 9019]
bon_enh_006006 = [285, 635, 3032, 5198, 10186, 12075]
bon_enh_003003 = [315, 637, 3065, 6429, 10226, 12112]
bon_enh_010010 = [232, 3046, 3932, 6369, 11423, 12108]
bon_enh_000000 = [320, 2270, 2581, 5343, 6465, 7259]
bon_enh_005005 = [331, 623, 3007, 4442, 6512, 12074]
bon_enh_008008 = [338, 1880, 2991, 6461, 6518, 6719]
bon_enh_009009 = [306, 3004, 4445, 6489, 6553, 6672]

bonaire_gopro_enhanced_sep16 = [bon_enh_011011, bon_enh_012012, bon_enh_014014, bon_enh_002002, bon_enh_013013, bon_enh_007007, bon_enh_004004, bon_enh_015015, bon_enh_001001, bon_enh_006006, bon_enh_003003, bon_enh_010010, bon_enh_000000, bon_enh_005005, bon_enh_008008, bon_enh_009009]

bonaire_gopro_enhanced_sep16_14count =  [bon_enh_011011, bon_enh_012012, bon_enh_002002, bon_enh_013013, bon_enh_007007, bon_enh_004004, bon_enh_001001, bon_enh_006006, bon_enh_003003, bon_enh_010010, bon_enh_000000, bon_enh_005005, bon_enh_008008, bon_enh_009009]

gopro_sep2024_v2_enhanced = bonaire_gopro_enhanced_sep16_14count

## New sep 2024 BK kralendijk dataset

bk_011011 = [180 , 1428 , 2270 , 2467 , 5542 , 5860]
bk_012012 = [1273 , 5823 , 6057 , 6318 , 7502 , 10682]
bk_014014 = [1276 , 2515 , 5721 , 7498 , 9078 , 9637]
bk_002002 = [1694 , 1862 , 2698 , 5531 , 9117 , 10678]
bk_013013 = [5543 , 5999 , 7024 , 7512 , 8789 , 10680]
bk_007007 = [2894 , 5791 , 7509 , 9007 , 10426 , 10854]
bk_004004 = [2588 , 4696 , 5709 , 6334 , 9104 , 10850]
bk_015015 = [2953 , 5711 , 5790 , 8844 , 9179 , 10860]
bk_001001 = [1282 , 1737 , 2887 , 6264 , 9110 , 9233]
bk_006006 = [7000 , 7949 , 8463 , 8796 , 10425 , 10675]
bk_003003 = [4698 , 5394 , 5849 , 7535 , 8673 , 9021]
bk_010010 = [1251 , 4687 , 5807 , 6348 , 10406 , 10681]
bk_000000 = [57 , 2952 , 5824 , 7022 , 9165 , 10280]
bk_005005 = [2625 , 5710 , 7599 , 9154 , 10428 , 10845]
bk_008008 = [1745 , 3019 , 5713 , 9017 , 9327 , 10784]
bk_009009 = [5718 , 7008 , 8791 , 9014 , 9097 , 10851]

bk_sep2024 = [bk_011011, bk_012012, bk_014014, bk_002002, bk_013013, bk_007007, bk_004004, bk_015015, bk_001001, bk_006006, bk_003003, bk_010010, bk_000000, bk_005005, bk_008008, bk_009009 ]

bk_sep2024_14set = [bk_011011, bk_012012, bk_002002, bk_013013, bk_007007, bk_004004, bk_001001, bk_006006, bk_003003, bk_010010, bk_000000, bk_005005, bk_008008, bk_009009 ]

BK_clear_benchmark = bk_sep2024_14set

# Powells murky benchmark

pow_002002 = [916, 1847, 2177, 2919, 3748, 4450]
pow_011010 = [688, 2162, 2524, 2943, 3915, 4038]
pow_010009 = [917, 2170, 2318, 2526, 3930, 4043]
pow_007007 = [695, 2167, 2530, 2944, 3914, 4043]
pow_001001 = [698, 799, 919, 2880, 3923, 4068]
pow_006006 = [928, 2183, 2370, 2933, 3945, 4042]
pow_003003 = [921, 2172, 2542, 2882, 3931, 4053]
pow_013012 = [2327, 2523, 2881, 2944, 3923, 4044]
pow_014013 = [923, 2492, 2517, 2882, 3168, 4045]
pow_016015 = [799, 2525, 2931, 3224, 3928, 4050]
pow_015014 = [686, 907, 2181, 2521, 3912, 4038]
pow_005005 = [2163, 2530, 2933, 3940, 4046, 4085]
pow_009008 = [806, 931, 2302, 2529, 3945, 4047]
pow_012011 = [920, 1692, 2181, 2543, 3245, 4069]

powells_murky_sep1324 = [pow_002002, pow_011010, pow_010009, pow_007007, pow_001001, pow_006006, pow_003003, pow_013012, pow_014013, pow_016015, pow_015014, pow_005005, pow_009008, pow_012011]

Powells_murky_benchmark = powells_murky_sep1324


pickle_file = ''

##
### Set the benchmarks set
##
benchmarks_set = []

def eval_many_to_one(frames, threshold, fps):
    benchmarks = benchmarks_set

    spot_on = threshold * fps

    current_score_matrix = [] # there's 6, one for each fi
    for fi in frames: # evaluate each frame in the computer generated summary
        fi = int(fi)
        score = 0
        found_times = 0   # count each time it is FOUND in each human summary (at most once in each human summary)
        for benchmark in benchmarks: # loop through each human generated summary
            current_match = 0
            for fb in benchmark: # loop through each frame of the human summary 
                delta = abs(fi-fb)
                if delta < spot_on: 
                    if current_match < 1:
                        current_match = 1
                else:                           
                    temp = math.exp(-1* (delta - spot_on)/fps) 
                    if current_match < temp:
                        current_match = temp
            # after one benchmark.
            if current_match > 0:
                score += current_match
        # after all benchmarks have been compared
        # normalize score to average of all benchmarks
        score /= len(benchmarks)
        current_score_matrix.append(score)
        # end of one frame
    # after all frames for one com generated summary are done
    total_score = 0
    for i in range(len(current_score_matrix)):
        total_score += current_score_matrix[i]
    return total_score/len(current_score_matrix)

## Preferred this one
def eval_one_to_many(frames, threshold, fps):
    benchmarks = benchmarks_set

    spot_on = threshold * fps

    current_score_matrix = [] # there's one of this for each human benchmark, depending on how many human summaries
    for benchmark in benchmarks: # loop through each human generated summary

        assert len(benchmark) == 6

        benchmark_total_score = 0

        for fb in benchmark: # evaluate each frame in human benchmark
            current_score = 0
            for ind, fi in enumerate(frames): # evaluate each frame in the computer generated summary
                fi = int(fi)
                # score = 0
                # current_match = 0
                delta = abs(fi-fb)
                if delta < spot_on: 
                    if current_score < 1:
                        current_score = 1
                else: # exponential decay
                    temp = math.exp(-1* (delta - spot_on)/fps)
                    if temp > current_score: # keep the max score, update if better
                        current_score = temp 
            # Give score to current human benchmark frame score and add to benchmark total score
            benchmark_total_score += current_score
        
        current_score_matrix.append(benchmark_total_score/len(benchmark))
    
    total_score = statistics.mean(current_score_matrix)
    return total_score

def read_video(vid_files, idx):
    image_file = vid_files[idx-1]
    return cv.imread(image_file, cv.IMREAD_UNCHANGED)

def time_score(fi, fb, fps, minimum = 0.5, threshold = 0.6):
    spot_on = threshold*fps
    delta = abs(fi-fb)
    time_score = math.exp(-1 * (delta - spot_on)/fps)
    score = max(min(time_score, 1), minimum) # Ensures that the score is between minimum and 1.0
    # print(score)
    return score

def time_score_linear(fi, fb, fps, minimum = 0.5, threshold = 0.6):
    spot_on = threshold*fps
    delta = abs(fi-fb)
    time_score = math.exp(-1 * (delta - spot_on)/fps)
    score = min(time_score, 1) # Ensures that the score is between 0.0 and 1.0
    # print(score)
    return score

###
### start Labels class
###
class LabelsObj:
    def __init__(self, labels):
        self.labels = labels
        self.status = self.determine_status()

    def determine_status(self):
        if self.is_list():
            return 'l'
        elif self.is_dict():
            return 'd'
        else:
            return 'nil'

    def is_list(self):
        return isinstance(self.labels, list)

    def is_dict(self):
        return isinstance(self.labels, dict)

    def get_labels_nested(self, frame_num):
        # According to the format: (feb 27 read_label_dict_manualformat.py)
        return  self.labels[2][self.labels[0][frame_num]]

    def __getitem__(self, frame):
        if self.status == 'l':
            return self.get_labels_nested(frame)
        elif self.status == 'd':
            return self.labels[frame]
###
### end Labels class
###


# This not preferred! use not remove
def eval_semantic(frames, labels, weight=0.5, threshold=2.0, fps=30):
    '''
    labels: dict[frame_num] = [semantic_labels]
    frames: [frame_nums]
    '''
    benchmarks = benchmarks_set

    current_score_matrix = [] # there's 3, one for each human summary
    for benchmark in benchmarks: # loop through each human generated summary
        current_score = 0
        skip_ind = []
        for curr_ind in range(6): # evaluate each frame in the human summary
            fb = benchmark[curr_ind]
            human_label = labels[fb]
            time_score_list = []
            time_score_ind = []
            for ind, fi in enumerate(frames): # evaluate each frame in the computer generated summary
                fi = int(fi)
                if ind in skip_ind:
                    continue
                automated_label = labels[fi]
                for l in automated_label:
                    if l in human_label and l != '' and l!= '[]' and l!= '['']': ## exclude empty
                        score = time_score_linear(fi, fb, fps, threshold = threshold)
                        score = weight*1 + (1-weight)*score
                        print("L: ", l, " H: ", human_label, " Sc: ", score)
                        time_score_list.append(score)
                        time_score_ind.append(ind)
            current_score+=max(time_score_list, default=0)
            if not len(time_score_list) == 0:
                skip_ind.append(time_score_ind[np.argmax(time_score_list)])
        current_score_matrix.append(current_score/6)
    total_score = statistics.mean(current_score_matrix)
    return total_score

# This one preferred
def eval_semantic_not_remove(frames, labels, weight=0.5, threshold=2.0, fps=30):
    '''
    labels: dict[frame_num] = [semantic_labels]
    frames: [frame_nums]
    '''
    benchmarks = benchmarks_set

    k = 1
    current_score_matrix = [] # there's 3, one for each human summary
    for benchmark in benchmarks: # loop through each human generated summary
        current_score = 0
        for curr_ind in range(6): #this is each frame in the current human benchmark summary
            fb = benchmark[curr_ind]
            human_label = labels[fb]
            time_score_list = []
            time_score_ind = []
            for ind, fi in enumerate(frames): # evaluate each frame in the computer generated summary
                fi = int(fi)
                automated_label = labels[fi]
                for l in automated_label:
                    if l in human_label and l != '' and l!= '[]' and l!= '['']': ## exclude empty
                        score = time_score_linear(fi, fb, fps, threshold = threshold)
                        score = weight*1 + (1-weight)*score
                        print("L: ", l, " H: ", human_label, " Sc: ", score)
                        time_score_list.append(score)
                        time_score_ind.append(ind) # not used
            print(f"best score {k}:\t", max(time_score_list, default=0))
            k += 1
            current_score+=max(time_score_list, default=0) # score for 1 human frame
        current_score_matrix.append(current_score/6)
    total_score = statistics.mean(current_score_matrix)
    return total_score

def read_file_semantic(input_path, labels, weight=0.5, threshold=2.0, fps=30):
    # Reads input folder, gets all the config files and saves the frames into a dictionary with key: config and value: [frames_exp1, frames_exp2, ...]
    files = input_path
    print("THESE ARE FILES SEMANTIC")
    print(files)
    print("DONE FILES SEMANTIC")
    frames_dict = defaultdict(list)
    for f in files:
        frames = np.loadtxt(f)

        if 'frames.txt' in f:
            ## Convert 2-based to 1-based.
            frames = frames-1

        score = 0
        if scoring == 'eval':
            print('EVAL VANILLA TIME DIFFERENCE THRESHOLD ', threshold, ' with fps ')
            score = eval_many_to_one(frames, threshold, fps)        
        elif scoring == 'eval_one2many':
            print('EVAL ONE TO MANY')
            score = eval_one_to_many(frames, threshold, fps)
        elif scoring == 'semantics':
            print('SEMANTICS')
            score = eval_semantic(frames, labels, weight=weight, threshold=threshold, fps=fps)
        elif scoring == 'semantics_not_remove':
            print('SEMANTICS NOT REMOVE')
            score = eval_semantic_not_remove(frames, labels, weight=weight, threshold=threshold, fps=fps)
        
        print('Final score of Semantic:\t', score)

        frames_dict['default'].append(score)
    return frames_dict

def get_highest(rost_dict):
    best_dict = {k: max(v) for k, v in rost_dict.items()}
    return best_dict

if __name__ == "__main__":

    ### Pickle label files
    # use lean strict to be lean and strict. this prevents the skewness of evaluation score as small_black_fish and yellow-black striped longfish are ubiquitous.
    bk_clear_v1_5_pkl_sep1324 = './labels/kralendijk_sep13_v1_5_lean_strict.pkl'
    powellscay_v1_pkl_sep1324 = './labels/powellscay_sep13_v1.pkl'
    gopro_bonaire_v2_pkl_sep1624 = './labels/gopro-bonaire_sep16_v2.pkl'
    
    pickle_file = None

    if args.labels != None and args.labels != '':
        labels_choice = args.labels
        if labels_choice == 'gopro_sep2024':
            pickle_file = gopro_bonaire_v2_pkl_sep1624
        elif labels_choice == 'bkclear_v15_sep2024':
            pickle_file = bk_clear_v1_5_pkl_sep1324
        elif labels_choice == 'powells_sep2024':
            pickle_file = powellscay_v1_pkl_sep1324
    
    ## Set params

    ## Set benchmark set

    benchmark_choice = 'gopro_sep24_enhanced'

    # default benchmark set
    # no defaults
    bencmarks_set = None

    if args.benchmark_dataset != None and args.benchmark_dataset != '':
        benchmark_choice = args.benchmark_dataset
        if benchmark_choice == 'gopro_sep24_unenh':
            benchmarks_set = copy.deepcopy(gopro_sep2024_v2_unen)
        elif benchmark_choice == 'gopro_sep24_enhanced':
            benchmarks_set = copy.deepcopy(gopro_sep2024_v2_enhanced)
        elif benchmark_choice == 'BK_clear':
            benchmarks_set = copy.deepcopy(BK_clear_benchmark)
        elif benchmark_choice == 'Powells_murky':
            benchmarks_set = copy.deepcopy(Powells_murky_benchmark)
        else:
            pass # don't use anything and throw error

    print(benchmarks_set)

    # Check params

    if args.fps != None and args.fps != 0:
        new_fps = args.fps
        if new_fps < 0:
            print('ERROR PARAMS!!! \n ERROR param fps cannot be < 0 \t fps param:\t', args.fps, ' set to default: 30')
            new_fps = 30
        set_fps =  new_fps
    
    if args.weight != None and args.weight != 0:
        new_weight = args.weight
        if new_weight > 1:
            print('ERROR PARAMS!!! \n ERROR param weight cannot be > 1 \t weight param:\t', args.weight, ' set to 1')
            new_weight = 1
        elif new_weight < 0:
            print('ERROR PARAMS!!! \n ERROR param weight cannot be < 0 \t weight param:\t', args.weight, ' set to default: 0.5')
            new_weight = 1
        set_weight = new_weight
    
    if args.threshold != None and args.threshold != 0:
        new_threshold = args.threshold
        if new_threshold < 0:
            print('ERROR PARAMS!!! \n ERROR param threshold cannot be < 0 \t threshold param:\t', args.threshold, ' set to default: 2')
            new_threshold = 2
        set_threshold = new_threshold

    set_scoring = 'semantics_not_remove'

    available_evals = ['semantics', 'semantics_not_remove', 'eval', 'eval_one2many',]

    if args.eval_mode != None and args.eval_mode != '':
        if args.eval_mode not in available_evals:
            print('ERRROR!!! Scoring eval method \t', args.eval_mode, '  is not available!')
            print('check:\n', available_evals)
        else:
            set_scoring = args.eval_mode

    scoring = set_scoring
    # scoring = 'semantics_not_remove' # use this
    weight= set_weight
    threshold= set_threshold
    fps = set_fps
    
    if weight == 0:
        scoring = 'eval_one2many' # pure time scoring

    # check pickle and assign to labels, then make LabelsObj object
    if weight != 0:
        with open(pickle_file, 'rb') as f:
            labels_raw = pickle.load(f)
            print(labels_raw)
            # make LabelsObj object
            labels = LabelsObj(labels_raw)

    experiment_path = "/home/user/Desktop/merlion/experiments/"

    semantic_files = 'default_path'
    
    # Python argument 
    if args.second_path is None or args.second_path == '':
        print('-----NO Input file is supplied!----- add second_path arg to point to frame file containing the frame numbers')
        quit()
    else:
        print('----------------Using custom paths-----------------')
        semantic_files = [args.second_path]

    print("---(B) / SECOND SET---")
    print(semantic_files)

    semantic_dict = read_file_semantic(semantic_files, labels, weight=weight, threshold=threshold, fps=fps) 
    
    print("\n\n\n")
    print("SEMANTIC DICT")
    print(semantic_dict)
    print(semantic_dict.items())
    semantic_dict = get_highest(semantic_dict)
    diff_list = []
    bins = []
    semantic_score = -1
    for k in sorted(semantic_dict.keys()):
        sem_v = semantic_dict[k]
        semantic_v = semantic_dict.get(k, [0])
        print('K : ' + str(k))
        print('SEM : ' + str(semantic_v))
        semantic_score = semantic_v
        bins.append(k)
        
    # human trials

    benchmarks_set_backup = copy.deepcopy(benchmarks_set)

    human_scores = []
    for i in range(len(benchmarks_set_backup)):
        eval_human = benchmarks_set.pop(i)
        if scoring == 'semantics':
            score = eval_semantic(eval_human, labels, weight=weight, threshold=threshold, fps=fps)
        elif scoring == 'semantics_not_remove': # Preferred!
            score = eval_semantic_not_remove(eval_human, labels, weight=weight, threshold=threshold, fps=fps)
        elif scoring == 'eval':
            score = eval_many_to_one(eval_human, threshold, fps)
        elif scoring == 'eval_one2many':
            score = eval_one_to_many(eval_human, threshold=threshold, fps=fps)
        human_scores.append(score)
        print("human ", i ," scored against the rest:", score)
        # replenish original benchmarks set
        benchmarks_set = copy.deepcopy(benchmarks_set_backup)
    human_avg = statistics.mean(human_scores)
    print("Humans average:", human_avg)
    print("Humans min:", min(human_scores))
    print("Humans max:", max(human_scores))
    print("Humans stdev:", np.std(human_scores))
    print("Humans 25:", np.percentile(human_scores, 25))
    print("Humans median:", np.percentile(human_scores, 50))
    print("Humans 75:", np.percentile(human_scores, 75))
    print("Semantic score:\t", semantic_score)
    print("Semantic score %:\t", 100*semantic_score/human_avg)


    print('\n', 'WEIGHT CHOSEN: \t ', weight)
    print('THRESHOLD CHOSEN: \t ', threshold)
    print('FPS CHOSEN: \t ', fps)
    print('SCORING METHOD CHOSEN: \t ', scoring)
    print('\n', 'HUMAN BENCHMARK CHOSEN: \t ', benchmark_choice)
    print('Please verify that human benchmark set is correct')
    sem = args.second_path.split('/')[-2]
    print('Dataset chosen:\t', sem)

    # append into csv
    csv_outfile = args.output_csv
    if csv_outfile is not None:
        if os.path.exists(csv_outfile):
            # Write the incremented number back to the file
            with open(csv_outfile, "a") as file:
                file.write(sem + ',' + str(semantic_score) + ',' + 'HUMAN:' + ',' + str(human_avg) + ',' + 'HUMAN MIN:' + ',' + str(min(human_scores)) + ',' + 'HUMAN MAX:' + ',' + str(max(human_scores)) + ',' + 'STD' + ',' + str(np.std(human_scores)) + ',' + '25th' + ',' + str(np.percentile(human_scores, 25)) + ',' + 'MED:' + ',' + str(np.percentile(human_scores, 50)) + ',' + '75th' + ',' + str(np.percentile(human_scores, 75))  + ',' + 'score_as_%:' + ',' + str(100*semantic_score/human_avg) + ',' + 'threshold:' + ',' + str(threshold) + ',' + 'weight:' + ',' + str(weight) + ',' + 'Benchmark:' + ',' + args.benchmark_dataset  + '\n')
        else:
            # Create the file and set count to 0 if it doesn't exist or is empty
            with open(csv_outfile, "w") as file:
                file.write(sem + ',' + str(semantic_score) + ',' + 'HUMAN:' + ',' + str(human_avg) + ',' + 'HUMAN MIN:' + ',' + str(min(human_scores)) + ',' + 'HUMAN MAX:' + ',' + str(max(human_scores)) + ',' + 'STD' + ',' + str(np.std(human_scores)) + ',' + '25th' + ',' + str(np.percentile(human_scores, 25)) + ',' + 'MED:' + ',' + str(np.percentile(human_scores, 50)) + ',' + '75th' + ',' + str(np.percentile(human_scores, 75))  + ',' + 'score_as_%:' + ',' + str(100*semantic_score/human_avg) + ',' + 'threshold:' + ',' + str(threshold) + ',' + 'weight:' + ',' + str(weight) + ',' + 'Benchmark:' + ',' + args.benchmark_dataset  + '\n')
    else:
        print('CSV File is not supplied! results not recorded')

