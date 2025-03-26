import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Evaluate score of summaries')
parser.add_argument('--summary_file', '-s', type=str, help='Read input summary frames from a text files')

args = parser.parse_args()
G =  [294, 2980, 4438, 6461, 6711, 12064]
H = [444, 2820, 6465, 6507, 9536, 12083]
I = [191, 1803, 3015, 6207, 6625, 12082]
J = [136, 3022, 4281, 6357, 6500, 12061]
K = [293, 597, 3268, 6466, 6716, 8082]
E_aug28_480_nobackbone_finetuned = [538, 2996, 6404, 6630, 7578, 9104]


sum_path = args.summary_file

original_imgs = '/home/exampleuser/datasets/IROS_Experiments/Extracted_frame/Gopro_456_256'

Bonaire_kralendijk = '/home/exampleuser/datasets/IROS_Experiments/Enhanced_frames_DM/DM_Bonaire-Kralendijk-GoProHero4Silver-BestSnorkelingandScubaDiving_rename_456_256'

images_dir = original_imgs


if args.summary_file is not None:
    summary = []

    with open(args.summary_file, 'r') as f:
        summary = [int(line.strip()) for line in f]
        if len(summary) > 6:
            print('Error! Summary length is more than 6')
    
    print('used the summary_file!')

print(summary)

# pad up to 5 chars with zeroes.
sum = [str(num).zfill(5) + '.png' for num in summary]

print(sum)

# Number of rows and columns in the grid
rows = 3
cols = 2

sum = [images_dir + '/' + num for num in sum]

print(sum)
print()

# Load all the images and store them in a list
images = [cv2.imread(path) for path in sum]

# Get the dimensions of a single image
height, width, _ = images[0].shape

# Create the canvas to hold the grid of images
canvas_width = cols * width
canvas_height = rows * height
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Fill the canvas with the images in the correct grid positions
for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        if index < len(images):
            y_offset = i * height
            x_offset = j * width
            canvas[y_offset:y_offset+height, x_offset:x_offset+width] = images[index]
        else:
            print("INDEX EXCEEDS LENGTH OF IMAGES. NOT ENOUGH IMAGES")

scale = 1

canvas = cv2.resize(canvas, (scale*canvas_width, scale*canvas_height))

# Show
cv2.namedWindow('Reconstructed Summary', cv2.WINDOW_NORMAL)
cv2.imshow('Reconstructed Summary', canvas)
cv2.waitKey(0)

# Save the canvas image
cv2.imwrite('reconstructed_summary.png', canvas)