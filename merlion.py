import os
import torch
import clip
import time
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from summarizer import VideoSummarizer
import json

# Default text prompt
long_text = [
    "underwater image with fish",
    "underwater image without fish"
]

def main(video_frame_path, summary_size, save_summary_to, threshold, prompt_file):
    global long_text
    if not os.path.isdir(video_frame_path):
        print("Invalid video frame path")
        return

    video_frame_path_list = sorted(os.listdir(video_frame_path))

    summary_instance = VideoSummarizer(max_summary_size=summary_size, distance='js')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used is: ", device)
    model, preprocess = clip.load("ViT-B/32", device=device) # Choose CLIP model. chosen: ViT-B/32

    # Use custom prompts
    if prompt_file is not None and prompt_file.strip() is not "":
        with open(prompt_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
            lines = [line for line in lines if line] # filter out empty lines
            long_text = lines

    # Save config
    os.makedirs(save_summary_to, exist_ok=True)
    with open(os.path.join(save_summary_to, "prompts_used.txt"), "w+") as pfile:
        for p in long_text:
            pfile.write(f"{p}\n")
    with open(os.path.join(save_summary_to, "args.txt"), "w+") as afile:
        json.dump(vars(args), afile, indent=4)

    # Cache text input result. (Modify code if text input is changeable during running of file)
    text_input = torch.cat([clip.tokenize(p) for p in long_text]).to(device)
    text_features = None
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Run MERLION for each image
    for frame_id, frame_path in enumerate(tqdm(video_frame_path_list)):
        full_img_path = os.path.join(video_frame_path, frame_path)
        image_input = preprocess(Image.open(full_img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _sim = similarity.squeeze().to(device)

        for index, value in enumerate(similarity.squeeze().cpu()):
            print(f"{long_text[index]:>16s}: {100 * value.item():.2f}%")

        if _sim[0] > threshold:
            image_features = torch.abs(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            summary_instance.add_observation(frame_path, torch.squeeze(image_features))

    summary_instance.save_selected_frames(video_frame_path, save_summary_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERLION video frame summarization.")
    parser.add_argument("--video_frame_path", type=str, required=True, help="Path to the folder containing video frames.")
    parser.add_argument("--prompt_file", type=str, required=False, help="Path to the file containing text prompts. It is a text file with each prompt on each line with no quotation marks. The first line must be the positive / desired prompt")
    parser.add_argument("--summary_size", type=int, default=6, help="Number of frames to include in the summary.")
    parser.add_argument("--save_summary_to", type=str, required=True, help="Directory to save the summary results.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Threshold for selecting frames.")

    args = parser.parse_args()

    start_time = time.time()
    main(args.video_frame_path, args.summary_size, args.save_summary_to, args.threshold, args.prompt_file)
    print("Time Taken:", time.time() - start_time)


# python merlion.py --video_frame_path "/path/to/frames" --summary_size 6 --save_summary_to "/path/to/save" --threshold 0.5