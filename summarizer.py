import torch
import torch.nn.functional as F
import numpy as np
import shutil
import os

def mean_squared_error(tensor1, tensor2, dim):
    return torch.mean((tensor1 - tensor2) ** 2, dim=dim)

def kl_divergence(p: torch.Tensor, q: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        return torch.sum(p * torch.log(p / q))
    else:
        return torch.sum(p * torch.log(p / q), dim)

def js_divergence(p: torch.Tensor, q: torch.Tensor, dim=None) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, dim) + 0.5 * kl_divergence(q, m, dim)

class VideoSummarizer:
    def __init__(self, max_summary_size, distance):
        self.selected_frames = {}
        self.max_summary_size = max_summary_size
        if distance == 'js':
            self.distance_function = js_divergence
        elif distance == 'kl':
            self.distance_function = kl_divergence
        self.threshold = 0
        self.unique_frame_ids = set()

    def add_observation(self, obs_id, observation):
        if len(self.unique_frame_ids) < self.max_summary_size:
            self.selected_frames[obs_id] = observation
            self.unique_frame_ids.add(obs_id)
        else:
            self.compute_gamma()
            surprise_score = self.compute_surprise_score(observation)
            if surprise_score > self.threshold:
            # if surprise_score > 20:
                self.selected_frames[obs_id] = observation
                self.unique_frame_ids.add(obs_id)
                self.trim_summary(obs_id)

    def compute_surprise_score(self, current_observation):
        selected_frames_matrix = torch.stack(list(self.selected_frames.values()))
        current_obs_matrix = current_observation.unsqueeze(0).expand_as(selected_frames_matrix)
        mse_values = self.distance_function(selected_frames_matrix, current_obs_matrix, dim=1)
        surprise_score = torch.min(mse_values)
        return surprise_score

    def trim_summary(self, obs_id=-1):
        if not self.selected_frames:
            return

        new_summary, new_uids = {}, set()
        new_summary.update({k: v for k, v in self.selected_frames.items() if k == obs_id})
        new_uids.add(obs_id)
        del self.selected_frames[obs_id]
        new_k = min(len(self.unique_frame_ids), self.max_summary_size) - 1

        while new_k > 0:
            selected_frame_key = self.select_max_min_frame_key(new_summary, self.selected_frames)
            new_summary[selected_frame_key] = self.selected_frames[selected_frame_key]
            new_uids.add(selected_frame_key)
            del self.selected_frames[selected_frame_key]
            new_k -= 1

        self.selected_frames, self.unique_frame_ids = new_summary, new_uids

    def compute_gamma(self):
        selected_frames_list = list(self.selected_frames.values())
        selected_frames_matrix = torch.stack(selected_frames_list)
        pairwise_distances = self.distance_function(selected_frames_matrix[:, None, :], selected_frames_matrix[None, :, :], dim=2)
        pairwise_distances.fill_diagonal_(float('inf'))
        min_distances, _ = torch.min(pairwise_distances, dim=1)
        self.threshold = torch.mean(min_distances)

    def select_max_min_frame_key(self, new_summary_dict, old_summary_dict):
        new_summary = torch.stack(list(new_summary_dict.values()))
        old_summary = torch.stack(list(old_summary_dict.values()))
        distances = self.distance_function(old_summary[:, None, :], new_summary[None, :, :], dim=2)
        min_distances, _ = torch.min(distances, dim=1)
        selected_frame_index = torch.argmax(min_distances)
        selected_frame_key = list(old_summary_dict.keys())[selected_frame_index.item()]
        return selected_frame_key

    def save_selected_frames(self, input_path, output_path):
        for path in self.selected_frames.keys():
            full_path = os.path.join(input_path, path)
            shutil.copy(full_path, output_path)
