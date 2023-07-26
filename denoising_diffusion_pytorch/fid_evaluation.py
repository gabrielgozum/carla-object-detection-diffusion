import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)
        features = self.inception_v3(samples)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            print(f"num batches load or precalc {num_batches}")
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
            # for _ in tqdm(range(len(self.dl))):
                # print(f"num batches load or precalc {num_batches}")
                try:
                    real_samples = next(self.dl)
                    # print(real_samples.shape)
                except StopIteration:
                # except:
                    break
                real_samples = real_samples.to(self.device)
                # print("calculating real features")
                real_features = self.calculate_inception_features(real_samples)
                # print('appending real features')
                stacked_real_features.append(real_features)
            # for real_samples in tqdm(self.dl):
            #     # print(f"num batches load or precalc {num_batches}")
            #     # print(f"length of dataloader {len(self.dl)}")
            #     # try:
            #     #     real_samples = next(self.dl)
            #     # # except StopIteration:
            #     # except:
            #     #     break
            #     real_samples = real_samples.to(self.device)
            #     print("calculating real features")
            #     real_features = self.calculate_inception_features(real_samples)
            #     print('calculated real features')
            #     stacked_real_features.append(real_features)
            print("Finished stacking real features")
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            # print("loading or precalculating dataset stats")
            self.load_or_precalc_dataset_stats()
            # print("finished loading or precalculating dataset stats")
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)
        # TODO: fix for CUDA OOM
        # batches = [1 for _ in batches]
        # print(f"fid batches {batches}")
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        batches = batches[:2] if len(batches) > 5 else batches
        for batch in tqdm(batches):
            # print(batch)
            fake_samples = self.sampler.sample(batch_size=batch)
            # print("made fake samples")
            fake_features = self.calculate_inception_features(fake_samples)
            # print("made fake features")
            stacked_fake_features.append(fake_features)
            # print("appended fake features")
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
