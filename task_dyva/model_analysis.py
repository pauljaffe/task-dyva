import os
import pickle
import itertools

import torch
import numpy as np
import pandas as pd

from .utils import get_all_trial_combos


class FixedPointFinder():
    # Find stable fixed points from a supplied model.
    # The default params are those used in the paper. 

    def __init__(self, expt, expt_stats, fp_path, fp_summary_path,
                 load_saved=True, z_dim=16, rand_seed=12345,
                 t_step=20, t_search_win=10000, max_sd_tol=1e-3,
                 max_dist_tol=1e-3):
        self.expt = expt
        self.expt_stats = expt_stats
        self.fp_path = fp_path
        self.fp_summary_path = fp_summary_path
        self.load_saved = load_saved
        self.z_dim = z_dim
        self.t_step = t_step
        self.t_win = t_search_win
        self.max_sd_tol = max_sd_tol
        self.max_dist_tol = max_dist_tol
        self.stimulus_combos = get_all_trial_combos()
        # Enforce reproducibility
        self.rng = np.random.default_rng(rand_seed)

    def find_fixed_points(self, N, T):
        # N: number of random latent states used to initialize the fixed
        # point finder.
        # T: duration of stimuli used to find fixed points (ms)

        if os.path.exists(self.fp_path) and self.load_saved:
            with open(self.fp_path, 'rb') as path:
                fps = pickle.load(path)
        else:
            all_fps = []
            # Randomly sample initial states to find fixed points
            z0 = self._make_z0(N)

            for c in self.stimulus_combos:
                pt, mv, cue = c[0], c[1], c[2]
                c_stimuli = self._make_fp_stimuli(N, T, pt=pt, mv=mv, cue=cue)
                # Generate responses
                _, _, _, z_out, _, _ = self.expt.model.forward(c_stimuli,
                                                               generate_mode=True,
                                                               clamp=True,
                                                               z0_supplied=z0)
                z_np = z_out.detach().numpy()
                c_fps = self._check_for_fps(z_np, c, N)
                if c_fps is not None:
                    all_fps.extend(c_fps)

            if len(all_fps) > 0:
                fp_df = pd.concat(all_fps)
                fps = self._project_fps(fp_df)
                fps.reset_index(drop=True, inplace=True)
            else:
                fps = pd.DataFrame({'type': [], 'zloc': [],
                                    'cue': [], 'mv': [], 'pt': []})
            fps.to_pickle(self.fp_path)
        return fps

    def get_fixed_point_summary(self, fps):
        # Calculate summary stats on the fixed points.
        if os.path.exists(self.fp_summary_path) and self.load_saved:
            with open(self.fp_summary_path, 'rb') as path:
                summary = pickle.load(path)
        elif len(fps) == 0:
            summary = {'within_task': [], 'between_task': [],
                       'same_response': [], 'different_response': [],
                       'N': 0, 'f_stimuli_with_fp': 0}
        else:
            summary = {'within_task': [], 'between_task': [],
                       'same_response': [], 'different_response': []}
            fp_inds = np.arange(len(fps))
            pairs = list(itertools.combinations(fp_inds, 2))
            for i1, i2 in pairs:
                fp1 = fps.iloc[i1, :]
                fp2 = fps.iloc[i2, :]
                z1, z2 = fp1['zloc'], fp2['zloc']
                this_dist = np.linalg.norm(z1 - z2, ord=2)
                within_task, same_dir = self._classify_pair(fp1, fp2)
                if within_task:
                    summary['within_task'].append(this_dist)
                    if same_dir:
                        summary['same_response'].append(this_dist)
                    else:
                        summary['different_response'].append(this_dist)
                else:
                    summary['between_task'].append(this_dist)
            summary['N'] = len(fps)
            summary['f_stimuli_with_fp'] = self._count_fps(fps)
        with open(self.fp_summary_path, 'wb') as path:
            pickle.dump(summary, path, protocol=4)
        return summary

    def _count_fps(self, fps):
        # Determine the fraction of the 32 possible stimulus
        # configurations that have a fixed point.
        n = len(fps.drop_duplicates(subset=['cue', 'mv', 'pt'], inplace=False))
        return n / 32

    def _classify_pair(self, fp1, fp2):
        mv1, pt1, cue1 = fp1['mv'], fp1['pt'], fp1['cue']
        mv2, pt2, cue2 = fp2['mv'], fp2['pt'], fp2['cue']
        if cue1 == cue2:
            within_task = True
        else:
            within_task = False
        if cue1 == 0: # moving task
            correct_dir1 = mv1
        else:
            correct_dir1 = pt1
        if cue2 == 0: # moving task
            correct_dir2 = mv2
        else:
            correct_dir2 = pt2
        if correct_dir1 == correct_dir2:
            same_dir = True
        else:
            same_dir = False
        return within_task, same_dir

    def _make_z0(self, N):
        # Select a set of initial states for finding fixed points:
        # randomly sample z from states visited when the model 
        # is used to generate responses.
        z = self.expt_stats.latents
        t_rand = self.rng.choice(z.shape[0], N)
        n_rand = self.rng.choice(z.shape[1], N)
        z0 = z[t_rand, n_rand, :]
        z0_torch = torch.Tensor(np.reshape(z0, (1, N, self.z_dim)))
        return z0_torch

    def _make_fp_stimuli(self, N, T, pt=None, mv=None, cue=None):
        # Make N static stimuli of length T: one stimulus configuration is
        # active for the entire stimulus duration. 
        pt_start_ind, mv_start_ind, cue_start_ind = 4, 8, 12
        T_samples = T // self.t_step
        stimuli = torch.zeros(T_samples, N, 14)
        if pt is not None:       
            stimuli[:, :, pt_start_ind + pt] = 1
        if mv is not None:
            stimuli[:, :, mv_start_ind + mv] = 1
        if cue is not None:
            stimuli[:, :, cue_start_ind + cue] = 1
        return stimuli

    def _check_for_fps(self, z, stimuli, N):
        # Find and validate fixed points from the latent state
        win_samples = self.t_win // self.t_step # window to look for fps
        z_win = z[-win_samples:, :, :]
        z_sd = np.std(z_win, axis=0)
        z_mean_sd = np.mean(z_sd, axis=1)
        stable_inds = np.nonzero(z_mean_sd <= self.max_sd_tol)[0]
        nonstable_inds = np.nonzero(z_mean_sd > self.max_sd_tol)[0]
        z_means = np.mean(z_win, axis=0)
        fp_pt = stimuli[0]
        fp_mv = stimuli[1]
        fp_cue = stimuli[2]

        if len(stable_inds) == 0:
            fps = None
        else:
            fps = []
            stable_z = []
            fp_type = 'stable'
            for si in stable_inds:
                si_z = z_means[si, :]
                if len(stable_z) > 0:
                    new_fp = self._check_redundant_fp(stable_z, si_z)
                else:
                    new_fp = True
                if new_fp:
                    stable_z.append(si_z)
                    this_fp = pd.DataFrame({'type': fp_type, 'zloc': [si_z],
                                            'cue': fp_cue, 'mv': fp_mv, 'pt': fp_pt})
                    fps.append(this_fp)
        return fps

    def _check_redundant_fp(self, all_z, test_z):
        # Check to see if fixed point is redundant with existing fixed points
        new_fp = True
        for z in all_z:
            this_dist = np.linalg.norm(z - test_z, ord=2)
            if this_dist <= self.max_dist_tol:
                new_fp = False
        return new_fp

    def _project_fps(self, fps, keep_dim=3):
        # Project the fixed points onto PC space
        pca_obj = self.expt_stats.pca_obj
        fps_proj = []
        for fp in fps['zloc']:
            this_proj = pca_obj.transform(fp.reshape((1, len(fp))))
            fps_proj.append(this_proj[0][:keep_dim])
        fps['zloc_pca'] = fps_proj
        return fps
