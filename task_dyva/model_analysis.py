import os
import pickle
import itertools

import torch
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import pearsonr

from .utils import get_stimulus_combos


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
        self.stimulus_combos = get_stimulus_combos()
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
                mv, pt, cue = c[0], c[1], c[2]
                c_stimuli = self._make_fp_stimuli(N, T, pt=pt, mv=mv, cue=cue)
                # Generate responses
                _, _, _, z_out, _, _ = self.expt.model.forward(c_stimuli,
                                                               generate_mode=True,
                                                               clamp=True,
                                                               z0_supplied=z0)
                z_np = z_out.detach().numpy()
                c_fps = self._check_for_fps(z_np, N, pt=pt, mv=mv, cue=cue)
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
        if cue2 == 0:
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

    def _check_for_fps(self, z, N, pt=None, mv=None, cue=None):
        # Find and validate fixed points from the latent state
        win_samples = self.t_win // self.t_step # window to look for fps
        z_win = z[-win_samples:, :, :]
        z_sd = np.std(z_win, axis=0)
        z_mean_sd = np.mean(z_sd, axis=1)
        stable_inds = np.nonzero(z_mean_sd <= self.max_sd_tol)[0]
        nonstable_inds = np.nonzero(z_mean_sd > self.max_sd_tol)[0]
        z_means = np.mean(z_win, axis=0)

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
                                            'cue': cue, 'mv': mv, 'pt': pt})
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
        # Project the fixed points onto the PCs
        pca_obj = self.expt_stats.pca_obj
        fps_proj = []
        for fp in fps['zloc']:
            this_proj = pca_obj.transform(fp.reshape((1, len(fp))))
            fps_proj.append(this_proj[0][:keep_dim])
        fps['zloc_pca'] = fps_proj
        return fps


class LatentsLDA():
    # Run the LDA analyses for the paper (default params are
    # those used in the paper). 

    def __init__(self, expt_stats, save_path, load_saved=True, 
                 time_range=[-100, 1600], n_shuffle=100, rand_seed=12345):
        self.expt_stats = expt_stats
        self.save_path = save_path
        self.load_saved = load_saved
        mean_rt = self.expt_stats.summary_stats['m_mean_rt']
        self.t_ind = np.argmin(np.absolute(self.expt_stats.t_axis - mean_rt))
        self.n_shuffle = n_shuffle
        self.rng = np.random.default_rng(rand_seed)
        self._split_trials_by_task()
        self._split_trials_by_direction()

    def _split_trials_by_direction(self):
        direction_inds = {'mv': {}, 'pt': {}}
        for cue_ind, cue in enumerate(['mv', 'pt']):
            for d in [0, 1, 2, 3]:
                this_filt = {'task_cue': cue_ind}
                if cue_ind == 0:
                    this_filt['mv_dir'] = d
                else:
                    this_filt['point_dir'] = d
                this_inds = self.expt_stats.select(**this_filt)
                direction_inds[cue][d] = this_inds
        self.direction_inds = direction_inds

    def _split_trials_by_task(self):
        mv_filt = {'task_cue': 0}
        pt_filt = {'task_cue': 1}
        mv_inds = self.expt_stats.select(**mv_filt)
        pt_inds = self.expt_stats.select(**pt_filt)
        self.mv_z = np.squeeze(self.expt_stats.windowed['latents'][
            self.t_ind, mv_inds, :])
        self.pt_z = np.squeeze(self.expt_stats.windowed['latents'][
            self.t_ind, pt_inds, :])
        self.mv_N = len(mv_inds)
        self.pt_N = len(pt_inds)
        self.task_y = np.hstack((np.zeros(self.mv_N), 
                                 np.ones(self.pt_N))) 
        self.task_x = np.concatenate((self.mv_z, self.pt_z), axis=0)

    def _get_lda_error(self, x, y):
        this_lda = LDA()
        _ = this_lda.fit(x, y)
        return 1 - this_lda.score(x, y)

    def run_lda_analysis(self):
        if os.path.exists(self.save_path) and self.load_saved:
            with open(self.save_path, 'rb') as path:
                lda_info = pickle.load(path)
        else:
            # True between task error
            bw_error = self._get_lda_error(self.task_x, self.task_y)

            # Between task error calculated on shuffled data
            bw_shuffle_error = self._get_shuffle_error(self.mv_z, self.pt_z)

            # Within task (same vs. different direction)
            # Average errors across all combinations
            cues = ['mv', 'pt']
            directions = [0, 1, 2, 3]
            true_within_errors = []
            shuffle_within_errors = []
            for cue in cues:
                for d in directions:
                    x1_inds = self.direction_inds[cue][d]
                    all_x2_inds = []
                    other_d = [i for i in directions if i != d]
                    for od in other_d:
                        all_x2_inds.extend(self.direction_inds[cue][od])
                    x2_inds = np.array(all_x2_inds)
                    this_x1 = np.squeeze(self.expt_stats.windowed['latents'][
                        self.t_ind, x1_inds, :])
                    this_x2 = np.squeeze(self.expt_stats.windowed['latents'][
                        self.t_ind, x2_inds, :])
                    this_x = np.concatenate((this_x1, this_x2), axis=0)
                    this_y = np.hstack((np.zeros(len(x1_inds)), 
                                        np.ones(len(x2_inds)))) 

                    # True within task error
                    true_within_errors.append(self._get_lda_error(this_x, this_y))

                    # Within task error calculated on shuffled data
                    shuffle_within_errors.append(
                        self._get_shuffle_error(this_x1, this_x2))

            within_error = np.mean(true_within_errors)
            within_shuffle_error = np.mean(shuffle_within_errors)
            lda_info = {'bw_error': bw_error, 
                        'bw_shuffle_error': bw_shuffle_error,
                        'within_error': within_error, 
                        'within_shuffle_error': within_shuffle_error}
            with open(self.save_path, 'wb') as path:
                pickle.dump(lda_info, path, protocol=4)
        return lda_info

    def _get_shuffle_error(self, x1, x2):
        shuffle_errors = []
        combined_data = np.concatenate((x1, x2), 0)
        N1 = x1.shape[0]
        N2 = x2.shape[0]
        all_inds = np.arange(N1 + N2)
        for n in range(self.n_shuffle):
            this_x1_inds = self.rng.choice(N1 + N2, N1, replace=False)
            this_x2_inds = np.setdiff1d(all_inds, this_x1_inds)
            this_y = np.hstack((np.zeros(len(this_x1_inds)), 
                                np.ones(len(this_x2_inds)))) 
            this_x1 = combined_data[this_x1_inds, :]
            this_x2 = combined_data[this_x2_inds, :]
            this_x = np.concatenate((this_x1, this_x2), axis=0)
            shuffle_errors.append(self._get_lda_error(this_x, this_y))
        shuffle_mean = np.mean(shuffle_errors)
        return shuffle_mean


class LatentSeparation():
    # Defines methods for calculating the distance between task centroids,
    # used e.g. in Fig. 4 of the paper.

    def __init__(self, expt_stats, min_N=5):
        self.expt_stats = expt_stats
        self.min_N = min_N
        self.t0_ind = expt_stats.n_pre
        self._get_task_centroids()
        self.combos = get_stimulus_combos()

    def _get_task_centroids(self):
        # Calculate latent state centroids at the mean RT for each task
        mv_filter = {'task_cue': 0, 'prev_task_cue': 0}
        self.mv_inds = self.expt_stats.select(**mv_filter)
        self.mv_centroid = self._get_centroid(self.mv_inds)
        pt_filter = {'task_cue': 1, 'prev_task_cue': 1}
        self.pt_inds = self.expt_stats.select(**pt_filter)
        self.pt_centroid = self._get_centroid(self.pt_inds)

    def _get_centroid(self, trial_inds):
        # Centroid calculated at the mean RT
        z = self.expt_stats.windowed['latents'][:, trial_inds, :]
        return np.mean(np.squeeze(z[self.t0_ind, :, :]), 0)
    
    def _get_trial_distances(self, trial_inds, centroid):
        z = self.expt_stats.windowed['latents'][self.t0_ind, trial_inds, :]   
        return np.linalg.norm(z - centroid, axis=1, ord=2)
        
    def _get_hypercube_norm(self, z):
        # Calculate the volume of the hypercube defined by the 
        # max/min of each latent state variable, 
        # return the 16th root (= z_dim).
        z_dim = z.shape[2]
        mins = np.amin(z, axis=(0, 1))
        maxs = np.amax(z, axis=(0, 1))
        ranges = maxs - mins
        return np.prod(ranges ** (1 / z_dim))

    def analyze(self):
        stats = {}

        # Get correlation between distance to task centroid and RT
        all_r = []
        num_low_N = 0
        num_constant = 0
        for c in self.combos:
            mv, pt, cue = c
            filt = {'mv_dir': mv, 'point_dir': pt, 'task_cue': cue,
                    'prev_task_cue': 1 - cue}
            trial_inds = self.expt_stats.select(**filt)
            if cue == 0:
                c_centroid = self.mv_centroid
            else:
                c_centroid = self.pt_centroid
            this_dists = self._get_trial_distances(trial_inds, c_centroid)
            rts = self.expt_stats.df['mrt_ms'][trial_inds]
            if len(trial_inds) >= self.min_N:
                combo_r, _ = pearsonr(this_dists, rts)
                if np.isnan(combo_r): # Happens when RT array is constant (very rare)
                    num_constant += 1
                else:
                    all_r.append(combo_r)
            else:
                num_low_N += 1
        stats['all_r'] = all_r
        stats['mean_r'] = np.mean(all_r) 
        stats['num_low_N'] = num_low_N
        stats['num_constant'] = num_constant

        # Get distance between task centroids
        norm_factor = self._get_hypercube_norm(self.expt_stats.latents)
        centroid_dist = np.linalg.norm(self.mv_centroid - self.pt_centroid, ord=2)
        stats['normed_centroid_dist'] = centroid_dist / norm_factor

        return stats
