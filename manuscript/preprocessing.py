import os
import copy
import pickle

import numpy as np
import pandas as pd

from task_dyva import Experiment
from task_dyva.model_analysis import FixedPointFinder, LatentsLDA


class Preprocess():
    # TODO: finish docstring
    """Run preprocessing for the manuscript:
    1) COMPLETE
    of stimulus noise (using the holdout data). 
    2) Find stable fixed points for each model.
    """

    analysis_dir = 'model_analysis'
    device = 'cpu'
    raw_fn = 'data_pre_split.pickle'
    params_fn = 'model_params.pth'
    rand_seed = 12345 # Enforce reproducibility

    # Noise conditions
    acc_noise_sds = np.arange(0.1, 0.65, 0.05)
    acc_noise_keys = ['01', '015', '02', '025', '03', '035', '04', '045',
                      '05', '055', '06']
    sc_noise_sds = np.array([0.7, 0.8, 0.9, 1.0])
    sc_noise_keys = ['07', '08', '09', '1']
    primary_noise_key = '01'
    primary_noise_sd = 0.1
    primary_outputs_fn = f'holdout_outputs_{primary_noise_key}SD.pkl'

    # Behavior summary
    behavior_summary_fn = 'behavior_summary.pkl'
    behavior_metrics = ['accuracy', 'acc_switch_cost', 'acc_con_effect',
                        'mean_rt', 'switch_cost', 'con_effect']

    # Fixed point params
    fp_fn = 'fixed_points.pkl'
    fp_summary_fn = 'fixed_point_summary.pkl'
    fp_N = 10
    fp_T = 50000

    # LDA params
    lda_fn = 'lda_summary.pkl'
    lda_time_range = [-100, 1600]
    lda_n_shuffle = 1000

    def __init__(self, model_dir, metadata, reload_primary_outputs=True,
                 reload_behavior_summary=True, reload_fixed_points=True,
                 reload_lda_summary=True):
        self.model_dir = model_dir
        self.expts = metadata['name']
        self.sc_status = metadata['switch_cost_type']
        self.reload_primary = reload_primary_outputs
        self.reload_behavior = reload_behavior_summary
        self.reload_fixed_points = reload_fixed_points
        self.reload_lda_summary = reload_lda_summary

    def run_preprocessing(self):
        for expt_str, model_type in zip(self.expts, 
                                        self.sc_status):
            print(f'Preprocessing experiment {expt_str}')
            this_model_dir = os.path.join(self.model_dir, expt_str)

            # Get model outputs used for the bulk of analyses (0.1SD noise)
            self._primary_outputs_wrapper(this_model_dir, expt_str)

            # Get model / participant behavior at all noise levels
            self._behavior_wrapper(this_model_dir, expt_str, model_type)

            # Find stable fixed points
            self._fp_wrapper(this_model_dir, expt_str, model_type)

            # LDA analyses
            self._lda_wrapper(this_model_dir, expt_str, model_type)

    def _get_model_outputs(self, model_dir, expt_str, 
                           noise_key, noise_sd, try_reload=False):
        save_str = f'holdout_outputs_{noise_key}SD.pkl'
        noise_params = {'noise_type': 'indep',
                        'noise_sd': noise_sd}
        expt_kwargs = {'do_logging': False, 
                       'test': noise_params,
                       'mode': 'testing', 
                       'params_to_load': self.params_fn}

        if noise_key == self.primary_noise_key:
            analyze_latents = True
        else:
            analyze_latents = False

        # Get model outputs and stats
        expt = Experiment(model_dir, model_dir, self.raw_fn, 
                          expt_str, processed_dir=model_dir,
                          device=self.device, **expt_kwargs)

        expt_stats = expt.get_behavior_metrics(expt.test_dataset, 
                                               save_str,
                                               save_local=True,
                                               load_local=try_reload,
                                               analyze_latents=analyze_latents,
                                               stats_dir=self.analysis_dir)
        return expt, expt_stats

    def _get_behavior_summary(self, model_dir, expt_str, model_type):
        noise_keys = self.acc_noise_keys.copy()
        if model_type in ['sc+', 'sc-']:
            noise_cons = np.concatenate((self.acc_noise_sds, 
                                         self.sc_noise_sds))
            noise_keys.extend(self.sc_noise_keys)
        else:
            noise_cons = self.acc_noise_sds

        summary = {key: {} for key in noise_keys}
        for noise_sd, noise_key in zip(noise_cons, noise_keys):
            expt, expt_stats = self._get_model_outputs(model_dir, 
                                                       expt_str,
                                                       noise_key, 
                                                       noise_sd,
                                                       try_reload=True)
            for metric in self.behavior_metrics:
                u_key = f'u_{metric}'
                m_key = f'm_{metric}'
                summary[noise_key][u_key] = expt_stats.summary_stats[u_key]
                summary[noise_key][m_key] = expt_stats.summary_stats[m_key]

        # Save and remove intermediate files
        save_path = os.path.join(model_dir, 
                                 self.analysis_dir, 
                                 self.behavior_summary_fn)
        with open(save_path, 'wb') as path:
            pickle.dump(summary, path, protocol=4)
        self._clean_up_behavior_summary(model_dir, noise_keys)

    def _clean_up_behavior_summary(self, model_dir, noise_keys):
        for key in noise_keys:
            if key == self.primary_noise_key:
                continue
            else:
                fn = os.path.join(model_dir, self.analysis_dir,
                                  f'holdout_outputs_{key}SD.pkl')
                os.remove(fn)

    def _primary_outputs_wrapper(self, model_dir, expt_str):
        primary_stats_path = os.path.join(model_dir,
                                          self.analysis_dir,
                                          self.primary_outputs_fn)
        if self.reload_primary and os.path.exists(primary_stats_path):
            pass
        else:
            _, _ = self._get_model_outputs(model_dir, 
                                           expt_str,
                                           self.primary_noise_key, 
                                           self.primary_noise_sd)

    def _behavior_wrapper(self, model_dir, expt_str, model_type): 
        behavior_summary_path = os.path.join(model_dir,
                                             self.analysis_dir,
                                             self.behavior_summary_fn)
        if self.reload_behavior and os.path.exists(behavior_summary_path):
            pass
        else:
            self._get_behavior_summary(model_dir, expt_str, model_type)

    def _fp_wrapper(self, model_dir, expt_str, model_type):
        fp_path = os.path.join(model_dir,
                               self.analysis_dir,
                               self.fp_fn)
        fp_summary_path = os.path.join(model_dir,
                                       self.analysis_dir,
                                       self.fp_summary_fn)
        if (self.reload_fixed_points and os.path.exists(fp_path)
                and os.path.exists(fp_summary_path)):
            pass
        elif model_type == 'sc-':
            pass
        else:
            expt, expt_stats = self._get_model_outputs(model_dir, 
                                                       expt_str,
                                                       self.primary_noise_key, 
                                                       self.primary_noise_sd,
                                                       try_reload=True)
            fpf = FixedPointFinder(expt, expt_stats, fp_path,
                                   fp_summary_path,
                                   load_saved=False,
                                   rand_seed=self.rand_seed)
            this_fps = fpf.find_fixed_points(self.fp_N, self.fp_T)
            fp_summary = fpf.get_fixed_point_summary(this_fps)

    def _lda_wrapper(self, model_dir, expt_str, model_type):
        lda_path = os.path.join(model_dir,
                                self.analysis_dir,
                                self.lda_fn)
        if self.reload_lda_summary and os.path.exists(lda_path):
            pass
        elif model_type == 'sc-':
            pass
        else:
            _, expt_stats = self._get_model_outputs(model_dir, 
                                                    expt_str,
                                                    self.primary_noise_key, 
                                                    self.primary_noise_sd,
                                                    try_reload=True)
            lda = LatentsLDA(expt_stats, lda_path, load_saved=False,
                             time_range=self.lda_time_range,
                             n_shuffle=self.lda_n_shuffle,
                             rand_seed=self.rand_seed)
            lda_summary = lda.run_lda_analysis()
