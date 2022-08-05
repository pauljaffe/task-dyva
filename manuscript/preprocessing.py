import os
import copy
import pickle

import numpy as np
import pandas as pd

from task_dyva import Experiment
from task_dyva.model_analysis import FixedPointFinder, LatentsLDA, LatentSeparation


class Preprocess():
    """Run preprocessing for the manuscript:
    1) Get the model outputs on the test set at different noise levels.
    2) Get model and behavior summary metrics
    3) Find stable fixed points for each model.
    4) Run the LDA analyses (Fig. 3).
    """

    analysis_dir = 'model_analysis'
    device = 'cpu'
    raw_fn = 'data_pre_split.pickle'
    params_fn = 'model_params.pth'

    # Noise conditions
    all_noise_sds = np.arange(0.1, 0.65, 0.05)
    all_noise_sds = np.append(all_noise_sds, [0.7, 0.8, 0.9, 1.0])
    all_noise_keys = ['01', '015', '02', '025', '03', '035', '04', '045',
                      '05', '055', '06', '07', '08', '09', '1']
    latents_noise_keys = ['01', '02', '03', '04' ,'05', '06', '07', '08', '09', '1']
    primary_noise_key = '01'
    primary_noise_sd = 0.1

    # Model and behavior summary
    outputs_save_str = 'holdout_outputs'
    summary_fn = 'summary.pkl'
    metrics = ['accuracy', 'acc_switch_cost', 'acc_con_effect',
               'mean_rt', 'switch_cost', 'con_effect',
               'normed_centroid_dist']

    # Fixed point params
    fp_fn = 'fixed_points.pkl'
    fp_summary_fn = 'fixed_point_summary.pkl'
    fp_N = 10
    fp_T = 50000

    # LDA params
    lda_fn = 'lda_summary.pkl'
    lda_time_range = [-100, 1600]
    lda_n_shuffle = 100

    def __init__(self, model_dir, metadata, rand_seed, batch_size=None):
        self.model_dir = model_dir
        self.expts = metadata['name']
        self.sc_status = metadata['switch_cost_type']
        self.exgauss = metadata['exgauss']
        self.batch_size = batch_size

    def run_preprocessing(self):
        for expt_str, model_type, exg in zip(self.expts,
                                             self.sc_status,
                                             self.exgauss):

            print(f'Preprocessing experiment {expt_str}')
            this_model_dir = os.path.join(self.model_dir, expt_str)
            # Get model outputs
            self._get_outputs_wrapper(this_model_dir, expt_str, exg)
            # Get model and behavior summary metrics
            # (including distance b/w task centroids)
            self._get_summary(this_model_dir, model_type, exg)
            # Find stable fixed points
            self._fp_wrapper(this_model_dir, expt_str, model_type, exg)
            # LDA analysis
            self._lda_wrapper(this_model_dir, model_type, exg)
            # Clean up
            self._clean_up(this_model_dir, exg)

    def _get_outputs_wrapper(self, model_dir, expt_str, exgauss):
        if exgauss == 'exgauss+':
            noise_keys = [self.primary_noise_key]
            noise_sds = [self.primary_noise_sd]
        else:
            noise_keys = self.all_noise_keys
            noise_sds = self.all_noise_sds

        for noise_key, noise_sd in zip(noise_keys, noise_sds):
            if noise_key in self.latents_noise_keys:
                analyze_latents = True
            else:
                analyze_latents = False

            self._get_model_outputs(model_dir, 
                                    expt_str,
                                    noise_key,
                                    noise_sd,
                                    analyze_latents)

    def _get_model_outputs(self, model_dir, expt_str, 
                           noise_key, noise_sd, analyze_latents):
        save_str = f'{self.outputs_save_str}_{noise_key}SD.pkl'
        noise_params = {'noise_type': 'indep',
                        'noise_sd': noise_sd}
        expt_kwargs = {'logger_type': None,
                       'test': noise_params,
                       'mode': 'testing', 
                       'params_to_load': self.params_fn}

        # Get model outputs, save
        expt = Experiment(model_dir, model_dir, self.raw_fn, 
                          expt_str, processed_dir=model_dir,
                          device=self.device, **expt_kwargs)

        _ = expt.get_behavior_metrics(expt.test_dataset, 
                                      save_fn=save_str,
                                      save_local=True,
                                      analyze_latents=analyze_latents,
                                      stats_dir=self.analysis_dir,
                                      batch_size=self.batch_size)

    def _get_summary(self, model_dir, model_type, exgauss):
        if exgauss == 'exgauss+':
            noise_keys = [self.primary_noise_key]
            noise_sds = [self.primary_noise_sd]
        else:
            noise_keys = self.all_noise_keys
            noise_sds = self.all_noise_sds

        summary = {key: {} for key in noise_keys}
        for noise_key, noise_sd in zip(noise_keys, noise_sds):
            outputs = self._reload_outputs(model_dir, noise_key)
            if noise_key in self.latents_noise_keys:
                latent_sep = LatentSeparation(outputs)
                dist_stats = latent_sep.analyze()

            for m in self.metrics:
                if m == 'normed_centroid_dist':
                    summary[noise_key][m] = dist_stats[m]
                else:
                    u_key = f'u_{m}'
                    m_key = f'm_{m}'
                    summary[noise_key][u_key] = outputs.summary_stats[u_key]
                    summary[noise_key][m_key] = outputs.summary_stats[m_key]

            if model_type in ['sc+', 'sc-']:
                # Get conditional error rates
                error_info = self._get_error_info(outputs.df)
                summary[noise_key].update(error_info)

        # Save
        save_path = os.path.join(model_dir, 
                                 self.analysis_dir, 
                                 self.summary_fn)
        with open(save_path, 'wb') as path:
            pickle.dump(summary, path, protocol=4)

    def _get_error_info(self, df):
        errors = {}

        n_con = len(df.query('is_congruent == 1'))
        n_con_errors = len(df.query('is_congruent == 1 and mcorrect == 0'))
        errors['con_error_rate'] = n_con_errors / n_con

        n_incon = len(df.query('is_congruent == 0'))
        n_incon_errors = len(df.query('is_congruent == 0 and mcorrect == 0'))
        errors['incon_error_rate'] = n_incon_errors / n_incon

        n_stay = len(df.query('is_switch == 0'))
        n_stay_errors = len(df.query('is_switch == 0 and mcorrect == 0'))
        errors['stay_error_rate'] = n_stay_errors / n_stay

        n_switch = len(df.query('is_switch == 1'))
        n_switch_errors = len(df.query('is_switch == 1 and mcorrect == 0'))
        errors['switch_error_rate'] = n_switch_errors / n_switch

        return errors

    def _fp_wrapper(self, model_dir, expt_str, model_type, exgauss):
        if exgauss == 'exgauss+':
            return

        fp_path = os.path.join(model_dir,
                               self.analysis_dir,
                               self.fp_fn)
        fp_summary_path = os.path.join(model_dir,
                                       self.analysis_dir,
                                       self.fp_summary_fn)
        if model_type != 'sc-':
            noise_params = {'noise_type': 'indep',
                            'noise_sd': self.primary_noise_sd}
            expt_kwargs = {'logger_type': None,
                           'test': noise_params,
                           'mode': 'testing', 
                           'params_to_load': self.params_fn}
            expt = Experiment(model_dir, model_dir, self.raw_fn, 
                              expt_str, processed_dir=model_dir,
                              device=self.device, **expt_kwargs)
            outputs = self._reload_outputs(model_dir,
                                           self.primary_noise_key)

            fpf = FixedPointFinder(expt, outputs, fp_path,
                                   fp_summary_path,
                                   load_saved=False,
                                   rand_seed=self.rand_seed)
            this_fps = fpf.find_fixed_points(self.fp_N, self.fp_T)
            fp_summary = fpf.get_fixed_point_summary(this_fps)

    def _lda_wrapper(self, model_dir, model_type, exgauss):
        if exgauss == 'exgauss+':
            return

        lda_path = os.path.join(model_dir,
                                self.analysis_dir,
                                self.lda_fn)
        if model_type != 'sc-':
            outputs = self._reload_outputs(model_dir,
                                           self.primary_noise_key)
            lda = LatentsLDA(outputs, lda_path, load_saved=False,
                             time_range=self.lda_time_range,
                             n_shuffle=self.lda_n_shuffle,
                             rand_seed=self.rand_seed)
            lda_summary = lda.run_lda_analysis()

    def _reload_outputs(self, model_dir, noise_key):
        outputs_fn = f'{self.outputs_save_str}_{noise_key}SD.pkl'
        outputs_path = os.path.join(model_dir, self.analysis_dir, 
                                    outputs_fn)
        with open(outputs_path, 'rb') as path:
            outputs = pickle.load(path)
        return outputs

    def _clean_up(self, model_dir, exgauss):
        if exgauss == 'exgauss+':
            return

        for noise_key in self.all_noise_keys:
            if noise_key != self.primary_noise_key:
                outputs_fn = f'{self.outputs_save_str}_{noise_key}SD.pkl'
                outputs_path = os.path.join(model_dir, self.analysis_dir, 
                                            outputs_fn)
                os.remove(outputs_path)
