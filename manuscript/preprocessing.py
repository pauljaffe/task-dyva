import os

import numpy as np
import pandas as pd

from task_dyva import Experiment


class Preprocess():
    """Run preprocessing for the manuscript:
    get model outputs from all models with different levels
    of stimulus noise (using the holdout data). 
    """

    analysis_dir = 'model_analysis'
    device = 'cpu'
    raw_fn = 'data_pre_split.pickle'
    params_fn = 'model_params.pth'
    acc_noise_sds = np.arange(0.1, 0.65, 0.05)
    acc_noise_keys = ['01', '015', '02', '025', '03', '035', '04', '045',
                      '05', '055', '06']
    sc_noise_sds = np.array([0.7, 0.8, 0.9, 1.0])
    sc_noise_keys = ['07', '08', '09', '1']

    def __init__(self, model_dir, metadata, load_saved_analysis=True):
        self.model_dir = model_dir
        self.expts = metadata['name']
        self.age_bins = metadata['age_range']
        self.user_ids = metadata['user_id']
        self.sc_status = metadata['switch_cost_type']
        self.load_saved = load_saved_analysis

    def run_preprocessing(self):
        for expt_str, ab, uid, sc in zip(self.expts, 
                                         self.age_bins,
                                         self.user_ids, 
                                         self.sc_status):

            print(f'Preprocessing experiment {expt_str}')
            this_model_dir = os.path.join(self.model_dir, expt_str)
            raw_dir = this_model_dir
            this_analysis_dir = os.path.join(this_model_dir, self.analysis_dir)

            # Generate responses with different noise conditions
            this_noise_keys = self.acc_noise_keys.copy()
            if sc in ['sc+', 'sc-']:
                this_noise_cons = np.concatenate((self.acc_noise_sds, 
                                                  self.sc_noise_sds))
                this_noise_keys.extend(self.sc_noise_keys)
            else:
                this_noise_cons = self.acc_noise_sds

            for sd, key in zip(this_noise_cons, this_noise_keys):
                this_save_str = f'holdout_outputs_{key}SD.pkl'
                this_noise_params = {'noise_type': 'indep',
                                     'noise_sd': sd}
                expt_kwargs = {'do_logging': False, 
                               'test': this_noise_params,
                               'mode': 'testing', 
                               'params_to_load': self.params_fn}

                if key == '01':
                    # Noise level used for training
                    analyze_latents = True
                else:
                    analyze_latents = False

                # Get model outputs and stats
                expt = Experiment(this_model_dir, raw_dir, self.raw_fn, 
                                  expt_str, processed_dir=this_model_dir,
                                  device=self.device, **expt_kwargs)

                expt_stats = expt.get_behavior_metrics(expt.test_dataset, 
                                                       this_save_str,
                                                       save_local=True,
                                                       load_local=self.load_saved,
                                                       analyze_latents=analyze_latents,
                                                       stats_dir=self.analysis_dir)
