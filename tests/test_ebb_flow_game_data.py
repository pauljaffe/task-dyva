# Test EbbFlowGameData
import numpy as np
import pytest

from task_dyva import EbbFlowGameData


def create_test_data():
    # Create synthetic game data for testing
    n_trials = 4
    rsi = 25  # ms
    start_time = 30000  # ms
    uids = n_trials * [1]
    game_ids = n_trials * [1]
    nth_play = np.arange(162, 162 + n_trials)
    mv = ['D', 'R', 'U', 'L']
    pt = ['D', 'L', 'U', 'R']
    cue = ['M', 'M', 'P', 'P']
    resp = ['D', 'U', 'L', 'R']
    rts = [501, 812, 1236, 5000]
    offsets = start_time + np.cumsum(rts[:-1]) + rsi * np.arange(1, 4)
    offsets = np.insert(offsets, 0, start_time)

    test_data = {'user_id': uids, 'game_result_id': game_ids,
                 'nth_master': nth_play, 'mv_dir': mv, 'point_dir': pt,
                 'urespdir': resp, 'urt_ms': rts, 'task_cue': cue,
                 'time_offset': offsets}

    expected = {'is_congruent': [0, 1, 0],
                'is_switch': [np.nan, 1, 0],
                'correct_dir': [1, 2, 1],
                'urespdir': [2, 0, 1],
                'ucorrect': [0, 0, 1],
                'urt_ms': rts[1:],
                'trial_type': [1, 2, 1]}

    params = {'step_size': 20, 'duration': 15000, 'post_resp_buffer': 500}

    return test_data, expected, params


test_conditions = [
    (30400, True, 'full', 3, 3),
    (30400, False, 'valid', 3, 4),
    (1000, False, 'valid', 0, 1),
    (50000, False, 'valid', 0, 1)
]


@pytest.mark.parametrize(
    'start_time, is_valid, check_type, n_trials, min_trials',
    test_conditions
)
def test_ebb_flow_game_data(start_time, is_valid, check_type, n_trials, 
                            min_trials):
    data, expected, params = create_test_data()
    params['min_trials'] = min_trials
    game = EbbFlowGameData.preprocessed_format(data, params, start_time)
    game.standard_prep()
    game.get_extra_stats()
    assert game.is_valid == is_valid
    assert len(game.discrete['mv_dir']) == n_trials
    if check_type == 'full':
        for key in expected.keys():
            assert expected[key] == game.discrete[key]
