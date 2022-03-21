'''
Joshua Stough, 6/19

Configurations for camus experiment.
'''

CAMUS_CONFIG = {}

_tmp = {
    'CAMUS_DIR': '~/data/CAMUS/', # non-docker run
#     'CAMUS_DIR': '/data/CAMUS/', # docker run
    # 'CAMUS_DIR': '/ghsdisigpulx1/jvstough/CAMUS/',
}
_tmp['CAMUS_TRAINING_DIR'] = _tmp['CAMUS_DIR'] + 'training/'
_tmp['CAMUS_TESTING_DIR'] = _tmp['CAMUS_DIR'] + 'testing/'
_tmp['CAMUS_RESULTS_DIR'] = _tmp['CAMUS_DIR'] + 'results/'
_tmp['folds_file'] = 'folds.pkl'
_tmp['midl_folds'] = 'midl_folds.pkl'
_tmp['ehr_file'] = 'ehr.pkl'

CAMUS_CONFIG['paths'] = dict(_tmp)

CAMUS_CONFIG['training'] = {
    'num_epochs': 300, # training parameters.
    'batch_size': 16,
    'effective_batchsize': 1,
    'learning_rate': 0.002, #0.001, # 0.002,
    'weight_decay': 1e-6,
    'patienceLimit': 41,
    'patienceToLRcut': 10,
    'howOftenToReport': 10,
    'loss_weights': [1,1,1,1],
    'image_size': [256, 256],
}

CAMUS_CONFIG['augment'] = {
    'windowing_scale': [.5, 1.0], # augmentation parameters, see camus_transforms.
    'rotation_scale': 5.0,
    'noise_scale': [0.0, 0.15],
    'training_augment': True, #True,
}

CAMUS_CONFIG['acnn'] = { # ACNN-specific configurations.
    'ae_salt_pepper_freq': 0.1,
    'ae_encoding_dim': 64,
    'ae_initial_filters': 4,
    'acnn_shape_prior_weight': 0.001,
    'acnn_learning_rate': 0.01,
}

CAMUS_CONFIG['unet'] = { # unetlike specific configurations.
    'n_channels': 1, 
    'n_classes': 4, 
    'n_filters': 32, 
    'normalization': 'groupnorm', # ('none'|'batchnorm'|'groupnorm')
}

