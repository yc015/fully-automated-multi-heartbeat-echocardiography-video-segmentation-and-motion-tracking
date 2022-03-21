'''
Utilities for dealing with the CAMUS echo dataset.
https://www.creatis.insa-lyon.fr/Challenge/camus/
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .camus_config import CAMUS_CONFIG

pathconfig = CAMUS_CONFIG['paths']

# Want to automatically expand the tilde in source folder paths if necessary.
# Thank you: https://stackoverflow.com/questions/2057045/pythons-os-makedirs-doesnt-understand-in-my-path
CAMUS_DIR = os.path.expanduser(pathconfig['CAMUS_DIR'])
CAMUS_TRAINING_DIR = os.path.expanduser(pathconfig['CAMUS_TRAINING_DIR'])
CAMUS_TESTING_DIR = os.path.expanduser(pathconfig['CAMUS_TESTING_DIR'])
CAMUS_RESULTS_DIR = os.path.expanduser(pathconfig['CAMUS_RESULTS_DIR'])


'''
make_camus_echo_dataset(rootpath): 
This will be initial call. returns a dictionary, one value for 
for each patient (since each patient has multiple pics and we wouldn't want
to split without that info). The value is a list of dictionaries as in the
Nvidia VNET example (one dictionary for each image and label pair. 
split_camus_echo will actually generate the 
training and validation sets as in the nvidia example. 
'''
def make_camus_echo_dataset(rootpath=CAMUS_TRAINING_DIR, view = '2CH'):
    dataByPat = {}
    
#     for root, subdirs, files in os.walk(rootpath):
#         templist = [os.path.join(root, filename) for filename in files if 'mhd' in filename]
#         dataset.extend([f for f in templist if chamber in f and 'seq' not in f])

    patients = os.listdir(rootpath)
    
    for pat in patients:
        subdir = os.path.join(rootpath, pat)
        files = os.listdir(subdir)
        if len(files) > 0:
            dataset = []
            for phase in ['ED', 'ES']:
                dataset.append({
                    'pat_view_phase': [(pat, view, phase)],
                    'images': [os.path.join(subdir, '{}_{}_{}.mhd'.format(pat, view, phase))],
                    'labels': [os.path.join(subdir, '{}_{}_{}_gt.mhd'.format(pat, view, phase))]
                })
            dataByPat[pat] = list(dataset)
            
        
    return dataByPat


'''
make_camus_labelset(rootpath):
Necessary for the shape autoencoding.
This will be initial call. returns a dictionary, one value for 
for each patient (since each patient has multiple pics and we wouldn't want
to split without that info). The value is a list of dictionaries as in the
Nvidia VNET example (one dictionary for each label and label pair. 
split_camus_echo will actually generate the 
training and validation sets as in the nvidia example. 
'''
def make_camus_labelset(rootpath=CAMUS_TRAINING_DIR, view = '2CH'):
    dataByPat = {}
    
#     for root, subdirs, files in os.walk(rootpath):
#         templist = [os.path.join(root, filename) for filename in files if 'mhd' in filename]
#         dataset.extend([f for f in templist if chamber in f and 'seq' not in f])

    patients = os.listdir(rootpath)
    
    for pat in patients:
        subdir = os.path.join(rootpath, pat)
        files = os.listdir(subdir)
        if len(files) > 0:
            dataset = []
            for phase in ['ED', 'ES']:
                dataset.append({
                    'pat_view_phase': [(pat, view, phase)],
                    'inputs': [os.path.join(subdir, '{}_{}_{}_gt.mhd'.format(pat, view, phase))],
                    'outputs': [os.path.join(subdir, '{}_{}_{}_gt.mhd'.format(pat, view, phase))]
                })
            dataByPat[pat] = list(dataset)
            
    return dataByPat

    
'''
split_camus_echo(dataByPat, train_idx, val_idx=None): return appropriate lists
of dictionaries as in the Nvidia example. The idx's here are lists of patient index,
like 'patient0001'.
'''
def split_camus_echo(dataByPat, train_idx, val_idx=[], test_idx=[]):
    trainDataset = []
    valDataset = []
    testDataset = []
    
    for pat in train_idx:
        trainDataset.extend(dataByPat[pat])
    
    for pat in val_idx:
        valDataset.extend(dataByPat[pat])
    
    for pat in test_idx:
        testDataset.extend(dataByPat[pat])
            
    return trainDataset, valDataset, testDataset


'''
make_camus_EHR(rootpath): return a dataframe indexed by patient, built 
on the Info_[2,4]CH.cfg data in the rootpath. If called with the testing
rootpath, the dataframe returned won't have LVedv, LVef, LVesv. 
'''
def make_camus_EHR(rootpath=CAMUS_TRAINING_DIR, view='2CH'):
    # Hardcode known column names. Again, testing is missing LV*
    colTypes = {'Age':'float', 
                'ED':'int', 
                'ES':'int', 
                'ImageQuality':'object', 
                'LVedv':'float', 'LVef':'float', 'LVesv':'float', 
                'NbFrame':'int', 
                'Sex':'object'}
    
    
    df = pd.DataFrame()
    
    infoFilename = 'Info_{}.cfg'.format(view)
    
    patients = os.listdir(rootpath)
    
    for pat in patients:
        subdir = os.path.join(rootpath, pat)
        files = os.listdir(subdir)
        if len(files) > 0:
            with open(os.path.join(subdir, infoFilename), 'r') as thefile:
                datums = thefile.read().split('\n')
                df = df.append(pd.DataFrame(dict([dat.split(': ') for dat in datums[:-1]]), index=[pat]))
                
    
    # We can't do the type conversion if the column doesn't exist
    # i.e., the test cases don't include LV*.
    colAcquired = df.columns.to_list()
    print('make_camus_EHR: found columns: {}'.format(colAcquired))
    for col in list(colTypes.keys()):
        if col not in colAcquired:
            print('make_camus_EHR: Ignoring missing column {}'.format(col))
            colTypes.pop(col, None);
            
    
    # Modify the types according to the columns.
    df = df.astype(colTypes)
    # Recast sex in particular.
    
    df.loc[:, ('Sex')] = (df.loc[:, ('Sex')] == 'F') * 1
    
    
    return df


# Obviated
# def make_camus_splits(rootpath=CAMUS_TRAINING_DIR, view='2CH'):
#     ehr = make_camus_EHR(CAMUS_TRAINING_DIR, VIEW)
#     camusByPat = make_camus_echo_dataset(CAMUS_TRAINING_DIR, VIEW)

#     # A single gender-stratified split on the data.
#     ss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_split+test_split))
#     # I think I'm probably abusing the sss. Need
#     # https://stackoverflow.com/questions/4741243/how-to-pick-just-one-item-from-a-generator-in-python
#     train_idx, valtest_idx = next(ss1.split(ehr.index.values, ehr.Sex.values))

#     # since new indices will be generated, i need these now. Otherwise, I'll have 
#     # val and test within the training set..
#     valtest_pats = ehr.index.values[valtest_idx]

#     ss2 = StratifiedShuffleSplit(n_splits=1, test_size=(val_split/(val_split+test_split)))
#     val_idx, test_idx = next(ss2.split(valtest_pats, ehr.Sex.values[valtest_idx]))


#     train_idx = ehr.index.values[train_idx]
#     val_idx = valtest_pats[val_idx] # not ehr.index.values[val_idx], since new indices have been generated.
#     test_idx = valtest_pats[test_idx] # ehr.index.values[test_idx]


#     print('Found {} ({} F) patients for training, \n'\
#           '{} ({} F) for validation, {} ({} F) for test.'.\
#           format(len(train_idx), ehr.Sex.loc[train_idx].sum(),
#                  len(val_idx), ehr.Sex.loc[val_idx].sum(),
#                  len(test_idx), ehr.Sex.loc[test_idx].sum()))


#     training_dataset, validation_dataset, test_dataset = \
#         split_camus_echo(camusByPat, 
#                          train_idx=train_idx, 
#                          val_idx=val_idx,
#                          test_idx=test_idx)

#     print('{} training images, {} validation, {} test'.\
#           format(len(training_dataset), len(validation_dataset), len(test_dataset)))
    

def camus_generate_folds(nfolds=10, random_state=None):
    '''
    camus_generate_folds: As in LeClerc, I split the patients roughly maintaining
    the image quality (good, medium, poor) and ejection fraction ranges 
    (<=45, >=55, other). 
    
    The EFs match between 2CH and 4CH, but the ImageQuality doesn't .
    I'm going to hardcode the 2CH as the view to use to split patients, as I 
    believe the paper does.
    '''
    print('camus_generate_folds: generating {} folds '\
          'stratified on ImageQuality and LVef.'.format(nfolds))
    
    ehr = make_camus_EHR(rootpath=CAMUS_TRAINING_DIR, view='2CH')
    
    # Now we'll add a column to do the stratification on,
    # having to do with both image quality and ef range.
    
    # Add a column that specifies the low/medium/high ef bin.
    ehr['efbin'] = 1
    ehr.loc[(ehr.LVef <= 45), 'efbin'] = 0
    ehr.loc[(ehr.LVef >= 55), 'efbin'] = 2
    
    # Thank you https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-dataframe-in-pandas-python
    ehr['combinedQualEF'] = ehr.ImageQuality + ehr.efbin.map(str)
    
    ############
    # Now we're ready to do an initial stratified kfold.
    ############
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    kf = []

    print('               :Good Med Poor;  <45  >55  else')

    for k, (train_idx, test_idx) in enumerate(skf.split(ehr.index.values, ehr.combinedQualEF.values)):

        foldquality = ehr.ImageQuality.iloc[test_idx].value_counts().values
        foldquality = np.round(100*(foldquality/np.sum(foldquality))).astype('int')

        lvefs = ehr.LVef.values
        efbins = [(ehr.LVef.iloc[test_idx].values<=45).sum(),
                  (ehr.LVef.iloc[test_idx].values>=55).sum()]
        efbins = np.array(efbins+[0])
        efbins = efbins/len(test_idx)
        efbins[-1] = 1-np.sum(efbins[:2])
        efbins = np.round(100*efbins).astype('int')

        print('fold {} ({}):    {}   {}   {};    {}   {}   {}'.format(k, len(test_idx),\
                foldquality[0], foldquality[1], foldquality[2],\
                efbins[0], efbins[1], efbins[2]))

        kf.append((ehr.index[train_idx].values, ehr.index[test_idx].values))
        
    
    ###########
    # Now roll back through the folds with train and test, and get validation sets
    # for each. The validation set for one fold will just be the test set of another
    # fold.
    ###########
    
    for k, (train_pats, test_pats) in enumerate(kf):
        val_pats = kf[(k+1)%10][1] # The validation for this fold is the test of the next fold.
        train_pats = np.array(list(set(train_pats) - set(val_pats)))

        print('Fold {}: taking {} from training for validation.'.format(k, len(val_pats)))

        assert len(set(train_pats) | set(val_pats) | set(test_pats))==len(ehr),\
            'Error: fold did not contain all cases in train/val/test combined.\n'
        
        assert not (set(train_pats) & set(val_pats)) and \
                not (set(val_pats) & set(test_pats)) and \
                not (set(train_pats) & set(test_pats)),\
            'Error: non-empty intersection among train/val/test'
        
        kf[k] = (train_pats, val_pats, test_pats)
        
    return kf, ehr
        
    