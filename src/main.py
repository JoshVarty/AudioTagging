import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from fastai.vision import *
import sklearn.metrics

# Competition Metric (I think)
def calculate_overall_lwlrap_sklearn(scores, truth):
    # Calculate the overall lwlrap using sklearn.metrics.lrap.
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(truth > 0, scores)
    
    return torch.Tensor([overall_lwlrap])

NFOLDS = 5
RANDOM_STATE = 42

DATA = Path('data')
CSV_TRN_CURATED = DATA/'train_curated.csv'
CSV_TRN_NOISY = DATA/'train_noisy.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'
TRN_CURATED = DATA/'train_curated'
TRN_NOISY = DATA/'train_noisy'
TEST = DATA/'test'
WORK = Path('work')
IMG_TRN_CURATED = WORK/'image/trn_curated'
IMG_TRN_NOISY = WORK/'image/trn_noisy'
IMG_TEST = WORK/'image/test'
for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]: 
    Path(folder).mkdir(exist_ok=True, parents=True)

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

print("Model: {}".format(MODEL_NAME))

print("Reading training data")
train = pd.read_csv('../data/train_clean.csv')
#test = pd.read_csv('data/test.csv')

X = train['fname']
y = train['labels'].apply(lambda f: f.split(','))
transformed_y = MultiLabelBinarizer().fit_transform(y)

mskf = MultilabelStratifiedKFold(n_splits=5, random_state=4, shuffle=True)

tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)

for train_index, val_index in mskf.split(X, transformed_y):

    #Our clasifier stuff    
    src = (ImageList.from_csv('../'/WORK/'image', Path('../../')/CSV_TRN_CURATED, folder='trn_curated', suffix='.jpg')
       .split_by_idx(val_index)
       .label_from_df(label_delim=','))


    data = (src.transform(tfms, size=128).databunch(bs=64).normalize(imagenet_stats))

    f_score = partial(fbeta, thresh=0.2)
    learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[f_score, calculate_overall_lwlrap_sklearn])
    learn.fit_one_cycle(5, slice(1e-6, 1e-1))
    learn.unfreeze()
    learn.fit_one_cycle(100, slice(1e-6, 1e-1))




