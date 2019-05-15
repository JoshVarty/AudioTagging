#Train on entire dataset
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import PIL
import fastai
from fastai.basic_train import _loss_func2activ
from fastai.vision import Path, ImageList, imagenet_stats, cnn_learner, get_transforms, DatasetType, models, load_learner, fbeta, Image, pil2tensor, Learner, get_preds
import sklearn.metrics
from functools import partial
import torch

NFOLDS = 5
script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)
print("Model: {}".format(MODEL_NAME))

# Make required folders if they're not already present
directories = ['../kfolds', '../model_predictions', '../model_source']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Competition Metric (I think)
def calculate_overall_lwlrap_sklearn(scores, truth):
    # Calculate the overall lwlrap using sklearn.metrics.lrap.
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(truth > 0, scores)
    
    return torch.Tensor([overall_lwlrap])

#Overrides fastai's default 'open_image' method to crop based on our crop counter
def setupNewCrop(counter):
    
    def open_fat2019_image(fn, convert_mode, after_open)->Image:
        
        x = PIL.Image.open(fn).convert(convert_mode)

        # crop (128x321 for a 5 second long audio clip)
        time_dim, base_dim = x.size
        
        #How many crops can we take?
        maxCrops = int(np.ceil(time_dim / base_dim))
        
        #What's the furthest point at which we can take a crop without running out of pixels
        lastValidCrop = time_dim - base_dim
        crop_x = (counter % maxCrops) * base_dim 

        # We don't want to crop any further than the last 128 pixels
        crop_x = min(crop_x, lastValidCrop)

        x1 = x.crop([crop_x, 0, crop_x+base_dim, base_dim])    
        
        # standardize    
        return Image(pil2tensor(x1, np.float32).div_(255))

    fastai.vision.data.open_image = open_fat2019_image

def custom_tta(learn:Learner, ds_type:DatasetType=DatasetType.Valid):
    dl = learn.dl(ds_type)
    ds = dl.dataset

    old_open_image = fastai.vision.data.open_image
    try:
        maxNumberOfCrops = 10
        for i in range(maxNumberOfCrops):
            #print("starting")
            setupNewCrop(i)
            yield get_preds(learn.model, dl, activ=_loss_func2activ(learn.loss_func))[0]
    finally:
            fastai.vision.data.open_image = old_open_image


DATA = Path('/home/josh/git/AudioTagging/data')
WORK = Path('/home/josh/git/AudioTagging/work')

CSV_TRN_MERGED = DATA/'train_merged.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'

TRN_CURATED = DATA/'train_curated'
TRN_NOISY = DATA/'train_noisy'

IMG_TRN_CURATED = WORK/'image/trn_curated'
IMG_TRN_NOISY = WORK/'image/trn_noisy'
IMG_TEST = WORK/'image/test'

TEST = DATA/'test'

print("Reading training data")
train = pd.read_csv(DATA/'train_curated.csv')
test = pd.read_csv(DATA/'sample_submission.csv')
train_noisy = pd.read_csv(DATA/'train_noisy.csv')

X = train['fname']
y = train['labels'].apply(lambda f: f.split(','))
y_noisy = train_noisy['labels'].apply(lambda f: f.split(','))
transformed_y = MultiLabelBinarizer().fit_transform(y)
transformed_y_noisy = MultiLabelBinarizer().fit_transform(y_noisy)
filenames = train['fname'].values
filenames = filenames.reshape(-1, 1)

oof_preds = np.zeros((len(train), 80))
test_preds = np.zeros((len(test), 80))

tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)
mskf = MultilabelStratifiedKFold(n_splits=5, random_state=4, shuffle=True)
df = pd.read_csv(CSV_TRN_MERGED)
cols = list(df.columns[1:])
i = 0
for _, val_index in mskf.split(X, transformed_y):

    #Our clasifier stuff    
    src = (ImageList.from_csv(WORK/'image', Path('../../')/DATA/'train_curated.csv', folder='trn_merged', suffix='.jpg')
        .split_by_idx(val_index)
       #.label_from_df(cols=list(df.columns[1:])))
       .label_from_df(label_delim=','))

    data = (src.transform(tfms, size=128).databunch(bs=64).normalize())

    f_score = partial(fbeta, thresh=0.2)
    learn = cnn_learner(data, models.xresnet101, pretrained=False, metrics=[f_score]).mixup(stack_y=False)
    learn.fit_one_cycle(150, 1e-2)

    all_preds = list(custom_tta(learn))
    val_preds = torch.stack(all_preds).mean(0)

    oof_preds[val_index, :] = val_preds

    #Save learner
    learn.export(fname=MODEL_NAME + '_' + str(i))
    i = i + 1

score = calculate_overall_lwlrap_sklearn(oof_preds, transformed_y).numpy()[0]

print("Score:", score)

print("Saving out-of-fold predictions...")
all_oof_preds = pd.DataFrame(np.hstack((filenames, oof_preds)), columns = test.columns)
all_oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(score)), index=False)

print("Saving code...")
shutil.copyfile(os.path.basename(__file__), '../model_source/{}__{}.py'.format(MODEL_NAME, str(score)))
