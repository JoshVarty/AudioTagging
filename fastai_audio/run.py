import pandas as pd
from fastai.vision import Path, ClassificationInterpretation, models
from fastai.metrics import accuracy_thresh, fbeta
from utils import my_cl_int_plot_top_losses, get_data_augmentation_transforms, get_frequency_batch_transforms, create_cnn
from functools import partial
from audio_databunch import AudioItemList

ClassificationInterpretation.plot_audio_top_losses = my_cl_int_plot_top_losses

DATA = Path('data')
CSV_TRN_CURATED = DATA/'train_curated.csv'
CSV_TRN_NOISY = DATA/'train_noisy.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'
TRN_CURATED = DATA/'train_curated'
TRN_NOISY = DATA/'train_noisy'
TEST = DATA/'test'

WORK = Path('work')
IMG_TRN_CURATED = WORK/'image/trn_curated'
IMG_TRN_NOISY = WORK/'image/trn_curated'
IMG_TEST = WORK/'image/test'
for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]: 
    Path(folder).mkdir(exist_ok=True, parents=True)

df = pd.read_csv(CSV_TRN_CURATED)
df_n = pd.read_csv(CSV_TRN_NOISY)
test_df = pd.read_csv(CSV_SUBMISSION)


n_fft = 512 # output of fft will have shape [513 x n_frames]
n_hop = 94  # width of Spectogram = max_seconds * sample rate / n_hop
n_mels = 128 # Height of spectogram
sample_rate = 48127
max_seconds = 2
f_min=0
f_max=8000
noise_scl=0.005


train_tfms = get_data_augmentation_transforms(sample_rate=sample_rate, max_seconds=max_seconds, 
                                              noise_scl=noise_scl)
valid_tfms = get_data_augmentation_transforms(sample_rate=sample_rate, max_seconds=max_seconds)

dl_tfms = get_frequency_batch_transforms(n_fft=n_fft, n_hop=n_hop,
                                            n_mels=n_mels, 
                                            f_min=f_min, f_max=f_max,
                                            sample_rate=sample_rate)



batch_size = 16

audios = (AudioItemList.from_df(df=df, path='data', folder='train_merged', using_librosa=True)
          .split_by_rand_pct(0.1)
          .label_from_df(label_delim=',')
          .add_test_folder('test')
          .transform(tfms=(train_tfms, valid_tfms))
          .databunch(bs=batch_size, tfms=dl_tfms)
         ).normalize()

f_score = partial(fbeta, thresh=0.2)
acc_02 = partial(accuracy_thresh, thresh=0.2)

learn = create_cnn(audios, models.vgg16_bn, pretrained=False, metrics=[f_score, acc_02])


learn.fit_one_cycle(100, 3e-2)