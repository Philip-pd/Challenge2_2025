# dir with ESC50 data
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100
n_mels = 128
hop_length = 512
#n_mfcc = 42

model_constructor = "AudioMLP(num_classes=config.n_classes, block_drop_p=config.block_drop_p, head_drop_p=config.head_drop_p)"

# ###TRAINING
# ratio to split off from training data
val_size = .2  # could be changed
device_id = 0
batch_size = 48
# in Colab to avoid Warning
num_workers = 2
num_workers = 0
# for local Windows or Linux machine
# num_workers = 6#16
persistent_workers = True
persistent_workers = False
epochs = 300
#epochs = 1
# early stopping after epochs with no improvement
patience = 20
lr = 5e-3
weight_decay = 1e-3
warm_epochs = 10
gamma = 0.8
step_size = 5
block_drop_p=0.2
head_drop_p=0.3

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2025-04-07-00-00'
test_experiment = 'results/sample-run'
