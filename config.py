import os
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

# MODEL PARAMETERS
USE_UNET = False
USE_DENOISING_AUTOENCODER = False
AE_TRANSFORMS = 'mse'
CNN_BATCH_SIZE = 32
AE_BATCH_SIZE = 32
ENCODER_CNN_BATCH_SIZE = 32
TRAIN_ENCODER_WEIGHTS = False
API_KEY = os.getenv('API_KEY')

LOSS_FUNCTION = nn.MSELoss()
LEARNING_RATE = 0.001

CHANNELS = 3
LATENT_CHANNELS = 2
HEIGHT = 112
WIDHT = 112

EPOCHS = 30
PATIENCE = 5


# DATASET PARAMETERS
UNLABELED_SET_SIZE = 0.8
LABELED_TRAIN_SET_ABSOLUTE_SIZE = 0.1
LABELED_TEST_SET_ABSOLUTE_SIZE = 0.1

BASE_DIR_RAW = os.path.join('Plant_leave_diseases_dataset', 'original')
BASE_DIR_NOISY = os.path.join('Plant_leave_diseases_dataset', 'with_noise')
BASE_DIR_MODELS = 'best_models'

os.makedirs(BASE_DIR_MODELS, exist_ok=True)

AUTOENCODER_SAVE_PATH = \
    os.path.join('best_models', f'h1_{int(UNLABELED_SET_SIZE*100)}-{int(LABELED_TRAIN_SET_ABSOLUTE_SIZE*100)}-{int(LABELED_TEST_SET_ABSOLUTE_SIZE*100)}_Autoencoder.pth')
DENOISING_AUTOENCODER_SAVE_PATH = \
    os.path.join('best_models', f'h2_{int(UNLABELED_SET_SIZE*100)}-{int(LABELED_TRAIN_SET_ABSOLUTE_SIZE*100)}-{int(LABELED_TEST_SET_ABSOLUTE_SIZE*100)}_DenoisingAutoencoder.pth')
CNN_SAVE_PATH = \
    os.path.join('best_models', f'h1_{int(UNLABELED_SET_SIZE*100)}-{int(LABELED_TRAIN_SET_ABSOLUTE_SIZE*100)}-{int(LABELED_TEST_SET_ABSOLUTE_SIZE*100)}_classifierA.pth')
CNN_ENCODER_SAVE_PATH = \
    os.path.join('best_models', f'h1_{int(UNLABELED_SET_SIZE*100)}-{int(LABELED_TRAIN_SET_ABSOLUTE_SIZE*100)}-{int(LABELED_TEST_SET_ABSOLUTE_SIZE*100)}_classifier{"B" if not TRAIN_ENCODER_WEIGHTS else "C"}.pth')
CNN_DENOISING_SAVE_PATH = \
    os.path.join('best_models', f'h2_{int(UNLABELED_SET_SIZE*100)}-{int(LABELED_TRAIN_SET_ABSOLUTE_SIZE*100)}-{int(LABELED_TEST_SET_ABSOLUTE_SIZE*100)}_DenoisingClassifier.pth')
