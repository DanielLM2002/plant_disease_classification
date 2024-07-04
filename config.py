import os
import torch.nn as nn
from dotenv import load_dotenv
import torch.nn.functional as F
import torch
def vae_loss(reconstructed_x, x, mu, logvar):
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

load_dotenv()

# MODEL PARAMETERS
USE_UNET = False
USE_VAE = True
USE_DENOISING_AUTOENCODER = False
AE_TRANSFORMS = 'mse'
CNN_BATCH_SIZE = 32
AE_BATCH_SIZE = 32
ENCODER_CNN_BATCH_SIZE = 32
TRAIN_ENCODER_WEIGHTS = True
API_KEY = os.getenv('API_KEY')

LOSS_FUNCTION = vae_loss
LEARNING_RATE = 0.0001

CHANNELS = 3
LATENT_CHANNELS = 1
HEIGHT = 16
WIDHT = 8

EPOCHS = 30
PATIENCE = 5


# DATASET PARAMETERS
UNLABELED_SET_SIZE = 0.5
LABELED_TRAIN_SET_ABSOLUTE_SIZE = 0.35
LABELED_TEST_SET_ABSOLUTE_SIZE = 0.15

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
