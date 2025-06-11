import torch

GAMMA = 1
EPSILON = 1e-4
LAMBD = 25
MU = 25
NU = 1
PROJ_D = 512
ENCODE_D = 128
B_SIZE = 256
LR = 3e-4
N_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'