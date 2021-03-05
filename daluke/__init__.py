import torch

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
