import os
import ssl
import torch
import torch.nn as nn
from kornia.feature import LoFTR

ssl._create_default_https_context = ssl._create_unverified_context

class HuggingFaceLoFTR(nn.Module):
    """PyTorch wrapper around Kornia's LoFTR implementation."""
    def __init__(self, pretrained_config: str = "outdoor"):
        super().__init__()
        print(f"Initializing the LoFTR model (configuration: '{pretrained_config}')...")
        self.model = LoFTR(pretrained=pretrained_config)

    def forward(self, image0, image1):

        if image0.shape[1] == 3:
            image0 = 0.299 * image0[:, 0:1] + 0.587 * image0[:, 1:2] + 0.114 * image0[:, 2:3]

        if image1.shape[1] == 3:
            image1 = 0.299 * image1[:, 0:1] + 0.587 * image1[:, 1:2] + 0.114 * image1[:, 2:3]

        batch = {
            "image0": image0,
            "image1": image1,
        }

        out = self.model(batch)

        if out is None:
            out = batch

        return out

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        weights_path = os.path.join(save_directory, "loftr_weights.pth")
        torch.save(self.state_dict(), weights_path)
        print(f"The model has been successfully saved in: {weights_path}")

    @classmethod
    def from_pretrained(cls, save_directory: str, device: torch.device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        instance = cls(pretrained_config="outdoor")
        weights_path = os.path.join(save_directory, "loftr_weights.pth")

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            instance.load_state_dict(state_dict)
            print(f"The model weights have been successfully loaded from: {weights_path}")
        else:
            print(f"Warning: The file {weights_path} was not found. Default weights are being used.")

        return instance.to(device)