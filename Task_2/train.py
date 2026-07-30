import os
import torch
from torch.utils.data import DataLoader
from src.dataset import SatelliteMatchingDataset
from src.model import HuggingFaceLoFTR
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    """Runs one training epoch over the dataset"""
    model.eval() # Keep backbone in eval mode while updating matching weights
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        img0 = batch['image0'].to(device)
        img1 = batch['image1'].to(device)
        optimizer.zero_grad()
        output = model(img0, img1)
        # Compute proxy loss from predicted confidence matrix
        if 'conf_matrix' in output and output['conf_matrix'] is not None:
            conf_matrix = output['conf_matrix']
            loss = -conf_matrix.mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)

def main():
    # Setup computational device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Learning device: {device}")
    # Filepath configuration
    dir_a = "data/processed/season_a"
    dir_b = "data/processed/season_b"
    save_dir = "models/loftr_finetuned"
    # Initialize PyTorch DataLoader
    dataset = SatelliteMatchingDataset(dir_a, dir_b)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # Model and optimizer setup
    model = HuggingFaceLoFTR(pretrained_config="outdoor").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Training execution loop
    epochs = 5
    print("The Start of the Learning Process...")
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Training is complete. The model has been saved in {save_dir}")


if __name__ == "__main__":
    main()