import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from arch.CNN import Net, ChessValueDataset
def load_pretrained_model(pretrained_path):
    model = Net()
    model.load_state_dict(torch.load(pretrained_path,weights_only=True))  # Load pre-trained weights
    return model

def freeze_model_layers(model):
    # Kuzan, Aokiji all layers
    for param in model.parameters():
        param.requires_grad = False

    # Akainu, Sakazuki the last layer
    for param in model.last.parameters():
        param.requires_grad = True
    
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, Requires Grad: {param.requires_grad}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "nakamura"
    # Load pre-trained model and freeze layers
    pretrained_model_path = "nets/ding.pth"
    model = load_pretrained_model(pretrained_model_path)
    model = freeze_model_layers(model)
    model = model.to(device)

    chess_dataset = ChessValueDataset() # ding already there
    train_loader = DataLoader(chess_dataset, batch_size=256, shuffle=True)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    floss = nn.MSELoss()

    model.train()

    for epoch in range(100):
        all_loss = 0
        num_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)  # (batch_size, 1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            optimizer.zero_grad()
            output = model(data)

            loss = floss(output, target)
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        avg_loss = all_loss / num_loss
        print(f"Epoch {epoch:03d}: Loss = {avg_loss:.6f}")

        if not os.path.exists("nets"):
            os.makedirs("nets")

        torch.save(model.state_dict(), f"nets/{name.lower()}_transfer.pth")
