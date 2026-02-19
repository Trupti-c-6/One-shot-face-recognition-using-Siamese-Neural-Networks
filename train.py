import torch
import torch.nn as nn
import torch.optim as optim
from models.siamese_model import SiameseNetwork


def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(backbone="resnet18").to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Model initialized successfully.")
    print(model)


if __name__ == "__main__":
    main()
