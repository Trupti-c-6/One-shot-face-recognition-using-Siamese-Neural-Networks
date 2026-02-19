import torch
from models.siamese_model import SiameseNetwork


def load_model(model_path):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == "__main__":
    print("Inference module ready.")
