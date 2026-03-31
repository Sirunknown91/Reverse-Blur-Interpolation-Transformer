"""
Creates and trains a new model based on the image dataset.

(Check the original paper's github to see how they created their model)
"""

import torch
import torch.nn as nn

class ReverseBIT(nn.Module):
    # Reverse Blur Interpolation Transformer
    # TODO create model a definied similarly to paper

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


def create_model():
    """
    Creates a fresh transformer model
    """
    return ReverseBIT()


def main():
    # TODO Load in dataset used in the paper

    model = create_model()

    # TODO train model in training loop

    # TODO save model
    

if __name__ == "__main__":
    main()