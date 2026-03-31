"""
Loads a previously trained ReverseBIT model and does inference stuff with it
"""

import torch
from reverse_blur_interpolation_transformer_training import create_model

def main():

    model = create_model()

    # TODO run model on some blurry images

    # TODO save inferred data
    

if __name__ == "__main__":
    main()