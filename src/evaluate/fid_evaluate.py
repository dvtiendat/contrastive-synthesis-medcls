import os
import argparse
from pytorch_fid import fid_score
from src.models import (
    acgan,
    dcgan
)
LABEL = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
def compute_fid(real_images_path: str, generated_images_path: str, device='cuda'):
    """
    Compute the Fréchet Inception Distance (FID) between two sets of images.
    
    :param real_images_path: Directory containing real images
    :param generated_images_path: Directory containing generated images
    :param device: Device to compute FID on ('cuda' or 'cpu')
    :return: FID score
    """
    # Ensure paths are valid
    if not os.path.exists(real_images_path) or not os.path.exists(generated_images_path):
        raise ValueError("The specified paths do not exist.")
    
    # Calculate FID using pytorch-fid
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path], 
        batch_size=10,  # Adjust batch size if needed
        device=device, 
        dims=768
    )
    
    return fid_value
def import_model(model_name):
    """
    Dynamically import model based on model name.
    """
    if model_name == "ACGAN":
        G, D = acgan.get_model({})
    elif model_name == "DCGAN":
        # If you have a DCGAN model, you can import it in a similar way
        G, D = dcgan.get_model({})
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return G, D
def run():
    pass
def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument(
        '--model_name', 
        type=str, 
        choices=["ACGAN", "DCGAN"], 
        default="ACGAN", 
        help="The model type ('ACGAN' or 'DCGAN'). Default to 'ACGAN'."
    )
    parser.add_argument(
        '--dims', 
        type=int, 
        choices=[92, 768, 2048], 
        default=2048, 
        help="Dimensionality for FID calculation (92, 768, or 2048). Default is 2048."
    )

    return parser.parse_args()
if __name__ == "__main__":
    # args = parse_args()
    model_name = "ACGAN"
    G, D = import_model(model_name)
    for label in LABEL:
        G.sample_images(label, 'images/gen_images', 100, 5)
        real_img_paths = f'images/real_images/{label}'
        gen_img_paths = f'images/gen_images/{model_name}/{label}'
        fid_score = compute_fid(real_img_paths, gen_img_paths, device = "cpu")