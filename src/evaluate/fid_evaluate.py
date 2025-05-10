import os
import src
import argparse
from pytorch_fid import fid_score
from src.models import (
    acgan,
    dcgan
)
from src.utils.helper import get_path
LABEL = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
def compute_fid(real_images_path: str, generated_images_path: str, dims: int, device='cuda'):
    if not os.path.exists(real_images_path) or not os.path.exists(generated_images_path):
        raise ValueError("The specified paths do not exist.")
    
    # Calculate FID using pytorch-fid
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path], 
        batch_size=10,  
        device=device, 
        dims=dims
    )
    
    return fid_value
def import_model(model_name: str):
    """
    Dynamically import model based on model name.
    """
    if model_name == "ACGAN":
        G, D = acgan.get_model({})
    elif model_name == "DCGAN":
        G, D = dcgan.get_model({})
    else:
        raise ValueError(f"Model {model_name} is not supported.") 
    return G, D

def run_fid_evaluate(model_name:str, dims: int, path: str = get_path()):
    G, D = import_model(model_name)
    for label in LABEL:
        real_img_paths = os.path.join(path, f'images/real_images/{label}')
        gen_img_paths = os.path.join(path, f'images/gen_images/{model_name}/{label}')
        if not os.path.exists(gen_img_paths) or len(os.listdir(gen_img_paths)) != 1000:
            G.sample_images(label, os.path.join(path, 'images/gen_images'), 1000, 25)
        
        fid_score_value = compute_fid(real_img_paths, gen_img_paths, dims, device = "cpu")
        print(f"FID score of {model_name} on label {label} is: {fid_score_value:.2f}")
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
        default=768, 
        help="Dimensionality for FID calculation (92, 768, or 2048). Default is 2048."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # print(get_path())
    abs_path = get_path()
    args = parse_args()
    run_fid_evaluate(args.model_name, args.dims, abs_path)
    # G, D = import_model(args.model_name)
    # for label in LABEL:
    #     real_img_paths = os.path.join(abs_path, f'images/real_images/{label}')
    #     gen_img_paths = os.path.join(abs_path, f'images/gen_images/{args.model_name}/{label}')
    #     if not os.path.exists(gen_img_paths) or len(os.listdir(gen_img_paths)) != 1000:
    #         G.sample_images(label, os.path.join(abs_path, 'images/gen_images'), 1000, 25)
        
    #     fid_score_value = compute_fid(real_img_paths, gen_img_paths, args.dims, device = "cpu")
    #     print(f"FID score of {args.model_name} on label {label} is: {fid_score_value:.2f}")