import torch
from models import Model
import argparse
import json


# Create argument parser
parser = argparse.ArgumentParser(description="Training configuration")

# Add arguments
parser.add_argument("--arch", type=str, default=None, help="Model architecture")
parser.add_argument("--Tmax", type=int, default=None, help="Maximum number of iterations")
parser.add_argument("--sigma_d", type=float, default=None, help="Deformable offsets variance")
parser.add_argument("--sigma_gamma", type=float, default=None, help="Affine transformation (gamma) variance")
parser.add_argument("--sigma_beta", type=float, default=None, help="Affine transformation (beta) variance")
parser.add_argument("--Lmax", type=int, default=None, help="Maximum repetition numbers")
parser.add_argument("--alpha", type=float, default=None, help="Learning rate")
parser.add_argument("--momentum", type=float, default=None, help="Momentum")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
parser.add_argument("--device", type=str, default="cuda",help="Device to use (cpu, cuda, mps). Default: cuda")
parser.add_argument("--domain", type=str, default=None, help="Dataset domain")
parser.add_argument("--source", type=str, default=None, help="Source domain")

# Parse the arguments
args = parser.parse_args()


domain = args.domain
device = args.device

config = json.load(open("configs.json"))[domain]

config["device"] = (
    "cuda"
    if (device == "cuda" and torch.cuda.is_available())
    else (
        "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    )
)

if args.domain is None:
    raise ValueError("Please specify the domain with --domain argument. Available domains: digits, pacs ,office")
if args.arch is None:
    print("Model architecture not specified. Fetching from configs.json for the domain")
else:
    config["arch"] = args.arch
if args.Tmax is None:
    print("Maximum number of iterations not specified. Fetching from configs.json for the domain")
else:
    config["Tmax"] = args.Tmax
if args.sigma_d is None:
    print("Deformable offsets variance not specified. Fetching from configs.json for the domain")
else:
    config["sigma_d"] = args.sigma_d
if args.sigma_gamma is None:
    print("Affine transformation (gamma) variance not specified. Fetching from configs.json for the domain")
else:
    config["sigma_gamma"] = args.sigma_gamma
if args.sigma_beta is None:
    print("Affine transformation (beta) variance not specified. Fetching from configs.json for the domain")
else:
    config["sigma_beta"] = args.sigma_beta
if args.Lmax is None:
    print("Maximum repetition numbers not specified. Fetching from configs.json for the domain")
else:
    config["Lmax"] = args.Lmax
if args.alpha is None:
    print("Learning rate not specified. Fetching from configs.json for the domain")
else:
    config["alpha"] = args.alpha
if args.momentum is None:
    print("Momentum not specified. Fetching from configs.json for the domain")
else:
    config["momentum"] = args.momentum
if args.batch_size is None:
    print("Batch size not specified. Fetching from configs.json for the domain")
else:
    config["batch_size"] = args.batch_size
if args.source is None:
    print("Source domain not specified. Fetching from configs.json for the domain")
else:
    config["source"] = args.source

print("Using device: ", config["device"])

# Initialize the network
model = Model(domain, config)

model.train_with_progressive_augmentation()
