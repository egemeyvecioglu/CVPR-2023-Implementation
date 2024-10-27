import argparse
import json
import torch
from models import Model
import loaders
from torch.utils.data import DataLoader

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
argparser.add_argument("--source_domain", type=str, default=None, help="Source domain")
argparser.add_argument("--domain", type=str, default=None, help="Domain. Must be one of digits or pacs")
argparser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu, cuda, mps). Default: cuda")
args = argparser.parse_args()


def get_loaders(dataset_name):
    if dataset_name == "SVHN":
        return loaders.SVHNLoader(root="data/svhn", batch_size=32, num_workers=4, resize_dim=32).get_loaders()
    elif dataset_name == "MNIST":
        return loaders.MNISTLoader(root="data/mnist", batch_size=32, num_workers=4, resize_dim=32).get_loaders()
    elif dataset_name == "USPS":
        return loaders.USPSLoader(root="data/usps", batch_size=32, num_workers=4, resize_dim=32).get_loaders()
    elif dataset_name == "SYN":
        return loaders.SYNLoader(root="data/syn", batch_size=32, num_workers=4).get_loaders()
    elif dataset_name == "MNIST-M":
        return loaders.MNIST_MLoader().get_loaders()
    elif dataset_name == "photo":
        dataset = loaders.PACSDataset(root_dir="data/archive", domain="photo", train=False) 
        return None, DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    elif dataset_name == "art_painting":
        dataset = loaders.PACSDataset(root_dir="data/archive", domain="art_painting", train=False) 
        return None, DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    elif dataset_name == "cartoon":
        dataset = loaders.PACSDataset(root_dir="data/archive", domain="cartoon", train=False) 
        return None, DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    elif dataset_name == "sketch":
        dataset = loaders.PACSDataset(root_dir="data/archive", domain="sketch", train=False) 
        return None, DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    else:
        raise ValueError("Dataset not found")
    

def evaluate_model(model, dataset_name):
    _, test_loader = get_loaders(dataset_name)

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy in the dataset {dataset_name}: {accuracy:.2f}%")

    return accuracy

if __name__ == "__main__":
    model_path = args.model_path

    # load a pretrained model
    state_dict = torch.load(model_path)

    domain = args.domain
    config = json.load(open("configs.json"))[domain]
    device = config["device"] = argparser.device
    model = Model(domain, config).model

    # load the model state
    model.load_state_dict(state_dict)

    domains = ["MNIST", "SVHN", "USPS", "SYN", "MNIST-M"] if domain == "digits" else ["photo", "art_painting", "cartoon", "sketch"] if domain == "pacs" else []
        
    source_domain = args.source_domain
    avg_accuracy = 0
    # evaluate the model on the datasets
    for domain in domains:
        # if domain != source_domain:
        acc = evaluate_model(model, domain)
        avg_accuracy += acc
    avg_accuracy /= len(domains) - 1
    print(f"Average accuracy: {avg_accuracy:.2f}%")

