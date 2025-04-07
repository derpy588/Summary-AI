from safetensors.torch import save_model, load_model
import torch
    
def save_model_as(model, file_path):
    
    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        torch.save(model.state_dict(), file_path)
    elif file_path.endswith(".safetensors"):
        save_model(model, file_path)
    else:
        raise ValueError("Unsupported file format. Supported file formats are: .pt, .pth, .safetensors")
    
def load_model_from(model, file_path, device="cpu"):
    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        model.load_state_dict(torch.load(file_path, weights_only=True))
    elif file_path.endswith(".safetensors"):
        load_model(model, file_path, device=device)
    else:
        raise ValueError("Unsupported file format. Supported file formats are: .pt, .pth, .safetensors")
