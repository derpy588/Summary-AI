import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class FileDataset(Dataset):
    def __init__(self, file_path, input_category, output_category, preprocess_fn):
        self.file_path = file_path
        self.input_category = input_category
        self.output_category = output_category
        self.preprocess_fn = preprocess_fn
        
        # Load the file based on its extension
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.data = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
        if input_category not in self.data.columns or output_category not in self.data.columns:
            raise ValueError("Input or output category not found in file.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data.loc[idx, self.input_category]
        output_text = self.data.loc[idx, self.output_category]
        
        inp = self.preprocess_fn(input_text)
        out = self.preprocess_fn(output_text)
        
        return torch.tensor(inp.input_ids, dtype=torch.long), torch.tensor(inp.attention_mask, dtype=torch.long), torch.tensor(out.input_ids, dtype=torch.long)


def get_data_loader(file_path, input_category, output_category, preprocess_fn, batch_size=32, shuffle=True):
    dataset = FileDataset(file_path, input_category, output_category, preprocess_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader