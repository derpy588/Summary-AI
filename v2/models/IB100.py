from model_lib import ModelBase, FileDataset, get_data_loader, save_model_as, load_model_from
from model_lib.model_modules import InbuiltTransformer


from transformers import AutoTokenizer
import torch
from torch.optim import Adam
import time
from tqdm import tqdm

class IB100(Modelbase):
    def __init__(self):
        self.ntokens=50265
        self.d_model=64
        self.num_heads=8
        self.num_layers=4
        self.d_ff=256
        self.max_seq_length=512
        self.dropout=0.1
        super().__init__(
            model=InbuiltTransformer(
                ntokens=self.ntokens, 
                ninp=self.d_model, 
                nhead=self.num_heads, 
                nhid=self.num_layers, 
                d_ff=self.d_ff, 
                max_seq_length=self.max_seq_length, 
                dropout=self.dropout))