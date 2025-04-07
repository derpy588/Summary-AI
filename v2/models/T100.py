from model_lib import ModelBase, FileDataset, get_data_loader, save_model_as, load_model_from
from model_lib.model_modules import BaseTransformer

from transformers import AutoTokenizer
import torch
from torch.optim import Adam
import time
from tqdm import tqdm

class T100(ModelBase):
    def __init__(self):
        self.src_vocab_size=50265
        self.tgt_vocab_size=50265
        self.d_model=64
        self.num_heads=8
        self.num_layers=4
        self.d_ff=256
        self.max_seq_length=512
        self.dropout=0.1
        super().__init__(model=BaseTransformer(src_vocab_size=self.src_vocab_size, tgt_vocab_size=self.tgt_vocab_size, d_model=self.d_model, num_heads=self.num_heads, num_layers=self.num_layers, d_ff=self.d_ff, max_seq_length=self.max_seq_length, dropout=self.dropout))
        
    def preprocess(self, inp):
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        
        return tokenizer(inp, max_length=self.max_seq_length, truncation=True, padding="max_length")
    
    def total_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_file(self, file_path, inp_cat, out_cat, epochs, batch_size, shuffle, learning_rate, save_on_epoch=False, model_path=None, data_to_device=False):
        loader = get_data_loader(file_path, inp_cat, out_cat, self.preprocess, batch_size, shuffle)
        self.train(loader, epochs, learning_rate, save_on_epoch, model_path)
    
    def train(self, dataloader, epochs, learning_rate, save_on_epoch=False, model_path=None, data_to_device=False):
        
        print("Checking devices...")
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(f"Moving model to {device_name}...")
        self.model.to(device)
        
        # Print all info about crrent training stuff
        print(f"""
              :==============================================:
              Model Params:         {self.total_params()}
              Total Data Points:    {len(dataloader.dataset)}
              Total Batches:        {len(dataloader)}
              Epochs:               {epochs}
              Learning Rate:        {learning_rate}
              Device:               {device_name}
              :==============================================:
              """)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        start_time = time.time()
        
        print("Starting Training...")
        
        epoch_bar = tqdm(total=epochs, desc="Epoch Progress", position=0, leave=False)
        
        for epoch in range(epochs):
            running_loss = 0
            
            batch_bar = tqdm(total=len(dataloader), desc=f"Batch", position=1, leave=False)
            
            for batch_indx, (input_ids, attention_mask, labels) in enumerate(dataloader):
                
                if data_to_device == True:
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                step_bar = tqdm(total=3, desc="Batch Steps", position=2, leave=False)
                
                step_bar.set_description("Forward pass")
                outputs = self.model(input_ids, labels[:, :-1])
                step_bar.update(1)
                
                step_bar.set_description("Computing loss")
                loss = criterion(outputs.contiguous().view(-1, self.tgt_vocab_size), labels[:, 1:].contiguous().view(-1))
                step_bar.update(1)
                
                step_bar.set_description("Backward pass")
                loss.backward()
                
                optimizer.step()
                step_bar.update(1)
                
                step_bar.close()
                
                running_loss += loss.item()
                
                batch_bar.set_postfix(loss=loss.item())
                batch_bar.update(1)
                
            if save_on_epoch == True:
                self.save(model_path)
            avg_loss = running_loss / len(dataloader)
            epoch_bar.set_postfix(loss=avg_loss)
            epoch_bar.update(1)
            
        epoch_bar.close()
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
                
    
    def evaluate(self, inp):
        self.model.eval()
        
        src = torch.tensor(self.preprocess(inp).input_ids, dtype=torch.long)
        tgt = torch.tensor(self.preprocess("the").input_ids, dtype=torch.long)#torch.cat((torch.zeros(1), torch.ones(self.max_seq_length - 1)))
        
        with torch.no_grad():
            print(len(src))
            out_data = self.model(src, tgt[:-1])
            
        
    
    
    def test_file(self, file_path, inp_cat, out_cat, batch_size, shuffle):
        loader = get_data_loader(file_path, inp_cat, out_cat, self.preprocess, batch_size, shuffle)
        self.test(loader, batch_size)
    
    def test(self, dataloader, batch_size):
        pass
    
    def save(self, path):
        save_model_as(self.model, path)
    
    
    def load(self, path, device="cpu"):
        load_model_from(self.model, path, device)
        