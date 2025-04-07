from abc import ABC, abstractmethod

class ModelBase(ABC):
    
    @abstractmethod
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def preprocess(self, inp):
        pass
    
    @abstractmethod
    def train_file(self, file_path, inp_cat, out_cat, epochs, batch_size, learning_rate, save_on_epoch=False, model_path=None, data_to_device=False):
        pass
    
    @abstractmethod
    def train(self, dataloader, epochs, learning_rate, save_on_epoch=False, model_path=None, data_to_device=False):
        pass
    
    @abstractmethod
    def test_file(file_path, inp_cat, out_cat, batch_size):
        pass
    
    @abstractmethod
    def test(self, dataloader, batch_size):
        pass
    
    @abstractmethod
    def evaluate(self, inp):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    @abstractmethod
    def total_params(self):
        pass