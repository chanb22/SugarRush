import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from model.model import O3Transformer
from model.norms import EquivariantLayerNorm
from e3nn import o3
import pickle
from datetime import datetime

test_criterion = torch.nn.L1Loss()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data_carb: DataLoader,
        optimizer: torch.optim.Optimizer,
        grad_clip_norm=1.0,
        mixed_precision=True
    ) -> None:
        super().__init__()
        self.gpu_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.gpu_id)
        self.train_data_carb = train_data_carb
        self.optimizer = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.mixed_precision = mixed_precision
        self.scaler = torch.amp.GradScaler(enabled=mixed_precision)

        self.derive_mean_and_std()
    def _run_batch(self, data: Data, criterion):
        self.optimizer.zero_grad(set_to_none=True)

        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)    
        nmr_true =  data.x[:,-1]
        nmr_masked = (nmr_true[mask]- self.mean)/self.std
        with torch.amp.autocast(device_type="cuda", enabled=self.mixed_precision):
            out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                        edge_index = data.edge_index, edge_attr = 
                        data.edge_attr.long(), batch = data.batch)
            loss = criterion(out[mask].flatten(), nmr_masked.clone())
        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def _val_batch(self, data: Data, criterion):
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)    
        nmr_true =  data.x[:,-1]
        nmr_masked = nmr_true[mask]
        with torch.no_grad():
            out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                        edge_index = data.edge_index, edge_attr = 
                        data.edge_attr.long(), batch = data.batch)

            out_masked = out[mask]* self.std +  self.mean
            loss = test_criterion(out_masked.flatten(), nmr_masked)
        
        return loss.item()

    def _run_epoch(self, epoch: int,criterion):
        self.model.train()
        torch.cuda.empty_cache()
        losses = []
        data_length = len(self.train_data_carb)
        for i, data in enumerate(self.train_data_carb):                   
#            torch.cuda.empty_cache()
            data = data.to(self.gpu_id, non_blocking=True)
            losses.append(self._run_batch(data, criterion))
#            free, total = torch.cuda.mem_get_info(self.gpu_id)
#            mem_used_MB = (total - free) / 1024 ** 3
#            print(mem_used_MB)
#            if (i%50 ==0):
#                self._save_checkpoint(epoch)
            print(f"Epoch {epoch+1} | Batch {i+1}/{data_length}")
        return sum(losses) / len(losses)

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        PATH = "_checkpoint_epoch_" + str(epoch)+".pkl"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def derive_mean_and_std(self):
        nmrs = []
        for data in self.train_data_carb:
            nmr_true =  data.x[:,-1]
            c_mask = data.x[:,0] == 2.0
            nmr_mask = data.x[:,-1] > -0.5
            mask = nmr_mask.logical_and(c_mask)    
            nmrs.append(nmr_true[mask])
        nmrs = torch.cat(nmrs)
        self.mean = nmrs.mean()
        self.std = nmrs.std()

    def train(self, max_epochs: int, criterion, lr_reduce):
        
        for epoch in range(max_epochs):
            time_start = datetime.now()
            loss = self._run_epoch(epoch, criterion)
            time_end = datetime.now()
            print(f"Epoch {epoch+1} | Train loss {loss} | Elapsed {(time_end - time_start)}")
            if epoch == (max_epochs - 1):
                self._save_checkpoint(epoch+1)
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*lr_reduce
                              

                
    def test(self, test_data: Data, message: str):
        torch.cuda.empty_cache()
        criterion = torch.nn.L1Loss()
    
        nmr_trues = []
        nmr_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, loader_dict in enumerate(test_data):
                    N = loader_dict["n_mols"]
                    N_nmr = loader_dict["n_nmr"]
                    for data in loader_dict["loader"]:
                            torch.cuda.empty_cache()
                            data = data.to(self.gpu_id, non_blocking=True) 
                            c_mask = data.x[:,0] == 2.0
                            nmr_mask = data.x[:,-1] > -0.5
                            mask = nmr_mask.logical_and(c_mask) 
                            nmr_true =  data.x[:,-1]
                            nmr_masked = nmr_true[mask]
                            out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                                        edge_index = data.edge_index, edge_attr = data.edge_attr.long(), batch = data.batch)
                            out_masked = out[mask]*self.std + self.mean
                            
                            out = (out_masked.reshape(N,N_nmr).T).mean(dim = 1)
                            trues = nmr_masked[:N_nmr].detach().flatten()
                            nmr_trues.append(trues)
                            nmr_preds.append(out.detach().flatten())
            if nmr_trues:
                    nmr_trues_ = torch.cat(nmr_trues)
                    nmr_preds_ = torch.cat(nmr_preds)
    
                    l_lest = criterion(nmr_trues_.flatten(), nmr_preds_.flatten()).item()
                    l_test_rmse = torch.nn.MSELoss()(nmr_trues_.flatten(), nmr_preds_.flatten()).item() ** 0.5

                    print("Test MAE error is " + str(l_lest) + " for " + message + ".")
                    print("Test RMSE error is " + str(l_test_rmse) + " for " + message + ".")
        self.model.train()
                


def load_train_data(train_path: str):      

    with open(train_path, 'rb') as handle:
        train_data_carb = pickle.load(handle)

    train_data_carb_ = InMemoryDataset()       
    train_data_carb_.data, train_data_carb_.slices = train_data_carb_.collate(train_data_carb)    
    
    return train_data_carb_     
             

def load_train_test_objs(train_path:str, test_path_mo:str, test_path_di:str, test_path_tri:str, test_path_tetra:str, test_path_oligo:str, test_path_poly:str):

    train_data_carb  = load_train_data(train_path)
    with open(test_path_mo, 'rb') as handle:
        test_data_mo = pickle.load(handle)    
    with open(test_path_di, 'rb') as handle:
        test_data_di = pickle.load(handle)       
    with open(test_path_tri, 'rb') as handle:
        test_data_tri = pickle.load(handle) 
    if test_path_tetra is not None:
        with open(test_path_tetra, 'rb') as handle:
            test_data_tetra = pickle.load(handle) 
    else:
        test_data_tetra = None
    if test_path_oligo is not None:
        with open(test_path_oligo, 'rb') as handle:
            test_data_oligo = pickle.load(handle) 
    else:
        test_data_oligo = None
    if test_path_poly is not None:
        with open(test_path_poly, 'rb') as handle:
            test_data_poly = pickle.load(handle) 
    else:
        test_data_poly = None

    model = O3Transformer(norm = EquivariantLayerNorm, n_input = 128,n_node_attr = 128,n_output =128, 
                                      irreps_hidden = o3.Irreps("64x0e + 32x1o + 8x2e"),n_layers = 7)
                                      
    return train_data_carb, test_data_mo, test_data_di, test_data_tri, test_data_tetra, test_data_oligo, test_data_poly, model


def prepare_dataloader(dataset: Dataset, batch_size: int):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

def prepare_test_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False)

def main(epochs: int, batch_size: int, checkpoint_path: str, train_path: str, test_path_mo: str, test_path_di: str, test_path_tri: str, test_path_tetra: str, test_path_oligo: str, test_path_poly: str):
    torch.manual_seed(42)

    train_carb, test_data_mo, test_data_di, test_data_tri, test_data_tetra, test_data_oligo, test_data_poly, model = load_train_test_objs(train_path, test_path_mo, test_path_di, test_path_tri, test_path_tetra, test_path_oligo, test_path_poly)

    train_data_carb = prepare_dataloader(train_carb, batch_size)   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    trainer = Trainer(model,train_data_carb, optimizer, grad_clip_norm=0.5, mixed_precision=True)
    
    if checkpoint_path is None:
        print(f"Training...")
        criterion = torch.nn.L1Loss()
        trainer.train(epochs, criterion, 0.1)
    else:
        print(f"Skipping training, going straight to testing. Loading checkpoint...")
        trainer.model.load_state_dict(torch.load(checkpoint_path))
    
    print(f"Testing...")
    trainer.test(test_data_mo,"mono")
    trainer.test(test_data_di,"di")
    trainer.test(test_data_tri, "tri")
    if test_data_tetra is not None:
        trainer.test(test_data_tetra, "tetra")
    if test_data_oligo is not None:
        trainer.test(test_data_oligo, "oligo")
    if test_data_poly is not None:
        trainer.test(test_data_poly, "poly")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model Script')
    parser.add_argument('--train_path', type=str, help='Path to train dataset')
    parser.add_argument('--test_path_mo', type=str, help='Path to mono test dataset')
    parser.add_argument('--test_path_di', type=str, help='Path to di test dataset')
    parser.add_argument('--test_path_tri', type=str, help='Path to tri test dataset')
    parser.add_argument('--test_path_tetra', type=str, help='Path to tetra test dataset')
    parser.add_argument('--test_path_oligo', type=str, help='Path to oligo test dataset')
    parser.add_argument('--test_path_poly', type=str, help='Path to poly test dataset')

    parser.add_argument('--batch_size', default=32, type=int, help='Training batch size (default 32)')
    parser.add_argument('--epochs', default=3, type=int, help='Number of training epochs (default 3)')
    parser.add_argument('--checkpoint_path', type=str, help='Only do testing from checkpoint located at path')

    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.checkpoint_path, args.train_path, args.test_path_mo, args.test_path_di, args.test_path_tri, args.test_path_tetra, args.test_path_oligo, args.test_path_poly)
