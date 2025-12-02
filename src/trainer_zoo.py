import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TrackMamba import TrackMamba
from config import TrackMambaTrainerCfg, TrackMambaConfig
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset
from dataset import DNAseSeqDataset
from torch.utils.data import DataLoader
from typing import Dict, List
import pandas as pd
import numpy as np
import wandb
from zoo.cdrge import cdrge_minimal, calculate_loss, cdrge_parallel
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrackMambaTrainer:

    def __init__(self, trainer_cfg: TrackMambaTrainerCfg, model_cfg: TrackMambaConfig):
        self.cfg = trainer_cfg
        self.model_cfg = model_cfg
        self.tokenizer = model_cfg.tokenizer
        self.model = TrackMamba(model_cfg).to(device)
        self.regression_loss_fn_zoo = torch.nn.MSELoss()
        self.regression_loss_fn = self.cfg.regression_loss
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Define two dicts for loss tracking
        self.train_regr_loss = {}
        self.val_regr_loss = {}

        # JSON -> pd.DataFrame -> hf Dataset to avoid overflow
        df = Dataset.load_from_disk(self.cfg.data_path)

        # Split into train test
        train_df = df.filter(
            lambda chrom: chrom in self.cfg.train_chroms,
            input_columns = ["chrom"]
        )
        test_df = df.filter(
            lambda chrom: chrom in self.cfg.test_chroms,
            input_columns = ["chrom"]
        )

        self.train_dataset = train_df.map(self.preprocess_function, batched=True)
        self.test_dataset = test_df.map(self.preprocess_function, batched=True)

        # Init wandb
        self.step = 0
        wandb.init(
            project=self.cfg.wandb_project,
            name=f"{self.cfg.wandb_name}_{model_cfg.num_layers}L_{model_cfg.hidden_dim}D",
            config={
                "learning_rate": self.cfg.lr,
                "epochs": self.cfg.num_epochs,
                "batch_size": self.cfg.batch_size,
                "regression_loss_function": self.cfg.regression_loss.__class__.__name__,
                "classification_loss_function": self.cfg.classification_loss.__class__.__name__,
                "num_layers": model_cfg.num_layers,
                "embed_size": model_cfg.hidden_dim
            }
        )

    def preprocess_function(self, examples):
        return self.tokenizer(examples['sequence'], padding=True, truncation=False, add_special_tokens=False)
    
    def train_dataloader(self):
        return DataLoader(DNAseSeqDataset(self.train_dataset), batch_size=self.cfg.batch_size, shuffle=True, collate_fn=self.data_collator)
    
    def test_dataloader(self):
        return DataLoader(DNAseSeqDataset(self.test_dataset), batch_size=self.cfg.batch_size, shuffle=False, collate_fn=self.data_collator)
    
    @torch.inference_mode()
    def val_step(self, input_ids, regression_label, classification_label):

        # Put everything on device
        input_ids = input_ids.to(device)
        regression_label = regression_label.to(device)
        classification_label = classification_label.to(device)

        # Obtain regression and classification prediction
        regression_pred, class_logits = self.model(input_ids)

        # Calculate losses
        regression_loss = self.regression_loss_fn(regression_pred, regression_label)
        # classification_loss = self.classification_loss_fn(class_logits, classification_label)
        # total_loss = regression_loss + self.cfg.comb_factor * classification_loss

        # Log loss into wandb
        wandb.log({
            "Val Regression Loss": regression_loss,
        }, step=self.step)

        # Log into dict
        self.val_regr_loss[self.step] = regression_loss.item()

        return  {
            "input_ids": input_ids,
            "regression_loss": regression_loss,
        }
    
    def train(self):
        
        progress_bar = tqdm.tqdm(
            range(self.cfg.num_epochs * len(self.train_dataloader())),
            total=len(self.train_dataloader()) * self.cfg.num_epochs,
            colour="GREEN"
        )
        best_val_loss = 10000
        
        for epoch in range(self.cfg.num_epochs):

            for batch in self.train_dataloader():
            
                # Cycle on input data
                loss = cdrge_minimal(
                    model=self.model,
                    batch=batch,
                    loss_calculator=self.regression_loss_fn,
                    lr=self.cfg.lr,
                    epsilon=self.cfg.lr, # eps / eta is good between 1 and 10
                    num_perturbations=16
                )

                # Update dictionary with loss
                self.train_regr_loss[self.step] = loss.item()
                self.step += 1

                # Log on Wandb
                wandb.log({
                    "Train Regression Loss": loss,
                }, step=self.step)

                progress_bar.set_description(f"Epoch: {epoch + 1}. L: {loss:.4f}")
                progress_bar.update()

            # Validation step
            val_loss_total = 0.0
            for batch in self.test_dataloader():
                eval_dict = self.val_step(
                    batch['input_ids'],
                    batch['signal'], 
                    batch['labels']
                )
                val_loss_total += eval_dict['regression_loss']

            avg_val_loss = val_loss_total / len(self.test_dataloader())
            self.val_regr_loss[self.step] = avg_val_loss.item()

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if_mlp = "_mlp" if self.model.cfg.use_MLP else ""
                if_pos = "_pos_embs" if self.model.cfg.use_pos_embs else ""
                save_path = f"{os.path.splitext(self.cfg.save_path)[0]}_{self.model_cfg.num_layers}L_{self.model_cfg.hidden_dim}D{if_mlp}{if_pos}_ZOO.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved in {save_path} with avg_val_loss={avg_val_loss}")

        # Finish training
        wandb.finish()
    

def save_loss_plot(train_losses: Dict[int, float], val_losses: Dict[int, float], outfile: str, num_params: int):

    # Extract x axes
    x_train_ax = np.array(list(train_losses.keys()))
    x_val_ax = np.array(list(val_losses.keys()))

    # Convert losses to np.array
    train_loss = np.array(list(train_losses.values()))
    val_loss = np.array(list(val_losses.values()))

    # Define the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot values
    plt.plot(x_train_ax, train_loss, color="orange", label="Training Loss")
    plt.plot(x_val_ax, val_loss, color="purple", label="Validation Loss")

    # Set title and axes labels
    plt.title(f"TrackMamba ({num_params}M)")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Save figure
    plt.savefig(outfile, format="png", dpi=300)
    plt.show()

if __name__ == '__main__':

    # Include argparse for output file
    p = argparse.ArgumentParser()
    p.add_argument("--plot_path", type=str, required=False, default=None, help="Path to results file.")
    p.add_argument("--num_layers", type=int, required=False, default=None, help="Number of Mamba layers.")
    p.add_argument("--hidden_dim", type=int, required=False, default=None, help="Dimensionality of the residual stream.")
    p.add_argument("--num_epochs", type=int, required=False, default=None, help="Number of training epochs.")
    p.add_argument("--use_MLP", action='store_true', default=False)
    p.add_argument("--use_pos_embs", action='store_true', default=False)
    p.add_argument("--loss_path", type=str, required=False, default=None, help="Where to store loss values.")
    p.add_argument("--lr", type=float, required=False, default=None, help="Learning rate used.")
    p.add_argument("--save_path", type=str, required=False, default=None, help="Where to save model.")
    
    # Parsing
    args = p.parse_args()
    
    # Initialize config files
    trainer_cfg = TrackMambaTrainerCfg(
        num_epochs=args.num_epochs if args.num_epochs else TrackMambaTrainerCfg().num_epochs,
        lr=args.lr if args.lr else TrackMambaTrainerCfg().lr,
        train_chroms=["chr2"],
        test_chroms=["chr1"]
    )
    model_cfg = TrackMambaConfig(
        num_layers=args.num_layers if args.num_layers else TrackMambaConfig().num_layers,
        hidden_dim=args.hidden_dim if args.hidden_dim else TrackMambaConfig().hidden_dim,
        use_MLP=args.use_MLP,
        use_pos_embs=args.use_pos_embs
    )

    # Set seed for reproducibility
    torch.manual_seed(trainer_cfg.seed)
    torch.cuda.manual_seed(trainer_cfg.seed)
    torch.cuda.manual_seed_all(trainer_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize trainer and train
    trainer = TrackMambaTrainer(trainer_cfg=trainer_cfg, model_cfg=model_cfg)
    trainer.train()

    # Update results
    if args.plot_path:
        
        # Calculate model parameters
        model_size = sum(p.numel() for p in trainer.model.parameters()) / 1e6

        # Save loss plots
        save_loss_plot(trainer.train_regr_loss, trainer.val_regr_loss, args.plot_path, model_size)

    if args.loss_path:

        df_train = pd.DataFrame({
            "step": list(trainer.train_regr_loss.keys()),
            "loss": list(trainer.train_regr_loss.values()),
            "type": "train"
        })
        df_val = pd.DataFrame({
            "step": list(trainer.val_regr_loss.keys()),
            "loss": list(trainer.val_regr_loss.values()),
            "type": "val"
        })

        df = pd.concat([df_train, df_val]).sort_values("step")
        df.to_csv(args.loss_path, sep="\t", index=False)