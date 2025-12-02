import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TrackMamba import TrackMamba, TrackMambaMetadata
from config import TrackMambaTrainerCfg, TrackMambaConfig
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset
from dataset import DNAseSeqDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import numpy as np
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrackMambaTrainer:

    def __init__(self, trainer_cfg: TrackMambaTrainerCfg, model_cfg: TrackMambaConfig):
        self.cfg = trainer_cfg
        self.model_cfg = model_cfg
        self.tokenizer = model_cfg.tokenizer
        self.model = TrackMamba(model_cfg).to(device)
        self.regression_loss_fn = trainer_cfg.regression_loss
        self.classification_loss_fn = trainer_cfg.classification_loss
        self.optimizer = trainer_cfg.optimizer(self.model.parameters(), lr=trainer_cfg.lr)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=trainer_cfg.num_epochs)

        # Define two dicts for loss tracking
        self.train_regr_loss = {}
        self.val_regr_loss = {}

        # JSON -> pd.DataFrame -> hf Dataset to avoid overflow
        # df = pd.read_json(os.path.join(trainer_cfg.data_path, trainer_cfg.data_file))

        df = Dataset.load_from_disk(trainer_cfg.data_path)

        # Split dataset based on train and test chroms
        # train_df = df[df["chrom"].isin(self.cfg.train_chroms)]
        # test_df = df[df["chrom"].isin(self.cfg.test_chroms)]

        train_df = df.filter(
            lambda chrom: chrom in trainer_cfg.train_chroms,
            input_columns = ["chrom"]
        )
        test_df = df.filter(
            lambda chrom: chrom in trainer_cfg.test_chroms,
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
    
    def train_step(self, input_ids, regression_label, class_label, idxs, attention_mask):

        # Put everything on device
        input_ids = input_ids.to(device)
        regression_label = regression_label.to(device)
        class_label = class_label.to(device)
        idxs = idxs.to(device)
        attention_mask = attention_mask.to(device)

        # Obtain regression and classification prediction
        regression_pred, class_logits = self.model(input_ids, attention_mask=None) # (B, len(sequence), num_heads)

        # Take predicitons from corresponding heads
        current_batch_size = len(regression_label)
        batches = torch.arange(current_batch_size, device=device)
        regression_pred = regression_pred[batches, :, idxs]
        class_logits = class_logits[batches, :, idxs]

        # Calculate losses
        regression_loss = self.regression_loss_fn(regression_pred, regression_label)
        classification_loss = self.classification_loss_fn(class_logits, class_label.float())
        total_loss = regression_loss + self.cfg.comb_factor * classification_loss 

        # Backprop
        self.step += 1
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Log loss into wandb
        wandb.log({
            "Regression Loss": regression_loss,
            "Classification Loss": classification_loss,
            "Total Loss": total_loss,
        }, step=self.step)

        # Log into dict
        self.train_regr_loss[self.step] = regression_loss.item()

        return regression_loss, classification_loss, total_loss
    
    @torch.inference_mode
    def val_step(self, input_ids, regression_label, classification_label, idxs, attention_mask):

        # Put everything on device
        input_ids = input_ids.to(device)
        regression_label = regression_label.to(device)
        classification_label = classification_label.to(device)
        idxs = idxs.to(device)
        attention_mask = attention_mask.to(device)

        # Obtain regression and classification prediction
        regression_pred, class_logits = self.model(input_ids, attention_mask=None)
        
        # Take predictions from right heads
        current_batch_size = len(regression_label)
        batches = torch.arange(current_batch_size, device=device)
        regression_pred = regression_pred[batches, :, idxs]
        class_logits = class_logits[batches, :, idxs]

        # Calculate losses per head index
        unique_heads = idxs.unique()
        per_head_loss = {}

        for h in unique_heads:
            mask = (idxs == h)
            preds_h = regression_pred[mask]
            labels_h = regression_label[mask]
            loss_h = self.regression_loss_fn(preds_h, labels_h)
            per_head_loss[self.model.metadata.get_head_index(h.item())] = loss_h.item()

        regression_loss = self.regression_loss_fn(regression_pred, regression_label)
        classification_loss = self.classification_loss_fn(class_logits, classification_label)
        total_loss = regression_loss + self.cfg.comb_factor * classification_loss 

        # Log loss into wandb
        wandb.log({
            "Val Regression Loss": regression_loss,
            "Val Classification Loss": classification_loss,
            "Val Total Loss": total_loss,
        }, step=self.step)

        # Class pred for logging
        class_pred = F.sigmoid(class_logits)

        return  {
            "input_ids": input_ids,
            "regression_label": regression_label,
            "classification_label": classification_label,
            "regression_pred": regression_pred,
            "classification_pred": class_pred,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
            "total_loss": total_loss,
            "loss_dict": per_head_loss
        }
    
    def train(self):

        torch.manual_seed(self.cfg.seed)
        
        progress_bar = tqdm.tqdm(
            range(self.cfg.num_epochs * len(self.train_dataloader())),
            total=len(self.train_dataloader()) * self.cfg.num_epochs,
            colour="GREEN"
        )
        best_val_loss = 10000

        for epoch in range(self.cfg.num_epochs):
            
            self.model.train()

            for batch in self.train_dataloader():
                regression_loss, classification_loss, total_loss = self.train_step(
                    batch['input_ids'],
                    batch['signal'], 
                    batch['labels'],
                    batch['idxs'],
                    batch['attention_mask']
                )
                progress_bar.set_description(
                    f"Epoch: {epoch + 1}. R: {regression_loss:.4f}. C: {classification_loss:.4f}. T: {total_loss:.4f}"
                )
                progress_bar.update()
            
            # Start validation
            self.model.eval()
            val_loss_total = {head: 0 for head in self.model.metadata.heads()}

            # Store predictions and targets for both tasks
            preds_r, targets_r = [], []
            preds_c, targets_c = [], []

            for batch in self.test_dataloader():
                eval_dict = self.val_step(
                    batch['input_ids'],
                    batch['signal'], 
                    batch['labels'],
                    batch['idxs'],
                    batch['attention_mask']
                )
                # val_loss_total += eval_dict['total_loss']
                # Update val loss dict
                for head, loss in eval_dict["loss_dict"].items():
                    val_loss_total[head] += loss

                # Update lists for regression
                preds_r.append(eval_dict['regression_pred'].cpu())
                targets_r.append(eval_dict['regression_label'].cpu())

                # Update lists for classification
                preds_c.append(eval_dict['classification_pred'].cpu())
                targets_c.append(eval_dict['classification_label'].cpu())

            # Create evalutation lists
            preds_r = torch.cat(preds_r).numpy()
            targets_r = torch.cat(targets_r).numpy()

            # Flatten here because each nucleotide is independent
            preds_c = torch.cat(preds_c).view(-1).numpy()
            targets_c = torch.cat(targets_c).view(-1).numpy()

            # Convert probabilities -> binary predictions
            binary_preds = (preds_c > 0.5).astype(int)

            # Calculate classification metrics
            prec, rec, f1, _ = precision_recall_fscore_support(targets_c, binary_preds, average='binary')
            auroc = roc_auc_score(targets_c, preds_c)

            # Log everything on wandb
            wandb.log({
                "Precision": prec,
                "Recall": rec,
                "F1 score": f1,
                "AUROC": auroc
            }, step=self.step)

            avg_val_loss = {head: v / len(self.test_dataloader()) for head, v in val_loss_total.items()}
            self.val_regr_loss[self.step] = avg_val_loss


            # Save best model
            avg_val_loss = sum(avg_val_loss.values()) / len(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if_mlp = "_mlp" if self.model.cfg.use_MLP else ""
                if_pos = "_pos_embs" if self.model.cfg.use_pos_embs else ""
                if_moe = "_MoE" if self.model.cfg.use_MoE else ""
                save_path = f"{os.path.splitext(self.cfg.save_path)[0]}_{self.model_cfg.num_layers}L_{self.model_cfg.hidden_dim}D{if_mlp}{if_pos}{if_moe}"
                torch.save(self.model.state_dict(), f"{save_path}.pt")

                # Save config
                self.model.cfg.save_cfg(f"{save_path}.json")

                # Save metadata
                self.model.metadata.save_metadata(f"{save_path}.csv")
                
                print(f"Model saved in {save_path} with avg_val_loss={avg_val_loss}")

        # Finish training
        wandb.finish()
        return regression_loss, classification_loss
    

def save_loss_plot(train_losses: Dict[int, float], val_losses: Dict[int, Dict[str, float]], outfile: str, num_params: int, metadata: TrackMambaMetadata):

    # Extract x axes
    x_train_ax = np.array(list(train_losses.keys()))

    # Convert losses to np.array
    train_loss = np.array(list(train_losses.values()))

    # Define the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot values
    plt.plot(x_train_ax, train_loss, color="orange", label="Training Loss")

    # Plot validation loss per head
    for head in metadata.heads():
        val_loss = {
            step: val_losses[step][head]
            if val_losses[step][head] else None
            for step in val_losses.keys()
        }
        plt.plot(list(val_loss.keys()), list(val_loss.values()), label=f"Val loss: {head}")

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
    p.add_argument("--use_conv", action='store_true', default=False)
    p.add_argument("--use_pos_embs", action='store_true', default=False)
    p.add_argument("--use_bidirectionality", action='store_true', default=False)
    p.add_argument("--loss_path", type=str, required=False, default=None, help="Where to save losses.")
    p.add_argument("--lr", type=float, required=False, default=None, help="Learning rate.")
    p.add_argument("--bs", type=int, required=False, default=None, help="Batch size used for training.")


    # Parsing
    args = p.parse_args()
    
    # Initialize config files
    trainer_cfg = TrackMambaTrainerCfg(
        num_epochs=args.num_epochs if args.num_epochs else TrackMambaTrainerCfg().num_epochs,
        lr=args.lr if args.lr else TrackMambaTrainerCfg().lr,
        batch_size=args.bs if args.bs else TrackMambaTrainerCfg().batch_size,
        train_chroms=['chr1'],
        test_chroms=['chr2']
    )
    model_cfg = TrackMambaConfig(
        num_layers=args.num_layers if args.num_layers else TrackMambaConfig().num_layers,
        hidden_dim=args.hidden_dim if args.hidden_dim else TrackMambaConfig().hidden_dim,
        use_MLP=args.use_MLP,
        use_pos_embs=args.use_pos_embs,
        use_conv=args.use_conv,
        use_bidirectionality=args.use_bidirectionality
    )

    # Set seed for reproducibility
    torch.manual_seed(trainer_cfg.seed)
    torch.cuda.manual_seed(trainer_cfg.seed)
    torch.cuda.manual_seed_all(trainer_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize trainer and train
    trainer = TrackMambaTrainer(trainer_cfg=trainer_cfg, model_cfg=model_cfg)
    final_regr_loss, final_class_loss = trainer.train()

    # Update results
    if args.plot_path:
        
        # Calculate model parameters
        model_size = sum(p.numel() for p in trainer.model.parameters()) / 1e6

        # Save loss plots
        save_loss_plot(trainer.train_regr_loss, trainer.val_regr_loss, args.plot_path, model_size, trainer.model.metadata)

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
