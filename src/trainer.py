import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import tqdm
import torch.nn.functional as F
from TrackMamba import TrackMamba
from config import TrackMambaTrainerCfg, TrackMambaConfig
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset
from dataset import DNAseSeqDataset
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, precision_recall_fscore_support, roc_auc_score
import pandas as pd
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

        # JSON -> pd.DataFrame -> hf Dataset to avoid overflow
        df = pd.read_json(os.path.join(trainer_cfg.data_path, trainer_cfg.data_file))

        # Split dataset based on train and test chroms
        train_df = df[df["chrom"].isin(self.cfg.train_chroms)]
        test_df = df[df["chrom"].isin(self.cfg.test_chroms)]

        self.train_dataset = Dataset.from_pandas(train_df).map(self.preprocess_function, batched=True)
        self.test_dataset = Dataset.from_pandas(test_df).map(self.preprocess_function, batched=True)

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
    
    def train_step(self, input_ids, regression_label, class_label):

        # Put everything on device
        input_ids = input_ids.to(device)
        regression_label = regression_label.to(device)
        class_label = class_label.to(device)

        # Obtain regression and classification prediction
        regression_pred, class_logits = self.model(input_ids)

        # Calculate losses
        regression_loss = self.regression_loss_fn(regression_pred, regression_label)
        classification_loss = self.classification_loss_fn(class_logits, class_label.float())
        total_loss = regression_loss + self.cfg.comb_factor * classification_loss

        # Backprop
        self.step += 1
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Log loss into wandb
        wandb.log({
            "Regression Loss": regression_loss,
            "Classification Loss": classification_loss,
            "Total Loss": total_loss,
        }, step=self.step)

        return regression_loss, classification_loss, total_loss
    
    @torch.inference_mode
    def val_step(self, input_ids, regression_label, classification_label):

        # Put everything on device
        input_ids = input_ids.to(device)
        regression_label = regression_label.to(device)
        classification_label = classification_label.to(device)

        # Obtain regression and classification prediction
        regression_pred, class_logits = self.model(input_ids)

        # Calculate losses
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
            "total_loss": total_loss
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
                    batch['labels']
                )
                progress_bar.set_description(
                    f"Epoch: {epoch + 1}. R: {regression_loss:.4f}. C: {classification_loss:.4f}. T: {total_loss:.4f}"
                )
                progress_bar.update()
            
            # Start validation
            self.model.eval()
            val_loss_total = 0

            # Store predictions and targets for both tasks
            preds_r, targets_r = [], []
            preds_c, targets_c = [], []

            for batch in self.test_dataloader():
                eval_dict = self.val_step(
                    batch['input_ids'],
                    batch['signal'], 
                    batch['labels']
                )
                val_loss_total += eval_dict['total_loss']

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

            # Calculate regression metrics
            pearson_corr = pearsonr(preds_r, targets_r)
            spearman_corr = spearmanr(preds_r, targets_r)

            # Convert probabilities -> binary predictions
            binary_preds = (preds_c > 0.5).astype(int)

            # Calculate classification metrics
            prec, rec, f1, _ = precision_recall_fscore_support(targets_c, binary_preds, average='binary')
            auroc = roc_auc_score(targets_c, preds_c)

            # Log everything on wandb
            wandb.log({
                "Pearson Corr": pearson_corr[0],
                "Spearman Corr": spearman_corr[0],
                "Precision": prec,
                "Recall": rec,
                "F1 score": f1,
                "AUROC": auroc
            }, step=self.step)

            # Save best model
            avg_val_loss = val_loss_total / len(self.test_dataloader())
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if_mlp = "_mlp" if self.model.cfg.use_MLP else ""
                if_pos = "_pos_embs" if self.model.cfg.use_pos_embs else ""
                save_path = f"{os.path.splitext(self.cfg.save_path)[0]}_{self.model_cfg.num_layers}L_{self.model_cfg.hidden_dim}D{if_mlp}{if_pos}.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved in {save_path} with avg_val_loss={avg_val_loss}")

        # Finish training
        wandb.finish()
        return regression_loss, classification_loss

if __name__ == '__main__':

    # Include argparse for output file
    p = argparse.ArgumentParser()
    p.add_argument("--results_file", type=str, required=False, default=None, help="Path to results file.")
    p.add_argument("--num_layers", type=int, required=False, default=None, help="Number of Mamba layers.")
    p.add_argument("--hidden_dim", type=int, required=False, default=None, help="Dimensionality of the residual stream.")
    p.add_argument("--num_epochs", type=int, required=False, default=None, help="Number of training epochs.")
    p.add_argument("--use_MLP", action='store_true', default=False)
    p.add_argument("--use_pos_embs", action='store_true', default=False)


    # Parsing
    args = p.parse_args()
    
    # Initialize config files
    trainer_cfg = TrackMambaTrainerCfg(
        num_epochs=args.num_epochs if args.num_epochs else TrackMambaTrainerCfg().num_epochs
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
    final_regr_loss, final_class_loss = trainer.train()

    # Update results
    if args.results_file:
        
        # Calculate model parameters
        model_size = sum(p.numel() for p in trainer.model.parameters()) / 1e6

        with open(args.results_file, "a") as outfile:
            outfile.write(
                f"{model_size:.2f}\t{model_cfg.num_layers}\t{model_cfg.hidden_dim}\t{trainer_cfg.num_epochs}\t{final_regr_loss:.4f}\t{final_class_loss:.4f}\n"
            )