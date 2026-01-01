import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import mlflow

# Ensure reproducibility across nodes
pl.seed_everything(42)

class WeatherDataset(Dataset):
    def __init__(self, data_path):
        # Spark writes a directory called 'data.parquet'
        parquet_path = os.path.join(data_path, "data.parquet")
        
        # --- 1. STRICT CHECK: Fail immediately if data is missing ---
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"CRITICAL ERROR: Data not found at {parquet_path}.\n"
                "Did the Spark preprocessing step finish successfully?"
            )

        print(f"Loading data from: {parquet_path}")
        # Pandas can read the folder structure created by Spark
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read Parquet file: {e}")

        # --- 2. Dynamic Feature Selection ---
        # Matches the columns created by your Spark script (ending in _norm)
        feature_cols = [c for c in df.columns if c.endswith('_norm')]
        
        if not feature_cols:
             raise ValueError("CRITICAL ERROR: No columns ending with '_norm' found. Check Spark logic.")

        print(f"✓ Loaded {len(df)} rows.")
        print(f"✓ Training with {len(feature_cols)} features: {feature_cols}")
        
        self.features = torch.FloatTensor(df[feature_cols].values)
        self.labels = torch.LongTensor(df['label_encoded'].values)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class WeatherClassifier(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.save_hyperparameters()
        
        # Simple MLP for classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2) # Output 2 classes (Rain / No Rain)
        )
        
    def forward(self, x): return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log validation metrics
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", acc, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def main():
    # --- 3. MLflow Setup ---
    mlflow_logger = MLFlowLogger(
        experiment_name="weather_forecasting",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"),
        log_model=True
    )

    # --- FIX 1: Explicitly Create Directory ---
    save_path = "/workspace/data/models"
    os.makedirs(save_path, exist_ok=True)

    # Save checkpoints based on Validation Loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="weather-best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True  # --- FIX 2: Always save a 'last.ckpt' as a fallback ---
    )

    # --- 4. Data Loading & Splitting ---
    # Path mapped in docker-compose
    full_dataset = WeatherDataset("/workspace/data/processed")
    
    # 80% Train / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Num_workers=0 is safer for Docker shared memory limits
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    input_dim = full_dataset.features.shape[1] 
    model = WeatherClassifier(input_dim=input_dim)

    # --- 5. Trainer Setup ---
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator="cpu", # Change to "gpu" if available
        devices=1,
        num_nodes=world_size,
        strategy=DDPStrategy(find_unused_parameters=False) if world_size > 1 else "auto",
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5
    )
    
    # Train using both loaders
    trainer.fit(model, train_loader, val_loader)
    
    # --- 6. Post-Training Log ---
    if trainer.global_rank == 0:
        # Determine which model to verify/upload
        best_path = checkpoint_callback.best_model_path
        if not os.path.exists(best_path):
            print(f"⚠ Best model not found at {best_path}. Checking for 'last.ckpt'...")
            best_path = os.path.join(save_path, "last.ckpt")

        print(f"Training finished. Saving model path: {best_path}")
        
        if os.path.exists(best_path):
            # Explicitly upload the best checkpoint to MLflow
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=best_path,
                artifact_path="best_checkpoints"
            )
            print("✓ Model uploaded to MLflow!")
        else:
            print("✗ CRITICAL: No model file found to upload!")

if __name__ == "__main__":
    main()