import torch
from torch.utils.data import DataLoader
import sys
import os

from transformer import G2PTransformer, P2GTransformer, train_model
from PhoneDataset import PhoneDataset

# -----------------------------------------------------------------------------
# Config / Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 128
DATA_DIR = 'data'
MODELS_DIR = 'models'
MAX_WORDS = 200_000
MAX_LEN = 48
TRAIN_SPLIT = 0.95
EPOCHS = 5
LR = 0.001

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_device():
    """Pick GPU if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataset(language: str, direction: str):
    """Load dataset for a given language and set direction."""
    data_path = os.path.join(DATA_DIR, f"{language}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found for language '{language}' at {data_path}")
    
    ds = PhoneDataset(
        data_path=data_path,
        max_words=MAX_WORDS,
        max_len=MAX_LEN
    )
    ds.set_direction(direction)
    return ds

def create_dataloaders(dataset, train_size: float, batch_size: int):
    """Random-split a dataset, then build train/val DataLoaders."""
    train_ds, val_ds = dataset.random_split(train_size=train_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader

def build_p2g_model(dataset) -> torch.nn.Module:
    """Instantiate P2G Transformer given the dataset's vocab sizes."""
    return P2GTransformer(
        phone_vocab_size=len(dataset.phone_to_idx),
        char_vocab_size=len(dataset.char_to_idx),
        d_model=256, nhead=8, num_layers=4, dim_feedforward=512, max_len=MAX_LEN
    )

def build_g2p_model(dataset) -> torch.nn.Module:
    """Instantiate G2P Transformer given the dataset's vocab sizes."""
    return G2PTransformer(
        char_vocab_size=len(dataset.char_to_idx),
        phone_vocab_size=len(dataset.phone_to_idx),
        d_model=256, nhead=8, num_layers=4, dim_feedforward=512, max_len=MAX_LEN
    )

def run_training(model, train_loader, val_loader, device, num_epochs, language):
    """Move model to device, print parameter count, then train."""
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training {model.__class__.__name__} for {language.upper()}...")
    print(f"Model has {total_params:,} parameters.")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
        
    train_model(model, train_loader, val_loader, device,
                num_epochs=num_epochs, language=language, learning_rate=LR)

def train_language(language: str, device: torch.device):
    """Run the full training pipeline for a single language."""
    print("-" * 60)
    print(f"Processing language: {language.upper()}")
    print("-" * 60)

    # ----- G2P (Grapheme to Phoneme) Training -----
    print("\nPreparing G2P dataset...")
    g2p_ds = prepare_dataset(language=language, direction='g2p')
    g2p_train_loader, g2p_val_loader = create_dataloaders(g2p_ds, TRAIN_SPLIT, BATCH_SIZE)
    g2p_model = build_g2p_model(g2p_ds)
    run_training(g2p_model, g2p_train_loader, g2p_val_loader, device,
                 num_epochs=EPOCHS, language=language)

    # ----- P2G (Phoneme to Grapheme) Training -----
    print("\nPreparing P2G dataset...")
    p2g_ds = prepare_dataset(language=language, direction='p2g')
    p2g_train_loader, p2g_val_loader = create_dataloaders(p2g_ds, TRAIN_SPLIT, BATCH_SIZE)
    p2g_model = build_p2g_model(p2g_ds)
    run_training(p2g_model, p2g_train_loader, p2g_val_loader, device,
                 num_epochs=EPOCHS, language=language)

    print(f"\nFinished training for {language.upper()}. Models saved to '{MODELS_DIR}/'.")

def main():
    """Main function to parse args and run training."""
    languages_to_train = sys.argv[1:]

    if not languages_to_train:
        print("Usage: python train.py <lang1> <lang2> ...")
        print("Example: python train.py en_US fr_FR")
        try:
            available = [f.replace('.txt', '') for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
            if available:
                print("\nAvailable languages in 'data/' directory:")
                print(" ".join(available))
        except FileNotFoundError:
            print(f"\nError: 'data' directory not found. Please create it and add data files (e.g., data/en_US.txt).")
        return

    device = get_device()
    print(f"Using device: {device}")

    for lang in languages_to_train:
        try:
            train_language(lang, device)
        except Exception as e:
            print(f"\nERROR: An uncaught exception occurred while training '{lang}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()