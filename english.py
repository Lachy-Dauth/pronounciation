import torch
from torch.utils.data import DataLoader

from transformer import G2PTransformer, P2GTransformer, train_model, inference

from PhoneDataset import PhoneDataset as Dataset, WeightedSampler

# -----------------------------------------------------------------------------
# Config / Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
DATA_PATH = 'cmudict.dict'
FREQ_PATH = 'word_frequencies.csv'
MAX_WORDS = 200_000
MAX_LEN = 32
TRAIN_SPLIT = 0.8
P2G_EPOCHS = 0
G2P_EPOCHS = 0
LR = 0.001
# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_device():
    """Pick GPU if available, else CPU."""
    # return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model, path):
    """Load trained model weights"""
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Loaded model weights from {path}")

def prepare_dataset(direction: str):
    """Load CMU-Dict dataset and set direction ('p2g' or 'g2p')."""
    ds = Dataset(
        data_path=DATA_PATH,
        freq_path=FREQ_PATH,
        max_words=MAX_WORDS,
        max_len=MAX_LEN
    )
    ds.set_direction(direction)
    return ds

def create_dataloaders(dataset, train_size: float, batch_size: int):
    """Random-split a dataset, then build train/val DataLoaders with weighted sampling."""
    train_ds, val_ds = dataset.random_split(train_size=train_size)
    train_sampler = WeightedSampler(train_ds)
    val_sampler = WeightedSampler(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, val_loader

def build_p2g_model(dataset) -> torch.nn.Module:
    """Instantiate P2G Transformer given the dataset's vocab sizes."""
    return P2GTransformer(
        phone_vocab_size=len(dataset.phone_to_idx),
        char_vocab_size=len(dataset.char_to_idx),
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_len=MAX_LEN
    )

def build_g2p_model(dataset) -> torch.nn.Module:
    """Instantiate G2P Transformer given the dataset's vocab sizes."""
    return G2PTransformer(
        char_vocab_size = len(dataset.char_to_idx),
        phone_vocab_size=len(dataset.phone_to_idx),
        d_model=128,
        nhead=16,
        num_layers=4,
        dim_feedforward=256,
        max_len=MAX_LEN
    )

def run_training(model, train_loader, val_loader, device, num_epochs, language, learning_rate=LR):
    """Move model to device, print parameter count, then train."""
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model.__class__.__name__} has {total_params:,} parameters")
    train_model(model, train_loader, val_loader, device,
                num_epochs=num_epochs, language=language, learning_rate=learning_rate)

def run_inference_p2g(model, dataset, device, test_phones):
    print("\nTesting P2G inference...")
    for phones in test_phones:
        word = inference(model, dataset, phones, device)
        print(f"{' '.join(phones)} -> {word}")

def run_inference_g2p(model, dataset, device, test_words):
    print("\nTesting G2P inference...")
    for word in test_words:
        phones = inference(model, dataset, word, device)
        print(f"{word} -> {' '.join(phones)}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"Using device: {device}")
   
    # ----- G2P -----
    print("\nSwitching to G2P direction...")
    g2p_ds = prepare_dataset(direction='g2p')
    train_loader, val_loader = create_dataloaders(g2p_ds, TRAIN_SPLIT, BATCH_SIZE)

    g2p_model = build_g2p_model(g2p_ds)
    try:
        load_model(g2p_model, 'models/english_g2p_best_model.pth')
    except Exception as e:
        print(f"Error loading G2P model: {e}")
        g2p_model = build_g2p_model(g2p_ds)
        print("Failed to load G2P model, starting fresh training...")
    print("Starting G2P training...")
    run_training(g2p_model, train_loader, val_loader, device,
                 num_epochs=G2P_EPOCHS, language='English', learning_rate=LR)

    test_words = ['hello', 'world', 'pronunciation', 'transformer']
    run_inference_g2p(g2p_model, g2p_ds, device, test_words)

    # ----- P2G -----
    print("Loading dataset for P2G...")
    p2g_ds = prepare_dataset(direction='p2g')
    train_loader, val_loader = create_dataloaders(p2g_ds, TRAIN_SPLIT, BATCH_SIZE * 4)

    p2g_model = build_p2g_model(p2g_ds)

    try:
        load_model(p2g_model, 'models/english_p2g_best_model.pth')
    except Exception as e:
        p2g_model = build_p2g_model(p2g_ds)
        print("Failed to load P2G model, starting fresh training...")

    print("Starting P2G training...")
    run_training(p2g_model, train_loader, val_loader, device,
                 num_epochs=P2G_EPOCHS, language='English', learning_rate=LR)

    test_phones = [
        ['h','ɛ','l','oʊ'], ['w','ɜː','l','d'],
        ['k','æ','t'],       ['d','ɔː','g']
    ]
    run_inference_p2g(p2g_model, p2g_ds, device, test_phones)

if __name__ == "__main__":
    main()