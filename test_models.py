import sys
from pathlib import Path
from typing import Dict, List
import torch
import os
import re

from PhoneDataset import PhoneDataset
from transformer import G2PTransformer, P2GTransformer, inference

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
MAX_LEN = 48  # Must match training configuration
MAX_WORDS = 200000 # Must match training configuration

# ---------------------------------------------------------------------------
# Language Builders
# ---------------------------------------------------------------------------

def prepare_dataset(language: str, direction: str):
    """Loads the dataset needed for model inference."""
    data_path = DATA_DIR / f"{language}.txt"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found for language '{language}' at {data_path}")
    
    ds = PhoneDataset(data_path=str(data_path), max_words=MAX_WORDS, max_len=MAX_LEN)
    ds.set_direction(direction)
    return ds

def build_p2g_model(dataset) -> torch.nn.Module:
    """Instantiates a P2G Transformer for inference."""
    return P2GTransformer(
        phone_vocab_size=len(dataset.phone_to_idx),
        char_vocab_size=len(dataset.char_to_idx),
        d_model=256, nhead=8, num_layers=4, dim_feedforward=512, max_len=MAX_LEN
    )

def build_g2p_model(dataset) -> torch.nn.Module:
    """Instantiates a G2P Transformer for inference."""
    return G2PTransformer(
        char_vocab_size=len(dataset.char_to_idx),
        phone_vocab_size=len(dataset.phone_to_idx),
        d_model=256, nhead=8, num_layers=4, dim_feedforward=512, max_len=MAX_LEN
    )

# ---------------------------------------------------------------------------
# Dynamic Language and Model Loading
# ---------------------------------------------------------------------------

def find_available_languages(model_dir: Path) -> List[str]:
    """Scans the model directory to find fully trained languages."""
    if not model_dir.exists(): return []
    
    g2p_pattern = re.compile(r"(.+)_g2p_best_model\.pth")
    p2g_pattern = re.compile(r"(.+)_p2g_best_model\.pth")
    g2p_langs = {g2p_pattern.match(f).group(1) for f in os.listdir(model_dir) if g2p_pattern.match(f)}
    p2g_langs = {p2g_pattern.match(f).group(1) for f in os.listdir(model_dir) if p2g_pattern.match(f)}
    
    return sorted(list(g2p_langs.intersection(p2g_langs)))

class LazyLanguage:
    """Lazy-loads datasets and models for a language the first time it's used."""

    def __init__(self, name: str, device: torch.device):
        self.name, self.device, self.loaded = name, device, False
        self.g2p_dataset, self.p2g_dataset = None, None
        self.g2p_model, self.p2g_model = None, None

    def _load_weights(self, model, tag: str):
        path = MODEL_DIR / f"{self.name}_{tag}_best_model.pth"
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            print(f"Loaded {self.name.upper()} {tag.upper()} weights from {path}")
        except FileNotFoundError:
            print(f"[WARN] Weights not found for {self.name}.{tag} at {path}")

    def _ensure_loaded(self):
        if self.loaded: return
        print(f"\n[INFO] Loading language '{self.name.upper()}' for the first time...")
        
        self.g2p_dataset = prepare_dataset(language=self.name, direction="g2p")
        self.p2g_dataset = prepare_dataset(language=self.name, direction="p2g")

        self.g2p_model = build_g2p_model(self.g2p_dataset).to(self.device)
        self.p2g_model = build_p2g_model(self.p2g_dataset).to(self.device)

        self._load_weights(self.g2p_model, "g2p")
        self._load_weights(self.p2g_model, "p2g")
        self.loaded = True

    def g2p(self, sentence: str) -> str:
        self._ensure_loaded()
        ipa_words = [" ".join(inference(self.g2p_model, self.g2p_dataset, word, self.device)) for word in sentence.strip().split()]
        return " | ".join(ipa_words)

    def p2g(self, ipa_sentence: str) -> str:
        self._ensure_loaded()
        groups = [g.strip() for g in ipa_sentence.split('|')]
        words = [inference(self.p2g_model, self.p2g_dataset, grp.split(), self.device) for grp in groups]
        return " ".join(words)

    def round_trip(self, sentence: str):
        self._ensure_loaded()
        ipa = self.g2p(sentence)
        recon = self.p2g(ipa)
        print(f"Original : {sentence}")
        print(f"IPA      : {ipa}")
        print(f"Rebuilt  : {recon}")
        print(f"Match    : {'✓' if sentence.lower() == recon.lower() else '✗'}")

class Tester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
        self.lang_cache: Dict[str, LazyLanguage] = {}
        self.available_languages = find_available_languages(MODEL_DIR)
        
        if not self.available_languages:
            print(f"[WARN] No models found in '{MODEL_DIR}/'. Please train models using train.py.")
        else:
            print(f"[INFO] Available languages: {', '.join(self.available_languages)}")

    def _get_lang(self, name: str) -> LazyLanguage:
        if name not in self.lang_cache:
            self.lang_cache[name] = LazyLanguage(name, self.device)
        return self.lang_cache[name]

    def _parse_lang(self, args: List[str]):
        if args and args[0].lower() in self.available_languages:
            return args[0].lower(), " ".join(args[1:])
        if len(self.available_languages) == 1:
            return self.available_languages[0], " ".join(args)
        return None, " ".join(args)

    def repl(self):
        print("\n--- Interactive Pronunciation Tool ---")
        print("Commands: g2p, p2g, round, help, quit")
        if len(self.available_languages) > 1:
            print(f"Specify a language, e.g., 'g2p {self.available_languages[0]} hello world'")

        while True:
            try:
                raw_input = input(">>> ").strip()
                if not raw_input: continue
                
                cmd, *args = raw_input.split()
                cmd = cmd.lower()

                if cmd in ("quit", "exit"): break
                if cmd == "help":
                    print("\nUsage: <command> [<language>] <text>")
                    print("  g2p en_us hello world")
                    print("  p2g fr_fr b ɔ̃ ʒ u ʁ")
                    print("  round de hallo welt")
                    print("\nIf only one language is available, specifying it is optional.")
                    continue

                lang, payload = self._parse_lang(args)

                if not lang:
                    print(f"[ERROR] Language missing or invalid. Choose from: {', '.join(self.available_languages)}")
                    continue
                if not payload:
                    print(f"[ERROR] No text provided for processing.")
                    continue
                
                handler = getattr(self, f"cmd_{cmd}", None)
                if handler:
                    handler(lang, payload)
                else:
                    print(f"Unknown command: '{cmd}'. Type 'help'.")

            except (EOFError, KeyboardInterrupt): break
            except Exception as e: print(f"[ERROR] {e}")
        print("\nExiting.")

    def cmd_g2p(self, lang: str, text: str):
        ipa = self._get_lang(lang).g2p(text)
        print(f"[{lang.upper()}] {text} → {ipa}")

    def cmd_p2g(self, lang: str, ipa: str):
        text = self._get_lang(lang).p2g(ipa)
        print(f"[{lang.upper()}] {ipa} → {text}")

    def cmd_round(self, lang: str, text: str):
        self._get_lang(lang).round_trip(text)

    def cli(self):
        if len(sys.argv) < 2:
            self.repl()
            return
            
        cmd = sys.argv[1].lower()
        lang, payload = self._parse_lang(sys.argv[2:])

        if not lang:
            print(f"Language missing or invalid. Choose from: {', '.join(self.available_languages)}")
            return
        if not payload:
            print("No text provided.")
            return

        handler = getattr(self, f"cmd_{cmd}", None)
        if handler:
            handler(lang, payload)
        else:
            print(f"Unknown command: {cmd}. Use g2p, p2g, or round.")

if __name__ == "__main__":
    Tester().cli()