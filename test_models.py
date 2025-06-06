"""
Interactive testing script for both G2P and P2G models with IPA support
"""
import torch
import sys
import re
from g2p_transformer import G2PTransformer, inference as g2p_inference
from p2g_transformer import P2GTransformer, inference as p2g_inference
from PhoneDataset import PhoneDataset as Dataset
from arpabet_ipa_converter import ARPAbetIPAConverter

class ModelTester:
    def __init__(self, data_path='cmudict.dict', freq_path='word_frequencies.csv'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize IPA converter
        self.ipa_converter = ARPAbetIPAConverter()
        
        # Load datasets
        print("Loading datasets...")
        self.g2p_dataset = Dataset(data_path, freq_path, max_words=200000)
        self.p2g_dataset = Dataset(data_path, freq_path, max_words=200000)
        self.g2p_dataset.set_direction('g2p')
        
        # Initialize models
        self.g2p_model = G2PTransformer(
            char_vocab_size=len(self.g2p_dataset.char_to_idx),
            phone_vocab_size=len(self.g2p_dataset.phone_to_idx),
            d_model=128,
            nhead=16,
            num_layers=4,
            dim_feedforward=256,
            max_len=32
        ).to(self.device)
                
        self.p2g_model = P2GTransformer(
            phone_vocab_size=len(self.p2g_dataset.phone_to_idx),
            char_vocab_size=len(self.p2g_dataset.char_to_idx),
            d_model=128,
            nhead=16,
            num_layers=4,
            dim_feedforward=256,
            max_len=32
        ).to(self.device)
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load trained model weights"""
        try:
            self.g2p_model.load_state_dict(torch.load('best_g2p_model.pth', map_location=self.device))
            print("Loaded G2P model successfully")
        except FileNotFoundError:
            print("Warning: G2P model not found. Please train the G2P model first.")
        
        try:
            self.p2g_model.load_state_dict(torch.load('best_p2g_model.pth', map_location=self.device))
            print("Loaded P2G model successfully")
        except FileNotFoundError:
            print("Warning: P2G model not found. Please train the P2G model first.")
    
    def test_g2p(self, word: str):
        """Test grapheme-to-phoneme conversion"""
        try:
            phones = g2p_inference(self.g2p_model, self.g2p_dataset, word, self.device)
            return phones
        except Exception as e:
            return f"Error: {e}"
    
    def test_p2g(self, phones_input):
        """Test phoneme-to-grapheme conversion"""
        try:
            if isinstance(phones_input, str):
                # Parse phone string (space-separated)
                phones = phones_input.strip().split()
            else:
                phones = phones_input
            
            word = p2g_inference(self.p2g_model, self.p2g_dataset, phones, self.device)
            return word
        except Exception as e:
            return f"Error: {e}"
    
    def test_ipa_to_word(self, ipa_str: str):
        """Test IPA to word conversion via ARPAbet"""
        try:
            # Convert IPA to ARPAbet
            arpabet_phones = self.ipa_converter.ipa_to_arpabet_convert(ipa_str)
            
            # Convert ARPAbet to word using P2G model
            word = self.test_p2g(arpabet_phones)
            return word, arpabet_phones
        except Exception as e:
            return f"Error: {e}", None
    
    def format_phonemes_with_ipa(self, phones):
        """Format phonemes showing both ARPAbet and IPA"""
        if not isinstance(phones, list):
            return str(phones)
        
        try:
            ipa = self.ipa_converter.arpabet_to_ipa_convert(phones)
            return f"{' '.join(phones)} [{ipa}]"
        except:
            return ' '.join(phones)
    
    def test_round_trip(self, word: str):
        """Test round-trip conversion: word -> phones -> word"""
        try:
            # G2P: word -> phones
            phones = self.test_g2p(word)
            if isinstance(phones, str) and phones.startswith("Error"):
                return f"G2P failed: {phones}"
            
            # Convert to IPA for display
            ipa = self.ipa_converter.arpabet_to_ipa_convert(phones)
            
            # P2G: phones -> word
            reconstructed = self.test_p2g(phones)
            if isinstance(reconstructed, str) and reconstructed.startswith("Error"):
                return f"P2G failed: {reconstructed}"
            
            return {
                'original': word,
                'phones': phones,
                'ipa': ipa,
                'phones_formatted': self.format_phonemes_with_ipa(phones),
                'reconstructed': reconstructed,
                'match': word.lower() == reconstructed.lower()
            }
        except Exception as e:
            return f"Error: {e}"
    
    def interactive_mode(self):
        """Interactive testing mode"""
        print("\n" + "="*60)
        print("INTERACTIVE MODEL TESTER WITH IPA SUPPORT")
        print("="*60)
        print("Commands:")
        print("  g2p <word>           - Convert word to phonemes (ARPAbet + IPA)")
        print("  p2g <phones>         - Convert phonemes to word (space-separated)")
        print("  ipa <IPA>            - Convert IPA to word via ARPAbet")
        print("  round <word>         - Test round-trip conversion")
        print("  batch <words>        - Test multiple words (comma-separated)")
        print("  examples             - Show example commands")
        print("  quit                 - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                parts = user_input.split(' ', 1)
                if len(parts) < 2:
                    print("Invalid command. Type 'examples' for help.")
                    continue
                
                command, args = parts
                command = command.lower()
                
                if command == 'g2p':
                    word = args.strip()
                    phones = self.test_g2p(word)
                    if isinstance(phones, list):
                        formatted = self.format_phonemes_with_ipa(phones)
                        print(f"'{word}' -> {formatted}")
                    else:
                        print(phones)
                
                elif command == 'p2g':
                    phones_str = args.strip()
                    word = self.test_p2g(phones_str)
                    print(f"'{phones_str}' -> {word}")
                
                elif command == 'ipa':
                    ipa_str = args.strip()
                    word, arpabet_phones = self.test_ipa_to_word(ipa_str)
                    if arpabet_phones:
                        print(f"'{ipa_str}' -> {' '.join(arpabet_phones)} -> {word}")
                    else:
                        print(f"'{ipa_str}' -> {word}")
                
                elif command == 'round':
                    word = args.strip()
                    result = self.test_round_trip(word)
                    if isinstance(result, dict):
                        print(f"Original:      {result['original']}")
                        print(f"Phonemes:      {result['phones_formatted']}")
                        print(f"Reconstructed: {result['reconstructed']}")
                        print(f"Match:         {'✓' if result['match'] else '✗'}")
                    else:
                        print(result)
                
                elif command == 'batch':
                    words = [w.strip() for w in args.split(',')]
                    print(f"\nTesting {len(words)} words:")
                    print("-" * 80)
                    matches = 0
                    for word in words:
                        if word:
                            result = self.test_round_trip(word)
                            if isinstance(result, dict):
                                status = "✓" if result['match'] else "✗"
                                phones_display = result['phones_formatted']
                                if len(phones_display) > 35:
                                    phones_display = phones_display[:32] + "..."
                                print(f"{word:12} -> {phones_display:35} -> {result['reconstructed']:12} {status}")
                                if result['match']:
                                    matches += 1
                            else:
                                print(f"{word:12} -> ERROR")
                    print(f"\nAccuracy: {matches}/{len(words)} ({100*matches/len(words):.1f}%)")
                
                else:
                    print("Unknown command. Type 'examples' for help.")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_examples(self):
        """Show example commands"""
        print("\nExample commands:")
        print("  g2p hello")
        print("  g2p pronunciation")
        print("  p2g HH EH 1 L OW 0")
        print("  p2g P R AH 0 N AH 0 N S IY 1 EY 2 SH AH 0 N")
        print("  ipa həˈloʊ")
        print("  ipa prənʌnsiˈeɪʃən")
        print("  round hello")
        print("  round transformer")
        print("  batch hello,world,cat,dog,pronunciation")

def main():
    """Main function with command line interface"""
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        tester = ModelTester()
        
        if command == 'g2p' and len(sys.argv) >= 3:
            word = ' '.join(sys.argv[2:])
            phones = tester.test_g2p(word)
            if isinstance(phones, list):
                formatted = tester.format_phonemes_with_ipa(phones)
                print(formatted)
            else:
                print(phones)
        
        elif command == 'p2g' and len(sys.argv) >= 3:
            phones_str = ' '.join(sys.argv[2:])
            word = tester.test_p2g(phones_str)
            print(word)
        
        elif command == 'ipa' and len(sys.argv) >= 3:
            ipa_str = ' '.join(sys.argv[2:])
            word, arpabet_phones = tester.test_ipa_to_word(ipa_str)
            if arpabet_phones:
                print(f"ARPAbet: {' '.join(arpabet_phones)}")
                print(f"Word: {word}")
            else:
                print(word)
        
        elif command == 'round' and len(sys.argv) >= 3:
            word = ' '.join(sys.argv[2:])
            result = tester.test_round_trip(word)
            if isinstance(result, dict):
                print(f"Original: {result['original']}")
                print(f"Phonemes: {result['phones_formatted']}")
                print(f"Reconstructed: {result['reconstructed']}")
                print(f"Match: {'Yes' if result['match'] else 'No'}")
            else:
                print(result)
        
        else:
            print("Usage:")
            print("  python test_models.py g2p <word>")
            print("  python test_models.py p2g <phonemes>")
            print("  python test_models.py ipa <IPA_transcription>")
            print("  python test_models.py round <word>")
            print("  python test_models.py  (for interactive mode)")
    
    else:
        # Interactive mode
        tester = ModelTester()
        tester.interactive_mode()

if __name__ == "__main__":
    main()