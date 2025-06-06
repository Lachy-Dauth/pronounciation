"""
Helper script to prepare and clean the data files
"""
import pandas as pd
import re

def clean_cmu_dict(input_path: str, output_path: str):
    """Clean CMU dictionary and extract valid entries"""
    clean_entries = []
    
    with open(input_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith(';;;'):
                continue
            
            # Split on first space(s)
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            
            word, phones_str = parts
            
            # Skip alternative pronunciations (contains parentheses)
            if '(' in word:
                continue
            
            # Skip words that don't start with A-Z
            if not word[0].isalpha():
                continue
            
            # Clean up phones (remove stress markers)
            phones = []
            for phone in phones_str.split():
                clean_phone = re.sub(r'\d', '', phone)
                phones.append(clean_phone)
            
            clean_entries.append(f"{word}\t{' '.join(phones)}")
    
    with open(output_path, 'w') as f:
        for entry in clean_entries:
            f.write(entry + '\n')
    
    print(f"Cleaned {len(clean_entries)} entries and saved to {output_path}")

def prepare_frequency_data(input_path: str, output_path: str):
    """Prepare frequency data in the correct format"""
    # Assuming the input is in the format shown in your example
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print(f"Prepared frequency data with {len(df)} entries")

if __name__ == "__main__":
    # Example usage:
    # clean_cmu_dict('cmudict-0.7b', 'cmudict_clean.txt')
    # prepare_frequency_data('word_frequencies_raw.csv', 'word_frequencies.csv')
    pass