"""
ARPAbet to IPA Converter

This module provides a class to convert between ARPAbet phonetic notation
and International Phonetic Alphabet (IPA) symbols.

ARPAbet is a phonetic transcription code developed by ARPA (Advanced Research 
Projects Agency) for American English pronunciation, commonly used in speech 
recognition and text-to-speech systems.
"""

class ARPAbetIPAConverter(object):
    """
    A class to convert between ARPAbet and IPA phonetic notations.
    
    Supports conversion of American English phonemes between the two systems.
    """
    
    def __init__(self):
        """Initialize the converter with mapping dictionaries."""
        # ARPAbet to IPA mapping
        self.arpabet_to_ipa = {
            # Vowels
            'AA': 'ɑ',      # father, cot
            'AE': 'æ',      # cat, bat
            'AH': 'ʌ',      # cut, but
            'AO': 'ɔ',      # caught, law
            'AW': 'aʊ',     # cow, how
            'AY': 'aɪ',     # eye, my
            'EH': 'ɛ',      # bed, red
            'ER': 'ɝ',      # bird, hurt
            'EY': 'eɪ',     # cake, day
            'IH': 'ɪ',      # bit, hit
            'IY': 'i',      # beat, see
            'OW': 'oʊ',     # boat, show
            'OY': 'ɔɪ',     # boy, toy
            'UH': 'ʊ',      # book, put
            'UW': 'u',      # boot, crew
            
            # Consonants
            'B': 'b',       # bat
            'CH': 'tʃ',     # chair
            'D': 'd',       # dog
            'DH': 'ð',      # then, this
            'F': 'f',       # fish
            'G': 'ɡ',       # go
            'HH': 'h',      # house
            'JH': 'dʒ',     # judge
            'K': 'k',       # cat
            'L': 'l',       # light
            'M': 'm',       # man
            'N': 'n',       # no
            'NG': 'ŋ',      # sing
            'P': 'p',       # put
            'R': 'ɹ',       # red
            'S': 's',       # see
            'SH': 'ʃ',      # she
            'T': 't',       # top
            'TH': 'θ',      # think
            'V': 'v',       # very
            'W': 'w',       # way
            'Y': 'j',       # yes
            'Z': 'z',       # zoo
            'ZH': 'ʒ',      # measure
        }
        
        # Create reverse mapping (IPA to ARPAbet)
        self.ipa_to_arpabet = {v: k for k, v in self.arpabet_to_ipa.items()}
        
        # Handle stress markers
        self.stress_markers = {
            '0': '',        # no stress
            '1': 'ˈ',       # primary stress
            '2': 'ˌ',       # secondary stress
        }
        
        # Reverse stress markers
        self.reverse_stress = {v: k for k, v in self.stress_markers.items() if v}

    def convert_character_from_arpabet(self, char):
        """Convert a single character from ARPAbet to IPA."""
        if char in self.arpabet_to_ipa:
            return self.arpabet_to_ipa[char]
        elif char in self.stress_markers:
            return self.stress_markers[char]
        raise ValueError(f"Unknown ARPAbet character: {char}")
    
    def convert_character_from_ipa(self, char):
        """Convert a single character from IPA to ARPAbet."""
        if char in self.ipa_to_arpabet:
            return self.ipa_to_arpabet[char]
        elif char in self.reverse_stress:
            return self.reverse_stress[char]
        raise ValueError(f"Unknown IPA character: {char}")
    
    def arpabet_to_ipa_convert(self, arpabet_sequence):
        """
        Convert ARPAbet sequence to IPA.
        
        Args:
            arpabet_sequence (str or list): ARPAbet phonemes as string (space-separated)
                                          or list of phonemes
        
        Returns:
            str: IPA transcription
            
        Example:
            >>> converter = ARPAbetIPAConverter()
            >>> converter.arpabet_to_ipa_convert("HH EH 1 L OW 0")
            'hɛˈloʊ'
        """
        if isinstance(arpabet_sequence, str):
            phonemes = arpabet_sequence.strip().split()
        else:
            phonemes = arpabet_sequence
        
        ipa_result = []
        i = 0
        
        while i < len(phonemes):
            if not phonemes[i]:
                i += 1
                continue
                
            phoneme = phonemes[i]
            stress = ''
            
            # Check if next token is a stress marker
            if i + 1 < len(phonemes) and phonemes[i + 1] in self.stress_markers:
                stress_num = phonemes[i + 1]
                stress = self.stress_markers.get(stress_num, '')
                i += 1  # Skip the stress marker in next iteration
            
            # Convert base phoneme
            if phoneme in self.arpabet_to_ipa:
                ipa_symbol = self.arpabet_to_ipa[phoneme]
                
                # Add stress marker before the vowel
                if stress and self._is_vowel(phoneme):
                    ipa_result.append(stress + ipa_symbol)
                else:
                    ipa_result.append(ipa_symbol)
            else:
                # If phoneme not found, keep original
                ipa_result.append(phoneme)
            
            i += 1
        
        return ''.join(ipa_result)
    
    def ipa_to_arpabet_convert(self, ipa_sequence):
        """
        Convert IPA sequence to ARPAbet with separated stress markers.
        
        Args:
            ipa_sequence (str): IPA transcription
        
        Returns:
            list: List of ARPAbet phonemes and stress markers separated
            
        Example:
            >>> converter = ARPAbetIPAConverter()
            >>> converter.ipa_to_arpabet_convert('hɛˈloʊ')
            ['HH', 'EH', '1', 'L', 'OW', '0']
        """
        arpabet_result = []
        i = 0
        
        while i < len(ipa_sequence):
            char = ipa_sequence[i]
            
            # Handle stress markers
            stress_level = None
            if char in self.reverse_stress:
                stress_level = self.reverse_stress[char]
                i += 1
                if i >= len(ipa_sequence):
                    break
                char = ipa_sequence[i]
            
            # Handle multi-character IPA symbols
            found = False
            
            # Check for two-character combinations first
            if i + 1 < len(ipa_sequence):
                two_char = char + ipa_sequence[i + 1]
                if two_char in self.ipa_to_arpabet:
                    phoneme = self.ipa_to_arpabet[two_char]
                    arpabet_result.append(phoneme)
                    if stress_level is not None and self._is_vowel_arpabet(phoneme):
                        arpabet_result.append(stress_level)
                    elif self._is_vowel_arpabet(phoneme):
                        arpabet_result.append('0')  # Add default stress marker to vowels
                    i += 2
                    found = True
            
            # Check for single character
            if not found and char in self.ipa_to_arpabet:
                phoneme = self.ipa_to_arpabet[char]
                arpabet_result.append(phoneme)
                if stress_level is not None and self._is_vowel_arpabet(phoneme):
                    arpabet_result.append(stress_level)
                elif self._is_vowel_arpabet(phoneme):
                    arpabet_result.append('0')  # Add default stress marker to vowels
                i += 1
                found = True
            
            if not found:
                # Skip unknown characters
                i += 1
        
        return arpabet_result
    
    def _is_vowel(self, arpabet_phoneme):
        """Check if ARPAbet phoneme is a vowel."""
        vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 
                 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
        return arpabet_phoneme in vowels
    
    def _is_vowel_arpabet(self, arpabet_phoneme):
        """Check if ARPAbet phoneme (without stress) is a vowel."""
        return self._is_vowel(arpabet_phoneme)
    
    def arpabet_list_to_ipa(self, arpabet_list):
        """
        Convert list of ARPAbet phonemes (with separated stress) to IPA.
        
        Args:
            arpabet_list (list): List of ARPAbet phonemes and stress markers
        
        Returns:
            str: IPA transcription
        """
        return self.arpabet_to_ipa_convert(arpabet_list)
    
    def get_phoneme_info(self, phoneme, notation='arpabet'):
        """
        Get information about a phoneme.
        
        Args:
            phoneme (str): The phoneme to look up
            notation (str): 'arpabet' or 'ipa'
        
        Returns:
            dict: Information about the phoneme
        """
        if notation.lower() == 'arpabet':
            if phoneme in self.arpabet_to_ipa:
                return {
                    'arpabet': phoneme,
                    'ipa': self.arpabet_to_ipa[phoneme],
                    'type': 'vowel' if self._is_vowel(phoneme) else 'consonant'
                }
        elif notation.lower() == 'ipa':
            if phoneme in self.ipa_to_arpabet:
                arpabet = self.ipa_to_arpabet[phoneme]
                return {
                    'ipa': phoneme,
                    'arpabet': arpabet,
                    'type': 'vowel' if self._is_vowel(arpabet) else 'consonant'
                }
        
        return None
    
    def list_all_mappings(self):
        """Return all phoneme mappings."""
        return dict(self.arpabet_to_ipa)
    
    def validate_arpabet(self, arpabet_sequence):
        """
        Validate ARPAbet sequence.
        
        Args:
            arpabet_sequence (str or list): ARPAbet phonemes
        
        Returns:
            tuple: (is_valid, invalid_phonemes)
        """
        if isinstance(arpabet_sequence, str):
            phonemes = arpabet_sequence.strip().split()
        else:
            phonemes = arpabet_sequence
        
        invalid = []
        i = 0
        while i < len(phonemes):
            phoneme = phonemes[i]
            
            # Skip stress markers
            if phoneme in ['0', '1', '2']:
                i += 1
                continue
            
            if phoneme not in self.arpabet_to_ipa:
                invalid.append(phoneme)
            
            i += 1
        
        return len(invalid) == 0, invalid


def main():
    """Demonstration of the ARPAbet-IPA converter."""
    converter = ARPAbetIPAConverter()
    
    print("ARPAbet to IPA Converter Demo")
    print("=" * 40)
    
    # Test cases with separated stress markers
    test_cases = [
        ['HH', 'EH', '1', 'L', 'OW', '0'],        # hello
        ['W', 'ER', '1', 'L', 'D'],               # world
        ['P', 'R', 'OW', '0', 'G', 'R', 'AE', '2', 'M'],   # program
        ['K', 'AE', '1', 'T'],                    # cat
        ['TH', 'IH', '1', 'NG', 'K'],             # think
        ['JH', 'AH', '1', 'JH'],                  # judge
        ['CH', 'EH', '1', 'R', 'IY', '0'],        # cherry
    ]
    
    print("\nARPAbet to IPA conversions:")
    for arpabet in test_cases:
        ipa = converter.arpabet_to_ipa_convert(arpabet)
        print(f"{' '.join(arpabet):25} → {ipa}")
    
    print("\nIPA to ARPAbet conversions:")
    for arpabet in test_cases:
        ipa = converter.arpabet_to_ipa_convert(arpabet)
        back_to_arpabet = converter.ipa_to_arpabet_convert(ipa)
        print(f"{ipa:15} → {' '.join(back_to_arpabet)}")
    
    print("\nPhoneme information:")
    sample_phonemes = ['AA', 'CH', 'NG', 'EY']
    for phoneme in sample_phonemes:
        info = converter.get_phoneme_info(phoneme)
        if info:
            print(f"{phoneme}: {info}")
    
    print(f"\nTotal mappings: {len(converter.list_all_mappings())}")


if __name__ == "__main__":
    main()