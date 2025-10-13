# it solve a basic problem of handling unknown token which is not seen during training
# TODO: Minbpe

# subword tokenization algorithm that works by repeatedly finding most frequent pair of adjacent symbols in text


from typing import Dict, List, Tuple, DefaultDict, Set, Optional
from collections import defaultdict
import random 


class BPE:
    """Byte Pair Encoding (BPE) tokenizer implementation.
    
    This class implements the BPE algorithm for subword tokenization, which is commonly used in
    modern NLP models. It works by iteratively merging the most frequent pairs of characters or
    character sequences in the training corpus.
    
    Args:
        vocab_size (int): The desired size of the vocabulary.
        corpus (List[str]): The training corpus as a list of strings.
        
    Attributes:
        tokenizer: HuggingFace tokenizer for pre-tokenization.
        corpus: The processed training corpus.
        vocab_size: Target vocabulary size.
        word_freqs: Dictionary of word frequencies.
        vocab: List of vocabulary tokens.
        splits: Dictionary mapping words to their character splits.
        pair_freqs: Frequencies of character pairs.
        merges: Dictionary of learned merge operations.
    """
    
    def __init__(self, vocab_size: int, corpus: List[str]) -> None:
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("Vocabulary size must be a positive integer")
            
        from transformers import AutoTokenizer
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
            
        self.corpus = [text.lower() for text in corpus]
        self.vocab_size = vocab_size
        self.word_freqs: Dict[str, int] = self._frequency_count()
        self.vocab: List[str] = ["<|text_end|>"] + self._character_vocabulary()
        self.vocab_to_id: Dict[str, int] = {}
        self.id_to_vocab: Dict[int, str] = {}
        self.splits: Dict[str, List[str]] = {word: list(word) for word in self.word_freqs.keys()}
        self.merges: Dict[Tuple[str, str], str] = {}
        self._build_vocabulary()

    def _frequency_count(self) -> Dict[str, int]:
        """Count word frequencies in the corpus.
        
        Returns:
            Dict[str, int]: A dictionary mapping words to their frequencies.
        """
        word_freqs: Dict[str, int] = defaultdict(int)
        for sentence in self.corpus:
            try:
                words_offset = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
                new_words = [word for word, _ in words_offset]
                for word in new_words:
                    word_freqs[word] += 1
            except Exception as e:
                print(f"Warning: Failed to process sentence: {sentence[:50]}... Error: {str(e)}")
        return dict(word_freqs)
    
    def _character_vocabulary(self) -> List[str]:
        """Extract and sort unique characters from the vocabulary.
        
        Returns:
            List[str]: Sorted list of unique characters in the vocabulary.
        """
        alphabets: Set[str] = set()
        for word in self.word_freqs:
            alphabets.update(word)
        return sorted(alphabets)
    
    def _compute_pair_frequencies(self) -> DefaultDict[Tuple[str, str], int]:
        """Compute frequencies of adjacent character pairs in the vocabulary.
        
        Returns:
            DefaultDict[Tuple[str, str], int]: Frequency count of each character pair.
        """
        pair_freqs: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) < 2:
                continue
                
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
                
        return pair_freqs
    
    def _merge_pair(self, a: str, b: str) -> Dict[str, List[str]]:
        """Merge the most frequent pair of tokens in the vocabulary.
        
        Args:
            a (str): First token in the pair to merge.
            b (str): Second token in the pair to merge.
            
        Returns:
            Dict[str, List[str]]: Updated word splits after merging.
        """
        merged = a + b
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) < 2:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [merged] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
            
        return self.splits
    
    def _build_vocabulary(self) -> None:
        """Build the vocabulary by iteratively merging the most frequent pairs.
        
        Raises:
            RuntimeError: If no more merges are possible before reaching vocab_size.
        """
        iterations = 0
        max_iterations = self.vocab_size * 2  # Prevent infinite loops
        
        while len(self.vocab) < self.vocab_size and iterations < max_iterations:
            iterations += 1
            pair_freqs = self._compute_pair_frequencies()
            
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            merged = ''.join(best_pair)
            
            if best_pair in self.merges:
                break
                
            self.merges[best_pair] = merged
            self._merge_pair(*best_pair)
            self.vocab.append(merged)
        
        if len(self.vocab) < self.vocab_size:
            print(f"Warning: Could only build vocabulary of size {len(self.vocab)} "
                 f"(requested {self.vocab_size})")
        
        self.vocab_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_vocab = {i: word for word, i in self.vocab_to_id.items()}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text using the learned BPE vocabulary.
        
        Args:
            text (str): The input text to tokenize.
            
        Returns:
            List[str]: List of tokens.
            
        Raises:
            ValueError: If the input text is empty or invalid.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")
            
        text = text.lower()
        
        try:
            pre_tokenize_result = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        except Exception as e:
            raise RuntimeError(f"Failed to pre-tokenize text: {str(e)}")
            
        pre_tokenized_text = [word for word, _ in pre_tokenize_result]
        splits = [[c for c in word] for word in pre_tokenized_text]
        
        # Apply all learned merge operations
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if (split[i], split[i + 1]) == pair:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
        
        # Flatten the list of lists and handle unknown tokens
        tokens = []
        for word_tokens in splits:
            if not word_tokens:  # Skip empty tokens
                continue
                
            # Try to merge tokens according to learned merges
            i = 0
            while i < len(word_tokens):
                token = word_tokens[i]
                
                # Check for multi-character tokens in vocabulary
                if token in self.vocab:
                    tokens.append(token)
                    i += 1
                    continue
                    
                # If not in vocab, try to merge with next token
                if i < len(word_tokens) - 1:
                    merged = token + word_tokens[i + 1]
                    if (token, word_tokens[i + 1]) in self.merges:
                        tokens.append(merged)
                        i += 2
                        continue
                        
                # If no merge possible, use character-level tokens
                tokens.append(token)
                i += 1
                
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode the input text using the learned BPE vocabulary.
        
        Args:
            text (str): The input text to encode.
            
        Returns:
            List[int]: List of token IDs.
            
        Raises:
            ValueError: If the input text is empty or invalid.
        """
        tokens = self.tokenize(text)
        return [self.vocab_to_id[token] for token in tokens if token in self.vocab_to_id]
    
    def decode(self, ids: List[int]) -> str:
        if not ids:
            raise ValueError("Input token IDs cannot be empty")
            
        tokens = []
        for id_ in ids:
            if id_ not in self.id_to_vocab:
                raise ValueError(f"Invalid token ID: {id_}")
            token = self.id_to_vocab[id_]
            # Handle special tokens or add spaces as needed
            tokens.append(token)
            
        # This is a simple join - might need adjustment based on your tokenization
        return "".join(tokens).replace("Ġ", " ").replace("</w>", " ")

        
    def __call__(self, text: str) -> List[str]:
        """Make the tokenizer callable. Equivalent to tokenize().
        
        Args:
            text (str): The input text to tokenize.
            
        Returns:
            List[str]: List of tokens.
        """
        return self.tokenize(text)
        
    def get_vocab(self) -> List[str]:
        """Get the current vocabulary.
        
        Returns:
            List[str]: The current vocabulary.
        """
        return self.vocab.copy()
        
    def get_merges(self) -> Dict[Tuple[str, str], str]:
        """Get the learned merge operations.
        
        Returns:
            Dict[Tuple[str, str], str]: Dictionary mapping character pairs to their merged form.
        """
        return self.merges.copy()

# Example usage
# if __name__ == "__main__":
#     corpus = [
#         "Here I am, in a journey to find a true fit for my skills.",
#         "The journey may not be easy but it will be worth it.",
#         "Hope you find it interesting."
#     ]
    
#     # Initialize BPE with a small vocab size for demonstration
#     bpe = BPE(vocab_size=50, corpus=corpus)
    
#     # Test tokenization
#     test_text = "This is a test of the BPE tokenizer."
#     tokens = bpe.tokenize(test_text)
#     print(f"Original: {test_text}")
#     print(f"Tokens: {tokens}")
#     print(f"Vocabulary size: {len(bpe.get_vocab())}")
#     print(f"Learned merges: {bpe.get_merges()}")


