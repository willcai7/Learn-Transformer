import regex as re
from collections import defaultdict
from cs336_basics.src.utils.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN
from tqdm import tqdm

class bilist:
    def __init__(self, val, id, previous=None, next=None):
        self.val = val 
        self.id = id 
        self.prev = previous
        self.next = next 
    
    def merge(self):
        if self.next is None:
            raise ValueError("Cannot merge last element")
        else:
            self.val = self.val + self.next.val
            self.next = self.next.next
            if self.next:
                self.next.prev = self
                

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    """ 
    Train a BPE model on a text file.
    
    Args:

    input_path: str
        The path to the input text file.
    vocab_size: int
        The size of the vocabulary.
    special_tokens: list[str]
        A list of special tokens.
    
    Returns:
    vocab: dict[int, bytes]
        A dictionary mapping token ids to tokens.
    merges: list[tuple[bytes, bytes]]
        A list of merges.

    """

    with open(input_path, "r") as file:
        text = file.read()
    prim_tokens = re.findall(GPT2_PRETOKENIZER_PATTERN, text) # split text into tokens

    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {i: bytes([i]) for i in range(256)}
    merges = [] # list of merges
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    

    id_to_token = {} # id to token dictionary
    token_to_id = {} # token to id dictionary
    frequency_tokens = {} # frequency of tokens table
    frequency_pairs = {}  # frequency of pairs table
    positions_pairs = defaultdict(list) # positions of pairs table, position is a bilist object


    # create token to id, id to token dictionaries, frequency_tokens table, and positions_pairs table
    for prim_token in prim_tokens:
        chars = [bytes([b]) for b in prim_token.encode("utf-8")] # get bytes of token
        if prim_token in token_to_id:
            frequency_tokens[token_to_id[prim_token]] += 1 # increment frequency of token
        else:
            id = len(token_to_id) # get new id
            token_to_id[prim_token] = id # add token to id dictionary
            id_to_token[id] = prim_token # add id to token dictionary
            frequency_tokens[id] = 1 # set frequency of token to 1

            bi_char = bilist(chars[0], id) # create first bilist object
            for char in chars[1:]:
                new_bi_char = bilist(char, id, bi_char) # create new bilist object
                bi_char.next = new_bi_char # link new bilist object to previous one
                pairs = (bi_char.val, new_bi_char.val) # create pair
                positions_pairs[pairs].append(bi_char) # add position to pair
                bi_char = new_bi_char # move to next bilist object

    # update the frequency_pairs table
    for pairs, positions in positions_pairs.items(): 
        for position in positions:
            frequency_pairs[pairs] = frequency_pairs.get(pairs, 0) + frequency_tokens[position.id]  # increment frequency of pair

    # print(frequency_pairs)

    # find the pair with the highest frequency and marge 
    while len(vocab) < vocab_size:
        
        max_pair = max(frequency_pairs, key=lambda x:(frequency_pairs[x], x)) # get pair with highest frequency
        # print(max_pair) # print pair
        merges.append(max_pair) # add pair to merges list
        new_char = max_pair[0] + max_pair[1] # create new character
        vocab[len(vocab)] = new_char # add new character to vocab
        positions = positions_pairs[max_pair] # get positions of pair
        
        for position in positions: 

            val1 = position.val # get value of position
            val2 = position.next.val # get value of next position
            temp_pos =  position.next # get next position
            position.merge() # merge position with next position

            if position.prev: # if there is a previous position
                prev_pair = (position.prev.val, val1)
                frequency_pairs[prev_pair] -= frequency_tokens[position.id] # decrement frequency of pair   
                positions_pairs[prev_pair].remove(position.prev) # remove previous position from pair
                new_prev_pair = (position.prev.val, position.val) # create new pair
                frequency_pairs[new_prev_pair] = frequency_pairs.get(new_prev_pair, 0) + frequency_tokens[position.id] # increment frequency of new pair
                positions_pairs[new_prev_pair].append(position.prev) # add previous position to new pair

            if position.next: # if there is a next position
                next_pair = (val2, position.next.val)
                frequency_pairs[next_pair] -= frequency_tokens[position.id] # decrement frequency of pair
                positions_pairs[next_pair].remove(temp_pos) # remove next position from pair
                new_next_pair = (position.val, position.next.val) # create new pair
                frequency_pairs[new_next_pair] = frequency_pairs.get(new_next_pair, 0) + frequency_tokens[position.id] # increment frequency of new pair
                positions_pairs[new_next_pair].append(position) # add position to new pair  

        del positions_pairs[max_pair] # delete pair from positions_pairs
        del frequency_pairs[max_pair] # delete pair from frequency_pairs
    
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab:dict[int,bytes], merges:list[tuple[(bytes,bytes)]], special_tokens=None):
        # load vocab 
        self.vocab = {}
        self.vocab['int_to_byte'] = vocab 
        self.vocab['byte_to_int'] = {v:k for k,v in vocab.items()} 

        # load merges 
        self.merges = {}
        for a, b in merges:
            id_pair = (self.vocab['byte_to_int'][a], self.vocab['byte_to_int'][b])
            self.merges[id_pair] = self.vocab['byte_to_int'][a+b]
        
        # load special tokens
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab['byte_to_int']:
                    self.vocab['int_to_byte'][len(self.vocab['int_to_byte'])] = token_bytes
                    self.vocab['byte_to_int'][token_bytes] = len(self.vocab['int_to_byte']) 
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_bytes]
                else:
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_bytes]

    
    @classmethod
    def from_files(cls, vocab_file_path, merges_file_path, special_tokens=None):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_file_path, merges_file_path)
        return cls(vocab, merges, special_tokens)

    def encode(self, text:str, progress_bar:bool=False)-> list[int]:
        """
        Encode a text into token ids.
        """
        if self.special_tokens:
            chunk_pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            split_chunks = re.split(chunk_pattern, text)
        else:
            split_chunks = [text]
        
        ids = [] 
        for chunk in tqdm(split_chunks, disable=not progress_bar, desc="Encoding {len(split_chunks)} chunks"):
            new_ids = self.encode_chunk(chunk)
            ids.extend(new_ids)
        return ids

    def encode_chunk(self, chunk:str)-> list[int]:
        """
        Encode a chunk of text into token ids.
        """
        if chunk in self.special_tokens:
            return [self.special_tokens[chunk]]
        else:
            tokens = re.findall(GPT2_PRETOKENIZER_PATTERN, chunk)
            total_ids = []
            for token in tokens:
                token_bytes = token.encode("utf-8")
                token_ids = [self.vocab['byte_to_int'][bytes([byte])] for byte in token_bytes]
            
                while len(token_ids) > 1:
                    pairs = [(token_ids[i], token_ids[i+1]) for i in range(len(token_ids)-1)] # get all pairs of token_ids
                    high_priority_pair = min(pairs, key=lambda x: self.merges.get(x, float('inf'))) # get the pair with the highest merge priority

                    # We need to merge all instances of high_priority_pair in token_ids
                    if high_priority_pair in self.merges: # if the pair is in merges, we merge
                        new_token_id = self.merges[high_priority_pair]
                        new_token_ids = []
                        ind = 0
                        while ind < len(token_ids): 
                            if ind < len(token_ids) - 1 and (token_ids[ind], token_ids[ind+1]) == high_priority_pair:
                                new_token_ids.append(new_token_id)
                                ind += 2
                            else:
                                new_token_ids.append(token_ids[ind])
                                ind += 1
                        token_ids = new_token_ids
                    else: # if the pair is not in merges, we break
                        break
                total_ids.extend(token_ids)
            return total_ids # return the token ids
                


    def encode_iterable(self, texts):
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id


    def decode(self, ids: list[int])-> str:
        text_bytes = b"".join([self.vocab['int_to_byte'][id] for id in ids])
        return text_bytes.decode("utf-8", errors="replace")
    