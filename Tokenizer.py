import re

import torch

_2D_DICT = {'<': 0, '&': 1, 'E': 2, '[MASK]': 3, 'O': 4, '[C-]': 5, '#': 6, 'N': 7, 'C': 8, '[O+]': 9, '=': 10, '[N]': 11, '[O-]': 12, 'F': 13, '[N-]': 14, '[N+]': 15, '1': 16, 'Cl': 17, '(': 18, ')': 19, 'S': 20, '[S-]': 21, 'B': 22, '[Si]': 23, 'P': 24, '[PH2]': 25, 'c': 26, '[nH]': 27, '[Cl+]': 28, 'n': 29, 'o': 30, '[n-]': 31, '[S+]': 32, '[PH]': 33, '[SH]': 34, '[Cl+2]': 35, 's': 36, '2': 37, '[B-]': 38, '[P+]': 39, '[n+]': 40, 'Br': 41, '[NH-]': 42, '[BH2-]': 43, '[Cl+3]': 44, '[Se-]': 45, '[Se]': 46, '[NH3+]': 47, '[SeH]': 48, '[As]': 49, '3': 50, '4': 51, '5': 52, '-': 53, '[Br+2]': 54, '[CH2]': 55, 'I': 56, '[O]': 57, '[P-]': 58, '[CH]': 59, '6': 60, '[I+2]': 61, '[As+]': 62, '[s+]': 63, '[se]': 64, '[AsH]': 65, '[NH+]': 66, '[I+3]': 67, '[BH-]': 68, '[cH-]': 69, '[o+]': 70, '[SH+]': 71, '[NH2+]': 72, '[I+]': 73, '[te]': 74, '[Se+]': 75, '7': 76, '[SH-]': 77, '[Te]': 78, '[C+]': 79, '[c-]': 80, '[CH-]': 81, '[nH+]': 82, '[c+]': 83, '[b-]': 84, '[SiH]': 85, '[SiH2]': 86, '[OH+]': 87, '[BH3-]': 88, '[se+]': 89, '[te+]': 90, '8': 91, 'p': 92, '[SiH-]': 93, '9': 94, '%10': 95, '%11': 96, '%12': 97, '%13': 98, '%14': 99, '%15': 100, '%16': 101, '%17': 102, '%18': 103, '%19': 104, '%20': 105, '%21': 106, '%22': 107, '%23': 108, '%24': 109, '%25': 110, '%26': 111, '%27': 112, '%28': 113}
_GPT_2D_EXTRA_TOKENS = ['%29', '[IH]', '[TeH]', 'b', '[H]', '[SH2]', '[Si-]', '[PH2+]', '[As-]']

def get_gpt_2d_dict():
    vocabulary = dict(_2D_DICT)
    for token in _GPT_2D_EXTRA_TOKENS:
        vocabulary.setdefault(token, len(vocabulary))
    return vocabulary

class _TokenizerBase:
    def __init__(self, pattern, vocabulary):
        self.pattern = re.compile(pattern)
        self.dict = dict(vocabulary)
        self.dict2 = {index: token for token, index in self.dict.items()}

    def tokenize(self, smiles):
        tokens = self.pattern.findall(smiles)
        return tokens if all(token in self.dict for token in tokens) else None

    def sm2index(self, tokens):
        if tokens is None:
            return None
        return [self.dict[token] for token in tokens]

    def index2sm(self, indices):
        values = indices.detach().cpu().tolist() if isinstance(indices, torch.Tensor) else indices
        is_single = not values or not isinstance(values[0], (list, tuple))
        sequences = [values] if is_single else values
        decoded = []
        for sequence in sequences:
            tokens = []
            for index in sequence:
                index = int(index)
                if index == 2:
                    break
                token = self.dict2.get(index)
                if token is not None:
                    tokens.append(token)
            decoded.append(tokens)
        return decoded[0] if is_single else decoded

    def get_dict(self):
        return self.dict

class GPTTokenizer(_TokenizerBase):
    def __init__(self):
        super().__init__(r"(\[[^\]]*\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])", get_gpt_2d_dict())

class BertTokenizer(_TokenizerBase):
    def __init__(self):
        vocabulary = {'<': 0, '&': 1, 'E': 2, '[MASK]': 3, 'C': 4, 'c': 5, '1': 6, '(': 7, '-': 8, '2': 9, 's': 10, 'N': 11, '=': 12, ')': 13, 'n': 14, '[C@H]': 15, 'O': 16, '[C@@H]': 17, 'S': 18, '[nH]': 19, '3': 20, 'Cl': 21, 'Br': 22, '/': 23, '[C@]': 24, 'o': 25, '[C@@]': 26, '4': 27, '5': 28, '6': 29, '7': 30, '[n+]': 31, 'I': 32, 'F': 33, '8': 34, '[O-]': 35, '#': 36, '[N+]': 37, 'P': 38, '9': 39, '%10': 40, '%11': 41, '%12': 42, '%13': 43, '%14': 44, '%15': 45, '%16': 46, '%17': 47, '%18': 48, '%19': 49, '%20': 50, '%21': 51, '%22': 52, '%23': 53, '[Si]': 54, '[N-]': 55, 'B': 56, '[S+]': 57, '[P+]': 58, '[PH]': 59, '[n-]': 60, '[B-]': 61, '[O]': 62, '%24': 63, '%25': 64, '[o+]': 65, '[N@@+]': 66, '[N@+]': 67, '%26': 68, '[se]': 69, '[Se]': 70, '[s+]': 71, '[P@]': 72, '[P@@]': 73, '[te+]': 74, '[se+]': 75, '[Te]': 76, '[S@@+]': 77, '[S@+]': 78, '[BH2-]': 79, '[N@@]': 80, '[S-]': 81, '[C+]': 82, '[P@+]': 83, '[N@]': 84, '[C-]': 85, '[SeH]': 86, '[BH-]': 87, '[SH2]': 88, '[I+]': 89, '[Cl+]': 90, '[Se+]': 91, '[As]': 92, '[SiH2]': 93, '[NH-]': 94, '[As+]': 95, '[te]': 96, '[S@]': 97, '[S@@]': 98, '[c+]': 99, '[P-]': 100, 'p': 101, '[NH+]': 102, 'b': 103, '[Se-]': 104, '[SH]': 105, '%27': 106, '%28': 107, '[O+]': 108, '[nH+]': 109, '[NH2+]': 110, '[TeH]': 111, '[BH3-]': 112, '[SiH]': 113, '[Si-]': 114, '[SH+]': 115, '[Cl+3]': 116, '[Br+2]': 117, '[I+2]': 118, '[I+3]': 119, '[N]': 120, '[Cl+2]': 121, '[IH]': 122, '[B@-]': 123, '[B@@-]': 124, '[cH-]': 125, '[Si@]': 126, '[c-]': 127, '[CH-]': 128, '[SiH-]': 129, '[b-]': 130, '[SH-]': 131, '[OH+]': 132, '[H]': 133, '[PH2]': 134, '%29': 135, '%30': 136, '%31': 137, '%32': 138, '[PH2+]': 139, '[P@@+]': 140, '[NH3+]': 141, '[CH]': 142, '[CH2]': 143, '[As-]': 144, '[AsH]': 145,'[SH2+]':146,'[FH+]':147,'[IH2]':148,'[NH]':149,'[CH2-]':150,'[N@@H+]': 151, '[N@H+]': 152,'[CH+]':153,'[CH2+]':154,'[cH+]':155}
        super().__init__(r"(\[[^\]]*\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])", vocabulary)
