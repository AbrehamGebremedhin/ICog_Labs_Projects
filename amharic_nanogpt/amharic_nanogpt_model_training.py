import torch
import os
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get "Amharic_corpus_merged_2023-04-16.csv" file from kaggle dataset "amharic-news-corpus-merged" and update the dataset_path to the CSV file
dataset_path = ""

df = pd.read_csv(dataset_path)

output_file = 'output.txt'

with open(output_file, 'w') as file:
    for index, row in df.iterrows():
        value = row['article']
        file.write(value + '\n')

print(f"Values from column 'article' have been written to {output_file}.")

with open("output.txt", "r", encoding="utf-8") as file:
    text = file.read()

os.remove("output.txt")

print(f"Initial file's total characters: {len(text)}")


def remove_unicode_and_english_words(input_string):
    characters = ['\x08', '\t', '\n', '!', '፨', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '\x7f', '\x8d', '\x90', '\x9d', '\xa0', '¡', '£', '¤', '¥', '¦', '§', '¨', '©', '«', '¬', '\xad', '®', '°', '²', '´', 'µ', '·', '¸', '»', '¼', '½', '¾', 'À', 'Â', 'Ã', 'Æ', 'È', 'É', 'Î', 'Ö', 'Ú', 'Û', 'Ý', 'à', 'á', 'â', 'ã', 'ä',
                  'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ó', 'ô', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ā', 'ć', 'č', 'Ė', 'ğ', 'ő', 'š', 'ʻ', 'ʼ', 'ʾ', 'ʿ', '˜', '˝', '˩', '̀', '̂', '̋', '̌', '̎', '̕', '̣', '߹', '᎐', '᎓', '\u2003', '\u200b', '\u200e', '‐', '‑', '–', '—', '―', '‘', '’', '‚', '“', '”', '•', '…', '\u202c', '′', '″', '‹', '›', '₍', '₎', '€', '℃', '℅', '−', '∕', '∙', '∶', '∷', '≪', '≫', '●', '★', '♦', '✿', '❖', '❷', '❸', '❹', '❺', '❻', '➊', '➢', '➤', 'ⵆ', 'ⶄ', 'ⶋ', 'ⶵ', '「', '㙀', '㝕', '\uf024', '\uf058', '\uf06c', '\uf07d', '\uf07e', '\uf09d', '\uf0a7', '\uf0d8', '\ufeff', '（', '）', '�', '🇧', '🇨', '🇩', '🇪', '🇬', '🇭', '🇮', '🇯', '🇲', '🇳', '🇹', '🇺', '🇼', '🇿', '🌻', '🌼', '👇', '👉']
    for character in characters:
        input_string = input_string.replace(character, '')

    print(len(input_string))

    with open("train.txt", 'w') as file:
        file.write(input_string)


remove_unicode_and_english_words(text)


batch_size = 64
block_size = 256
max_iters = 6000
eval_interval = 600
eval_iters = 200
learning_iters = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open("train.txt", "r", encoding="utf-8") as file:
    content = file.read()

print(f"Total characters in the dataset: {len(content)}")

chars = sorted(list(set(content)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


print(chars)
print(f"Number of Unique Characters in the dataset: {vocab_size}")

data = torch.tensor(encode(content), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(0, keepdim=True)
        xvar = x.var(0, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.postion_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.lm_hed = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_hed(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()

m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: training loss {
              losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.ones((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(m, "model.pth")
