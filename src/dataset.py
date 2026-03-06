import torch
import tiktoken
import urllib.request
import os

enc = tiktoken.get_encoding("gpt2")

def prepare_data():
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "data/input.txt"
    )
    with open("data/input.txt", "r") as f:
        text = f.read()
    data  = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
    n     = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    torch.save(train, "data/train.pt")
    torch.save(val,   "data/val.pt")
    return train, val

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([data[i:i+seq_len]     for i in ix])
    y  = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    train, val = prepare_data()
    print(f"train: {len(train):,} val: {len(val):,}")