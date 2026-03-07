import torch
import tiktoken
import pandas as pd
import os

enc = tiktoken.get_encoding("gpt2")

def prepare_data(path: str):
    os.makedirs("data", exist_ok=True)

    if os.path.exists("data/train.pt") and os.path.exists("data/val.pt"):
        print("already prepared, loading...")
        train = torch.load("data/train.pt")
        val   = torch.load("data/val.pt")
        print(f"train: {len(train):,} val: {len(val):,}")
        return train, val

    print("reading parquet...")
    df     = pd.read_parquet(path)
    texts  = df["text"].tolist()
    print(f"documents: {len(texts):,}")

    tokens = []
    for i, text in enumerate(texts):
        tokens.extend(enc.encode_ordinary(text))
        tokens.append(enc.eot_token)
        if i % 10_000 == 0:
            print(f"  {i:,} / {len(texts):,} docs — {len(tokens):,} tokens")

    print(f"total tokens: {len(tokens):,}")

    data  = torch.tensor(tokens, dtype=torch.long)
    n     = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    torch.save(train, "data/train.pt")
    torch.save(val,   "data/val.pt")
    print(f"train: {len(train):,} val: {len(val):,}")
    return train, val

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([data[i:i+seq_len]     for i in ix])
    y  = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    train, val = prepare_data(r"G:\attention-is-all-you-need\train-00000-of-00080.parquet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y   = get_batch(train, batch_size=4, seq_len=256, device=device)
    print(f"x: {x.shape} y: {y.shape}")
    print(f"sample: {enc.decode(x[0][:20].tolist())}")