import torch
import torch.nn.functional as F
from layers import PocketTransformer, PocketConfig
from dataset import prepare_data, get_batch
import math
import os


cfg = PocketConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_STEPS = 5000
EVAL_EVERY = 200
SAVE_EVERY = 1000
LR = 3e-4
MIN_LR = 3e-5
WARMUP = 200
GRAD_CLIP = 1.0
DATA_PATH = r""


def get_lr(step: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LR - MIN_LR) * cosine


@torch.no_grad()
def eval_loss(model, val_data, eval_steps=50):
    model.eval()
    losses = [
        F.cross_entropy(
            model(x := get_batch(val_data, BATCH_SIZE, cfg.seq_len, DEVICE)[0]).view(-1, cfg.vocab_size),
            get_batch(val_data, BATCH_SIZE, cfg.seq_len, DEVICE)[1].view(-1)
        ).item()
        for _ in range(eval_steps)
    ]
    model.train()
    return sum(losses) / len(losses)

@torch.no_grad()
def eval_loss(model, val_data, eval_steps=50):
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = get_batch(val_data, BATCH_SIZE, cfg.seq_len, DEVICE)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    os.makedirs("checkpoints", exist_ok=True)

    train_data, val_data = prepare_data(DATA_PATH)

    model     = PocketTransformer(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    print(f"device:     {DEVICE}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"train tokens: {len(train_data):,}")
    print(f"steps:      {MAX_STEPS}\n")

    model.train()
    for step in range(MAX_STEPS):

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y   = get_batch(train_data, BATCH_SIZE, cfg.seq_len, DEVICE)
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | grad {grad_norm:.2f}")

        if step % EVAL_EVERY == 0 and step > 0:
            val = eval_loss(model, val_data)
            print(f"\n{'─'*50}")
            print(f"step {step:5d} | train {loss.item():.4f} | val {val:.4f}")
            print(f"{'─'*50}\n")

        # checkpoint every 1000 steps
        if step % SAVE_EVERY == 0 and step > 0:
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      loss.item(),
            }, f"checkpoints/step_{step}.pt")
            print(f"checkpoint saved step_{step}.pt")

    torch.save({
        "step":      MAX_STEPS,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss":      loss.item(),
    }, "checkpoints/final.pt")
    print("done checkpoints/final.pt")

if __name__ == "__main__":
    train()
