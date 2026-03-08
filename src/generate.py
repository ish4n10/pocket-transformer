import torch
import tiktoken
from layers import PocketTransformer, PocketConfig


cfg        = PocketConfig()
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/step_10000.pt"

enc = tiktoken.get_encoding("gpt2")


def load_model():
    model = PocketTransformer(cfg).to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint — step {ckpt['step']} | loss {ckpt['loss']:.4f}")
    return model


@torch.no_grad()
def generate(
    model,
    prompt:      str,
    max_tokens:  int   = 200,
    temperature: float = 0.8,
    top_k:       int   = 40,
) -> str:

    tokens = enc.encode_ordinary(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, seq)

    for _ in range(max_tokens):

        x_crop = x[:, -cfg.seq_len:]

        logits = model(x_crop)         
        logits = logits[:, -1, :]    

        logits = logits / temperature

        if top_k is not None:
            values, _  = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_val, float('-inf'))

        # sample
        probs= torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # append
        x = torch.cat([x, next_token], dim=1)

        # stop at end of text token
        if next_token.item() == enc.eot_token:
            break

    return enc.decode(x[0].tolist())


if __name__ == "__main__":
    model = load_model()

    prompts = [
        "The meaning of life is",
        "In the beginning",
        "The best way to learn machine learning is",
    ]

    for prompt in prompts:
        print(f"\nprompt: {prompt}")
        print(generate(model, prompt))
        print()