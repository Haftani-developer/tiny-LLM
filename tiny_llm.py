"""
یک LLM کوچک و ساده (GPT سطح کاراکتر) با PyTorch

نصب:      pip install torch
آموزش:    python tiny_llm.py train
تولید متن: python tiny_llm.py chat "شروع متن"

فایل data.txt (متن آموزشی) باید کنار این فایل باشد.
هرچه متن بیشتر و یک‌دست‌تر باشد (مثلاً چند صد کیلوبایت)، خروجی بهتر می‌شود.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- تنظیمات (برای سبک‌تر شدن، اعداد را کم کنید) ----------
block_size = 128   # طول زمینه (تعداد کاراکتر قابل دیدن)
batch_size = 32
n_embd = 128       # اندازه بردار هر کاراکتر
n_head = 4
n_layer = 4
dropout = 0.1
lr = 3e-4
steps = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# ---------- مدل ----------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new=300, temperature=0.8):
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ---------- آموزش ----------
def train():
    text = open("data.txt", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = TinyGPT(len(chars)).to(device)
    print(f"پارامترها: {sum(p.numel() for p in model.parameters()):,} | دستگاه: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps + 1):
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(model(*get_batch("val"))[1].item() for _ in range(10)) / 10
            model.train()
            print(f"گام {step}: خطای اعتبارسنجی = {vl:.3f}")
        _, loss = model(*get_batch("train"))
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.save({"model": model.state_dict(), "chars": chars}, "model.pt")
    print("مدل در model.pt ذخیره شد.")


# ---------- تولید متن ----------
def chat(prompt):
    ckpt = torch.load("model.pt", map_location=device)
    chars = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    model = TinyGPT(len(chars)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    out = model.generate(torch.tensor([ids], device=device))[0].tolist()
    print("".join(chars[i] for i in out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat(sys.argv[2] if len(sys.argv) > 2 else "\n")
    else:
        print(__doc__)
