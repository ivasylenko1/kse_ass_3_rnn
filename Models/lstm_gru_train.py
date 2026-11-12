"""
5-fold ensemble of recurrent location taggers that survives a power cut.

Pipelines in PIPELINES run one after the other (default: the bidirectional GRU only).
After every epoch the full training state is saved (atomically), so restarting the same command continues
from the last finished epoch. The best checkpoint of each fold is kept; an older best is deleted when a
better one is written. After every finished fold the ensemble of the folds done so far is written to
submissions/, so a partial night still leaves a usable submission.

    python -u train_ensemble.py            full run (after a crash: run the same command again)
    python -u train_ensemble.py --smoke    quick check of the whole pipeline on a tiny subset (a few minutes)
"""
import argparse, gc, json, math, os, pickle, re, string, time, uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tokenizers import Tokenizer

# ============================ CONFIG ============================
DATA_FILE = "data/splits/long_aug3_all.parquet"
LABELING  = "data/splits/val_labeling_sample.parquet"
TEST_CSV  = "Data/test.csv"
BPE_JSON  = "sensitive_rua_bpe_tokenizer.json"
QWEN_REPO = "Qwen/Qwen3-0.6B"        # embedding table width 1024 (Qwen3-1.7B: 2048); EMB follows the table

RNN       = "GRU"                                    # "GRU" or "LSTM"  (command line: --rnn LSTM)
FOLDS     = [0]                                      # fold numbers, or "all" = train on every dev row (command line: --folds all)
HID, LAYERS, DROPOUT = 256, 2, 0.2
EPOCHS, PATIENCE, BATCH = 25, 5, 64                  # 30 epochs: about 20% longer; change before the first run, not mid-run
LR, WEIGHT_DECAY, WARMUP, ETA_MIN, CLIP = 2e-3, 1e-2, 0.03, 1e-6, 1.0
FREEZE_EMB_EPOCHS = 1                                # pretrained embedding frozen for the first epoch
VAL_N     = 20_000                                   # validation posts per fold (early stopping + threshold)
TRAIN_N   = None                                     # None = all training posts of the fold
THRS      = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]
SEED      = 42
OUT       = Path("runs_aug3")                         # command line: --out runs_full
USE_WANDB, WANDB_ENTITY, WANDB_PROJECT = True, "aboichuk06-kyiv-school-of-economics", "rnn"

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
ap.add_argument("--rnn", choices=["GRU", "LSTM"])
ap.add_argument("--folds", nargs="+", help='e.g. --folds 0 1 2   or   --folds all')
ap.add_argument("--out")
ap.add_argument("--epochs", type=int)
args = ap.parse_args()
if args.rnn: RNN = args.rnn
if args.folds: FOLDS = [f if f == "all" else int(f) for f in args.folds]
if args.out: OUT = Path(args.out)
if args.epochs: EPOCHS = args.epochs
if args.smoke:
    FOLDS, EPOCHS, VAL_N, TRAIN_N, OUT, USE_WANDB = [0, 1], 2, 500, 3000, Path("runs_smoke"), False
PIPELINES = [("bi" + RNN.lower(), True)]             # (name, bidirectional)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
OUT.mkdir(parents=True, exist_ok=True); Path("submissions").mkdir(exist_ok=True)


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def atomic_save(obj, path, use_torch=True):
    """Write to a temporary file, flush it to disk, then rename: a power cut never leaves a half-written file."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f) if use_torch else pickle.dump(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ============================ DATA ============================
tok = Tokenizer.from_file(BPE_JSON); tok.no_truncation()
PAD, UNK, V = tok.token_to_id("<pad>"), tok.token_to_id("<unk>"), tok.get_vocab_size()

df = pd.read_parquet(DATA_FILE, columns=["text", "loc_markers", "locations", "lang", "split", "fold"])
holdout = df[df["split"] == "holdout"].reset_index(drop=True)
labeling = pd.read_parquet(LABELING)
test = pd.read_csv(TEST_CSV); test["text"] = test["text"].astype(str)


def token_labels(offs, markers):
    """0 = O, 1 = B-LOC (token holds an entity start), 2 = I-LOC; -100 for tokens without characters."""
    y = np.zeros(len(offs), np.int64)
    for a, b in markers:
        inside = (offs[:, 0] < b) & (offs[:, 1] > a)
        y[inside] = 2
        y[inside & (offs[:, 0] <= a)] = 1
    y[offs[:, 1] <= offs[:, 0]] = -100
    return y


def encode(d, labels=True):
    ids, offs, ys = [], [], []
    texts = d["text"].tolist()
    markers = d["loc_markers"].tolist() if labels else None
    for i in range(0, len(texts), 20_000):
        for k, e in enumerate(tok.encode_batch(texts[i:i + 20_000])):
            ii = np.array(e.ids or [UNK], np.int64)
            oo = np.array(e.offsets, np.int64).reshape(-1, 2) if e.ids else np.zeros((1, 2), np.int64)
            ids.append(ii); offs.append(oo)
            if labels: ys.append(token_labels(oo, markers[i + k]))
    return dict(ids=ids, offs=offs, y=ys if labels else None, text=texts)


enc_hold, enc_lab, enc_test = encode(holdout), encode(labeling), encode(test, labels=False)
log(f"data: {len(df):,} rows | holdout {len(holdout):,} | labeling {len(labeling)} | test {len(test)} | device {DEVICE}")


# ============================ MODEL ============================
_INIT = None
def qwen_init():
    """Our BPE vocabulary -> Qwen input-embedding rows (mean over the Qwen tokens of each token's text), centred, std 1.
    Large checkpoints are split into shards: only the shard that holds the embedding table is downloaded."""
    global _INIT
    if _INIT is not None:
        return _INIT
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    key = "model.embed_tokens.weight"
    try:
        index = json.load(open(hf_hub_download(QWEN_REPO, "model.safetensors.index.json"), encoding="utf-8"))
        shard = index["weight_map"][key]
    except Exception:
        shard = "model.safetensors"                     # small models come as one file
    with safe_open(hf_hub_download(QWEN_REPO, shard), framework="pt") as f:
        table = f.get_tensor(key).float().numpy()
    qtok = Tokenizer.from_file(hf_hub_download(QWEN_REPO, "tokenizer.json"))
    emb = table.shape[1]
    texts = [tok.decode([i], skip_special_tokens=False) for i in range(V)]
    special = {PAD, UNK, tok.token_to_id("<mask>")}
    W = np.random.default_rng(SEED).normal(size=(V, emb)).astype(np.float32)   # tokens without a Qwen vector stay random
    have = np.zeros(V, bool)
    ok = [i for i in range(V) if i not in special and texts[i] and "�" not in texts[i]]
    for i, e in zip(ok, qtok.encode_batch([texts[i] for i in ok], add_special_tokens=False)):
        if e.ids:
            W[i] = table[e.ids].mean(0); have[i] = True
    W[have] = (W[have] - W[have].mean(0)) / W[have].std()
    W[PAD] = 0
    del table; gc.collect()
    log(f"{QWEN_REPO}: embedding width {emb}, Qwen vector for {have.mean():.1%} of the vocabulary")
    _INIT = torch.from_numpy(W)
    return _INIT


class Tagger(nn.Module):
    def __init__(self, init, bidir):
        super().__init__()
        emb = init.shape[1]
        self.emb = nn.Embedding(V, emb, padding_idx=PAD); self.emb.weight.data.copy_(init)
        self.drop = nn.Dropout(DROPOUT)
        self.rnn = (nn.GRU if RNN == "GRU" else nn.LSTM)(emb, HID, num_layers=LAYERS, batch_first=True, bidirectional=bidir, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HID * (2 if bidir else 1), 3))

    def forward(self, ids, lens):
        x = self.drop(self.emb(ids))
        y, _ = self.rnn(pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False))
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=ids.size(1))
        return self.head(y)


# ============================ METRIC + DECODING ============================
def hard_processing(text):                                  # competition metric, verbatim (including its quirks)
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def comp_metric(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp, fp, fn, p,  = 0.0, 0.0, 0.0, 0.0
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        y_true_sample = set([hard_processing(s) for s in y_true_sample])
        y_pred_sample = set([hard_processing(s) for s in y_pred_sample])
        tp += len(y_true_sample & y_pred_sample)
        fp += len(y_pred_sample - y_true_sample)
        fn += len(y_true_sample - y_pred_sample)
        p += len(y_true_sample)
    if tp + fp == 0:
        precision = 1.0 if p == 0 else 0.0
    else:
        precision = tp/(tp + fp)
    if tp + fn == 0:
        recall = 1.0 if p == 0 else 0.0
    else:
        recall = tp/(tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 8)


STRIP = set(" \t\r\n,.;:!?…«»\"“”„()[]{}<>/\\|-–—")
def is_word_char(t, i):
    return 0 <= i < len(t) and (t[i].isalnum() or (t[i] in "'’ʼ`" and 0 < i < len(t) - 1 and t[i - 1].isalnum() and t[i + 1].isalnum()))


def decode(text, offs, P, thr):
    """Runs of entity tokens (a new run starts where the tagger says B), stripped of punctuation, expanded to whole words."""
    ent = (P[:, 1] + P[:, 2] >= thr) & (offs[:, 1] > offs[:, 0])
    runs, cur = [], None
    for i in range(len(offs)):
        if ent[i] and cur is not None and not P[i, 1] > P[i, 2]:
            cur[1] = offs[i, 1]
        elif ent[i]:
            if cur: runs.append(cur)
            cur = [offs[i, 0], offs[i, 1]]
        elif cur:
            runs.append(cur); cur = None
    if cur: runs.append(cur)
    out = []
    for a, b in runs:
        while a < b and text[a] in STRIP: a += 1
        while b > a and text[b - 1] in STRIP: b -= 1
        if a == b: continue
        while is_word_char(text, a - 1): a -= 1
        while is_word_char(text, b): b += 1
        if text[a:b] not in out: out.append(text[a:b])
    return out


def predictions(enc, probs, thr):
    return [decode(t, o, P, thr) for t, o, P in zip(enc["text"], enc["offs"], probs)]


def f1(enc, gold, probs, thr):
    return comp_metric([list(g) for g in gold], predictions(enc, probs, thr))


# ============================ BATCHING ============================
def batches(enc, shuffle, rng, bs=BATCH):
    idx = rng.permutation(len(enc["ids"])) if shuffle else np.arange(len(enc["ids"]))
    out = []
    for i in range(0, len(idx), bs * 50):                       # sort inside chunks: similar lengths, little padding
        c = idx[i:i + bs * 50]; c = c[np.argsort([len(enc["ids"][j]) for j in c], kind="stable")]
        out += [c[k:k + bs] for k in range(0, len(c), bs)]
    if shuffle: rng.shuffle(out)
    return out


def collate(enc, b):
    L = [len(enc["ids"][j]) for j in b]; m = max(L)
    x = np.full((len(b), m), PAD, np.int64); y = np.full((len(b), m), -100, np.int64)
    for r, j in enumerate(b):
        x[r, :L[r]] = enc["ids"][j]
        if enc["y"] is not None: y[r, :L[r]] = enc["y"][j]
    return torch.from_numpy(x).to(DEVICE), torch.tensor(L), torch.from_numpy(y).to(DEVICE)


@torch.no_grad()
def predict(model, enc):
    model.eval(); out = [None] * len(enc["ids"])
    for b in batches(enc, False, None, bs=128):
        x, L, _ = collate(enc, b)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=AMP):
            P = torch.softmax(model(x, L).float(), -1).cpu().numpy()
        for r, j in enumerate(b): out[j] = P[r, :L[r]]
    return out


# ============================ ONE FOLD (resumable) ============================
EPOCH_TIMES = []

def train_fold(pipe, bidir, fold, epochs_left_after):
    fdir = OUT / pipe / f"fold{fold}"; fdir.mkdir(parents=True, exist_ok=True)
    for t in fdir.glob("*.tmp"): t.unlink()                     # half-written files from a crash
    resume_path = fdir / "resume.pt"

    full = fold == "all"                       # full data: train on every dev row, watch the holdout, no early stopping
    fnum = 5 if full else fold
    tr = df[df["split"] == "dev"] if full else df[(df["split"] == "dev") & (df["fold"] != fold)]
    if TRAIN_N: tr = tr.sample(n=min(TRAIN_N, len(tr)), random_state=SEED)
    if full:
        va = holdout
    else:
        va = df[(df["split"] == "dev") & (df["fold"] == fold)]
        va = va.sample(n=min(VAL_N, len(va)), random_state=SEED).reset_index(drop=True)
    enc_tr, enc_va = encode(tr), encode(va)

    torch.manual_seed(SEED + fnum)
    model = Tagger(qwen_init(), bidir).to(DEVICE)
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or n.startswith("emb")]
    decay = [p for n, p in model.named_parameters() if not (p.ndim <= 1 or n.startswith("emb"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    n = len(enc_tr["ids"])
    steps = sum(math.ceil(min(BATCH * 50, n - i) / BATCH) for i in range(0, n, BATCH * 50))
    total = EPOCHS * steps; warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
        (ETA_MIN + 0.5 * (LR - ETA_MIN) * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))) / LR)

    state = dict(epoch=-1, best=-1.0, bad=0, best_path=None, wandb_id=None)
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]; del ck
        log(f"[{pipe} fold {fold}] resuming after epoch {state['epoch']} (best val F1 {state['best']:.4f})")
    for p in fdir.glob("best_*.pt"):                            # a best checkpoint the saved state does not know about
        if p.name != state["best_path"]: p.unlink()

    import wandb
    state["wandb_id"] = state["wandb_id"] or uuid.uuid4().hex[:12]     # own id (newer wandb has no util.generate_id)
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=f"{pipe}_{QWEN_REPO.split('/')[-1]}_f{fold}",
                     id=state["wandb_id"], resume="allow", mode="online" if USE_WANDB else "disabled",
                     config=dict(pipeline=pipe, bidirectional=bidir, fold=fold, qwen=QWEN_REPO, data=DATA_FILE,
                                 hid=HID, layers=LAYERS, lr=LR, epochs=EPOCHS, patience=PATIENCE))

    for ep in range(state["epoch"] + 1, EPOCHS):
        if state["bad"] >= PATIENCE and not full:
            break
        t0 = time.time()
        torch.manual_seed(SEED + 1000 * fnum + ep)                # per-epoch seeds: a resumed epoch is the same epoch
        rng = np.random.default_rng(SEED + 1000 * fnum + ep)
        model.emb.weight.requires_grad_(ep >= FREEZE_EMB_EPOCHS)
        model.train(); tot = 0.0; nb = 0
        for b in batches(enc_tr, True, rng):
            x, L, y = collate(enc_tr, b)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=AMP):
                logits = model(x, L)
            loss = nn.functional.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        score = f1(enc_va, va["locations"], predict(model, enc_va), 0.5)

        old = None
        if score > state["best"]:
            name = f"best_ep{ep:02d}_f1_{score:.4f}.pt"
            atomic_save(model.state_dict(), fdir / name)          # 1. the new best
            old = state["best_path"]
            state.update(best=score, bad=0, best_path=name)
        else:
            state["bad"] += 1
        state["epoch"] = ep
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=state), resume_path)  # 2. resume point
        if old: (fdir / old).unlink(missing_ok=True)              # 3. only now drop the previous best

        EPOCH_TIMES.append(time.time() - t0)
        left = (EPOCHS - 1 - ep) + epochs_left_after
        run.log({"epoch": ep, "train/loss": tot / nb, "val/f1": score})
        log(f"[{pipe} fold {fold}] ep {ep:2d} | train {tot / nb:.4f} | val F1 {score:.4f} | best {state['best']:.4f} "
            f"| {EPOCH_TIMES[-1] / 60:.1f} min | at most {left * np.mean(EPOCH_TIMES) / 3600:.1f} h left in total")
        if state["bad"] >= PATIENCE and not full:
            log(f"[{pipe} fold {fold}] early stopping")

    # ---- finish the fold: best weights -> threshold -> probabilities ----
    model.load_state_dict(torch.load(fdir / state["best_path"], map_location=DEVICE, weights_only=False))
    pv = predict(model, enc_va)
    sweep = {t: f1(enc_va, va["locations"], pv, t) for t in THRS}
    thr = max(sweep, key=sweep.get)
    half = lambda ps: [p.astype(np.float16) for p in ps]
    res = dict(thr=thr, val_f1=sweep[thr], best_epoch=int(state["best_path"][7:9]),
               holdout=half(predict(model, enc_hold)), labeling=half(predict(model, enc_lab)), test=half(predict(model, enc_test)))
    atomic_save(res, OUT / pipe / f"fold{fold}.pkl", use_torch=False)
    resume_path.unlink(missing_ok=True)                           # the fold is done; its best checkpoint stays
    log(f"[{pipe} fold {fold}] done: val F1 {sweep[thr]:.4f} at thr {thr} (0.5 -> {sweep[0.5]:.4f})")
    run.summary.update({"val/f1_best": sweep[thr], "val/thr": thr}); run.finish()
    del model, opt, enc_tr, enc_va; gc.collect(); torch.cuda.empty_cache()


# ============================ ENSEMBLE ============================
def ensemble(name, files):
    """Average the token probabilities of the given fold files, decode once, write a submission."""
    res = [pickle.load(open(f, "rb")) for f in files]
    thr = float(np.median([r["thr"] for r in res]))
    mean = lambda key: [np.mean([r[key][i].astype(np.float32) for r in res], 0) for i in range(len(res[0][key]))]
    ens = {k: mean(k) for k in ("holdout", "labeling", "test")}
    rows = [dict(model=Path(f).parent.name + "/" + Path(f).stem, val=r["val_f1"], thr=r["thr"],
                 holdout=f1(enc_hold, holdout["locations"], r["holdout"], thr),
                 labeling=f1(enc_lab, labeling["locations"], r["labeling"], thr)) for f, r in zip(files, res)]
    rows.append(dict(model="ENSEMBLE", val=np.nan, thr=thr, holdout=f1(enc_hold, holdout["locations"], ens["holdout"], thr),
                     labeling=f1(enc_lab, labeling["locations"], ens["labeling"], thr)))
    pred = predictions(enc_test, ens["test"], thr)
    out = f"submissions/ens_{name}.csv"
    pd.DataFrame({"text_id": test["text_id"], "locations": [json.dumps(p, ensure_ascii=False) for p in pred]}).to_csv(out, index=False)
    log(f"\n== ensemble {name} (threshold {thr}) ==\n" + pd.DataFrame(rows).round(4).to_string(index=False) +
        f"\nwrote {out}; {np.mean([bool(p) for p in pred]):.1%} of test posts have >= 1 location\n")


# ============================ MAIN ============================
if __name__ == "__main__":
    log(f"=== start: pipelines {[p for p, _ in PIPELINES]}, folds {FOLDS}, {QWEN_REPO}, {EPOCHS} epochs ===")
    todo = [(p, b, k) for p, b in PIPELINES for k in FOLDS]
    for i, (pipe, bidir, fold) in enumerate(todo):
        if (OUT / pipe / f"fold{fold}.pkl").exists():
            log(f"[{pipe} fold {fold}] already done, skipped"); continue
        epochs_after = EPOCHS * sum(not (OUT / p / f"fold{k}.pkl").exists() for p, _, k in todo[i + 1:])
        train_fold(pipe, bidir, fold, epochs_after)
        done = [OUT / pipe / f"fold{k}.pkl" for k in FOLDS if (OUT / pipe / f"fold{k}.pkl").exists()]
        ensemble(f"{OUT.name}_{pipe}_{len(done)}folds", done)     # updated after every fold
    for pipe, _ in PIPELINES:
        done = [OUT / pipe / f"fold{k}.pkl" for k in FOLDS if (OUT / pipe / f"fold{k}.pkl").exists()]
        if done: ensemble(f"{OUT.name}_{pipe}_{len(done)}folds", done)
    every = [OUT / p / f"fold{k}.pkl" for p, _ in PIPELINES for k in FOLDS if (OUT / p / f"fold{k}.pkl").exists()]
    if len(PIPELINES) > 1 and every:
        ensemble(f"{OUT.name}_all_{len(every)}models", every)               # bidirectional + unidirectional together
    log("=== finished ===")