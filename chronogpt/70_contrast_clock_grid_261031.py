# ============================================================
# START DATE   : 2026-10-31
# CODE RUN DATE: (GPU cluster, 2026-11-01)
# TIME TAKEN   : (filled after the run)
# DESCRIPTION  : Spec 70_spec_chronogpt_contrast_clock_261031.md. For one ChronoGPT vintage, scores minimal sentence
#                pairs: Part A "has already happened" vs "has not yet happened" for 13 dated events + 6 controls;
#                Part B "only a few years away" vs "still decades away" for 7 technologies + 1 neutral topic.
#                Saves the log-prob of every sentence; margins and tests are computed later on the Mac.
# INPUT FILES  : <weights_dir>/{config.json,pytorch_model.bin} (manelalab/chrono-gpt-v1-<vintage>), <tok_dir> (gpt2 BPE),
#                vendor_chronogpt_inference_manelalab.py (same folder, MIT license)
# OUTPUT FILES : <out_dir>/70_contrast_clock_<vintage>.json
# CHANGES      : 2026-11-01 14:1x EST: CUDA out-of-memory retry (wait 2 min, up to 10 tries) at model load, bf16 probe
#                and scoring, as in code 77. On 2026-11-01 vintages 2003, 2004, 2005, 2024 had no output because
#                another process held 25-30 GB of the shared V100. An OOM no longer switches a vintage to float32.
# ============================================================
import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

HERE = Path(__file__).resolve().parent

# ★ IMPORTANT: sentences fixed in spec 70 before any vintage was run. (subject, plural?, past participle phrase)
EVENTS = {
    "euro_coins": ("Europe", False, "started using euro coins and notes as everyday money"),
    "genome_complete": ("scientists", True, "finished sequencing the complete human genome"),
    "pluto_dwarf": ("astronomers", True, "reclassified Pluto as a dwarf planet"),
    "iphone": ("Apple", False, "released the iPhone"),
    "bitcoin": ("someone", False, "mined the first bitcoin"),
    "watson_jeopardy": ("IBM's Watson computer", False, "won Jeopardy against human champions"),
    "higgs": ("physicists", True, "discovered the Higgs boson"),
    "grav_waves": ("scientists", True, "detected gravitational waves directly"),
    "blackhole_image": ("astronomers", True, "photographed a black hole"),
    "moon_farside": ("a spacecraft", False, "landed on the far side of the Moon"),
    "mrna_vaccine": ("regulators", True, "approved an mRNA vaccine"),
    "mars_helicopter": ("a helicopter", False, "flown on Mars"),
    "chatgpt": ("OpenAI", False, "released ChatGPT to the public"),
    # controls: happened before 1999
    "ctrl_moon_landing": ("astronauts", True, "walked on the Moon"),
    "ctrl_deep_blue": ("a computer", False, "beaten the world chess champion"),
    # controls: not happened by the end of 2024
    "ctrl_fusion_plant": ("a fusion power plant", False, "delivered electricity to the grid"),
    "ctrl_mars_humans": ("humans", True, "landed on Mars"),
    "ctrl_rsa_broken": ("a quantum computer", False, "broken today's internet encryption"),
    "ctrl_flying_cars": ("flying cars", True, "become common in everyday life"),
}
A_LEADS = ["", "Reporters note that", "It is widely known that"]

TOPICS = {  # (noun phrase, plural?)
    "quantum_practical": ("practical quantum computers", True),
    "quantum_crypto": ("quantum computers that can break today's encryption", True),
    "fusion_power": ("commercial fusion power", False),
    "self_driving": ("self-driving cars on ordinary roads", True),
    "human_level_ai": ("computers with human-level intelligence", True),
    "humanoid_robots": ("humanoid robots in ordinary homes", True),
    "mars_humans": ("a human landing on Mars", False),
    "neutral_highway": ("the completion of the new highway", False),
}
B_LEADS = ["", "Experts say that", "Many people believe that"]


def a_pairs(subj, plural, vp):
    have = "have" if plural else "has"
    yes = [f"{subj} {have} already {vp}.", f"by now, {subj} {have} {vp}."]
    no = [f"{subj} {have} not yet {vp}.", f"so far, {subj} {have} never {vp}."]
    return yes, no


def b_pairs(np_, plural):
    be = "are" if plural else "is"
    near = [f"{np_} {be} only a few years away.", f"{np_} will arrive within the next five years.",
            f"{np_} {be} just around the corner."]
    far = [f"{np_} {be} still decades away.", f"{np_} will not arrive for at least thirty years.",
           f"{np_} {be} a distant dream."]
    return near, far


def with_lead(lead, s):
    # subjects are stored in mid-sentence case (proper nouns capitalized); only the first character is raised
    if lead:
        return lead, " " + s
    return "", s[0].upper() + s[1:]


def is_oom(ex):
    return isinstance(ex, torch.OutOfMemoryError) or "out of memory" in str(ex).lower()


def oom_retry(fn, what, tries=10, wait=120):
    # ★ IMPORTANT: the V100 is shared (same handling as code 77). Wait and retry instead of losing the vintage.
    for k in range(tries):
        try:
            return fn()
        except Exception as ex:
            if not is_oom(ex) or k == tries - 1:
                raise
            print(f"OOM during {what} (try {k + 1}/{tries}), waiting {wait} s: {str(ex)[:160]}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(wait)


def load_model(weights_dir, device):
    spec = importlib.util.spec_from_file_location("chrono", HERE / "vendor_chronogpt_inference_manelalab.py")
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)
    cfg = json.loads((Path(weights_dir) / "config.json").read_text())
    cfg = {k: cfg[k] for k in ("vocab_size", "num_layers", "num_heads", "model_dim")}
    m = ch.ChronoGPT(**cfg)
    m.load_state_dict(torch.load(Path(weights_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True))

    def move():
        try:
            return m.to(device).eval()
        except Exception:
            m.to("cpu")  # free the half-moved weights before the next try
            raise
    return oom_retry(move, "model load")


@torch.inference_mode()
def score(model, tok, items, device, bs=64):
    # ★ CORE LOGIC: log P(continuation | lead), summed over the continuation's tokens; an empty lead uses <|endoftext|>
    enc = []
    for lead, cont in items:
        p = tok.encode(lead) if lead else [tok.eos_token_id]
        enc.append((p, tok.encode(cont)))
    out = []
    for s in range(0, len(enc), bs):
        chunk = enc[s:s + bs]
        L = max(len(p) + len(c) for p, c in chunk)
        ids = torch.full((len(chunk), L), tok.eos_token_id, dtype=torch.long)
        for i, (p, c) in enumerate(chunk):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        logits, _ = model(ids.to(device))
        lp = torch.log_softmax(logits.float(), dim=-1).cpu()
        for j, (p, c) in enumerate(chunk):
            out.append(float(sum(lp[j, len(p) - 1 + k, c[k]] for k in range(len(c)))))
    return out


def build():
    rows = []  # (part, key, side, phrasing, lead, lead_text, continuation)
    for key, (subj, pl, vp) in EVENTS.items():
        yes, no = a_pairs(subj, pl, vp)
        for side, sents in (("yes", yes), ("no", no)):
            for k, s in enumerate(sents):
                for li, lead in enumerate(A_LEADS):
                    lt, cont = with_lead(lead, s)
                    rows.append(("A", key, side, k, li, lt, cont))
    for key, (np_, pl) in TOPICS.items():
        near, far = b_pairs(np_, pl)
        for side, sents in (("near", near), ("far", far)):
            for k, s in enumerate(sents):
                for li, lead in enumerate(B_LEADS):
                    lt, cont = with_lead(lead, s)
                    rows.append(("B", key, side, k, li, lt, cont))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", required=True)
    ap.add_argument("--weights_dir")
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry_run", action="store_true", help="print the sentences and exit (no model)")
    a = ap.parse_args()
    rows = build()
    if a.dry_run:
        for r in rows:
            print(r[0], r[1], r[2], repr(r[5] + r[6]))
        print(len(rows), "sentences")
        return
    out = Path(a.out_dir) / f"70_contrast_clock_{a.vintage}.json"
    if out.exists():
        print("exists, skipping", out)
        return
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = GPT2TokenizerFast.from_pretrained(a.tok_dir)
    model = load_model(a.weights_dir, device)
    compute = "bfloat16 (model default)"
    probe = lambda: model(torch.tensor([[50256, 50256]]).to(device))
    try:
        oom_retry(probe, "bf16 probe")
    except Exception as ex:  # ★ IMPORTANT: GPUs without bf16 support -> compute in float32 for every vintage alike
        if is_oom(ex):
            raise  # an OOM must not switch the vintage to float32
        compute = f"float32 fallback ({str(ex)[:120]})"
        torch.Tensor.bfloat16 = lambda self, *args, **kw: self.float()
        oom_retry(probe, "float32 probe")
    lps = oom_retry(lambda: score(model, tok, [(r[5], r[6]) for r in rows], device), "scoring")
    res = {"vintage": a.vintage, "device": device, "compute": compute, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "rows": [dict(zip(["part", "key", "side", "phrasing", "lead", "lead_text", "text", "logprob"], list(r) + [lp]))
                    for r, lp in zip(rows, lps)], "seconds": round(time.time() - t0, 1)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(out)
    print("done", out, res["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
