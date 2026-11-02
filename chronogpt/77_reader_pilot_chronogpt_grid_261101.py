# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: (GPU cluster, 2026-11-01)
# TIME TAKEN   : (filled after the run)
# DESCRIPTION  : Reader pilot step R5 (protocol 75). For one ChronoGPT-Instruct vintage, reads pilot chunks with the
#                same prompt as the Claude reader (prompts/75_reader_prompt_quantum_261101.txt), in the model's own
#                chat format (vendor extract_response). Three outputs per chunk:
#                  (1) greedy reply, 64 new tokens (main answer; the authors' portfolio test also reads the first words)
#                  (2) one sampled reply at temperature 0.7 (rerun-agreement check, Jha et al. report 98.68%)
#                  (3) log-probability of each of the 7 answer strings after "### Response:" (cloze scoring of a
#                      fixed choice list; does not depend on the small model following the format)
#                Mode "realtime": only chunks whose vintage (call year - 1, capped at 2024) equals this vintage.
#                Mode "all": every chunk (run with vintage 20241231 to compare the latest model with the realtime one).
# INPUT FILES  : <run>/input/02_pilot_sample.jsonl, <run>/input/<--prompt>,
#                <weights_dir>/{config.json,pytorch_model.bin} (manelalab/chrono-gpt-instruct-v1-<vintage>),
#                <tok_dir> (gpt2 BPE), vendor_chronogpt_instruct_manelalab.py (same folder, MIT license)
# OUTPUT FILES : <run>/output/77_reader_<prompt tag>_<mode>_<vintage>.jsonl (one line per chunk; resumable)
#                <run>/output/77_calib_<prompt tag>_<mode>_<vintage>.json (option log-probs for content-free inputs)
# CHANGES      : 2026-11-01 14:3x EST, during the run (jobs started before this keep the old file):
#                (a) MAX_NEW 64 -> 24. The first 8 chunks (vintage 2024, prompt v1) followed the answer format 0/8
#                    times, and generation took about 2/3 of the 80-140 s per chunk. Without a cache, the first 24
#                    greedy tokens are the same as the first 24 of 64, so the parsed choice does not change; only
#                    the explanation used for the forbidden-word check is shorter.
#                (b) content-free calibration inputs (Zhao et al. 2021, ICML: "N/A", "[MASK]", empty string,
#                    probabilities averaged): the same 7 option log-probs, once per call year. The 8 chunks gave
#                    an informative cloze answer 7/8 times (Claude v1: 1/8), which suggests an answer prior.
#                    Using them is exploratory (decided after seeing those 8 chunks), see protocol 75 R6-C.
# ============================================================
import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

HERE = Path(__file__).resolve().parent
CTX = 1792
EOS = 50256
MAX_NEW = 24  # was 64 until 2026-11-01 14:3x EST (see CHANGES)
# ★ IMPORTANT: the system prompt and layout are copied from the vendor's extract_response() (ChronoGPT_instruct.py on
# Hugging Face). Our prompt + excerpt go where the vendor puts input_text.
SYSTEM = """You are ChronoGPT, a large language model trained by ManelaLab at WashU.
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request."""
OPTIONS = ["already in commercial use", "within 2 years", "in 3 to 5 years", "in 6 to 10 years",
           "more than 10 years from now", "never or not foreseeable", "no information is provided"]
CONTENT_FREE = ["N/A", "[MASK]", ""]  # Zhao et al. (2021, ICML): average the probabilities over these three inputs


def is_oom(ex):
    return isinstance(ex, torch.OutOfMemoryError) or "out of memory" in str(ex).lower()


def oom_retry(fn, what, tries=10, wait=120):
    # ★ IMPORTANT: the V100 is shared. On 2026-11-01 spec 70 lost 4 of 26 vintages (2003, 2004, 2005, 2024) because
    # another process held 25-30 GB of the GPU while the model was moved onto it. Wait and retry instead of failing.
    for k in range(tries):
        try:
            return fn()
        except Exception as ex:
            if not is_oom(ex) or k == tries - 1:
                raise
            print(f"OOM during {what} (try {k + 1}/{tries}), waiting {wait} s: {str(ex)[:160]}", flush=True)
            torch.cuda.empty_cache()
            time.sleep(wait)


def load_model(weights_dir, device):
    spec = importlib.util.spec_from_file_location("chrono", HERE / "vendor_chronogpt_instruct_manelalab.py")
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


def fmt(prompt, r):
    # prompt v4 carries a {call_year} placeholder, filled from the call date as for the Claude reader (code 76)
    p = prompt.replace("{call_year}", str(r["call_year"]))
    return f"\n\n### Instruction:\n{SYSTEM}\n{p}\n\n{r['text']}\n\n### Input:\n### Response:\n"


@torch.inference_mode()
def generate(model, ids, device, temperature, gen):
    idx = torch.tensor([ids], device=device)
    for _ in range(MAX_NEW):
        logits = model(idx[:, -CTX:])[:, -1, :]
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=gen)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if int(nxt) == EOS:
            break
        idx = torch.cat([idx, nxt], dim=1)
    return idx[0, len(ids):].tolist()


@torch.inference_mode()
def option_logprobs(model, tok, ids, device):
    # ★ IMPORTANT (2026-11-01 17:2x EST, during the run): one option at a time, and log-softmax only at the option's
    # own positions. The earlier version scored all 7 options in one batch, which held about 4 GB of logits at once;
    # vintage 2010 then hit CUDA out-of-memory on the shared V100 (another process held 18.5 GB). Same quantity:
    # sum of log P(option token k | prompt + option tokens before k). Vintage 2024 was scored with the batched version.
    out = {}
    for name in OPTIONS:
        o = tok.encode(name)
        logits = model(torch.tensor([ids + o], device=device))[0, len(ids) - 1:len(ids) - 1 + len(o)]
        lp = torch.log_softmax(logits.float(), dim=-1)
        s = float(sum(lp[k, o[k]] for k in range(len(o))))
        out[name] = {"sum": s, "mean": s / len(o), "n_tok": len(o)}
        del logits, lp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", required=True)
    ap.add_argument("--mode", choices=["realtime", "all"], default="realtime")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--weights_dir")
    ap.add_argument("--prompt", default="75_reader_prompt_quantum_261101.txt", help="file name in <run>/input")
    ap.add_argument("--dry_run", action="store_true", help="count chunks and prompt lengths, no model")
    a = ap.parse_args()
    run = Path(a.run_dir)
    tok = GPT2TokenizerFast.from_pretrained(run / "tokenizer")
    prompt = (run / "input" / a.prompt).read_text().strip()
    rows = [json.loads(l) for l in (run / "input" / "02_pilot_sample.jsonl").open()]
    if a.mode == "realtime":
        rows = [r for r in rows if r["vintage"] == a.vintage]
    tag = Path(a.prompt).stem.split("261101")[-1].lstrip("_") or "v1"
    out = run / "output" / f"77_reader_{tag}_{a.mode}_{a.vintage}.jsonl"
    years = sorted({r["call_year"] for r in rows})
    done = {json.loads(l)["pilot_id"] for l in out.open()} if out.exists() else set()
    rows = [r for r in rows if r["pilot_id"] not in done]
    lens = [len(tok.encode(fmt(prompt, r))) for r in rows]
    print(f"vintage {a.vintage} mode {a.mode}: {len(rows)} chunks to do ({len(done)} done), "
          f"prompt tokens max {max(lens) if lens else 0}", flush=True)
    # ★ IMPORTANT: prompt + 64 new tokens must fit the 1,792-token context, otherwise the model silently drops the
    # start of the instruction (vendor generate() keeps only the last 1,792 tokens).
    assert not lens or max(lens) + MAX_NEW <= CTX, "chunk too long for the context"
    if a.dry_run or not rows:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.weights_dir, device)
    compute = "bfloat16 (model default)"
    probe = lambda: model(torch.tensor([[EOS, EOS]]).to(device))
    try:
        oom_retry(probe, "bf16 probe")
    except Exception as ex:  # ★ IMPORTANT: same fallback as code 70, so every vintage is computed alike
        if is_oom(ex):
            raise  # an OOM must not switch the vintage to float32
        compute = f"float32 fallback ({str(ex)[:120]})"
        torch.Tensor.bfloat16 = lambda self, *args, **kw: self.float()
        oom_retry(probe, "float32 probe")
    # ★ IMPORTANT: content-free calibration (exploratory, added 2026-11-01 14:3x EST). One file per prompt, mode and
    # vintage, with the option log-probs for each content-free input and each call year read in this run.
    cal = run / "output" / f"77_calib_{tag}_{a.mode}_{a.vintage}.json"
    if not cal.exists():
        res = {}
        for y in years:
            res[str(y)] = {cf: oom_retry(lambda: option_logprobs(model, tok, tok.encode(fmt(prompt, {"text": cf, "call_year": y})),
                                                                   device), f"calibration {y}") for cf in CONTENT_FREE}
        tmp = cal.with_suffix(".tmp")
        tmp.write_text(json.dumps({"vintage": a.vintage, "mode": a.mode, "prompt_file": a.prompt, "compute": compute,
                                   "content_free": CONTENT_FREE, "by_call_year": res}))
        tmp.rename(cal)
        print("calibration saved", cal.name, flush=True)
    gen = torch.Generator(device=device).manual_seed(20261101)
    with out.open("a") as f:
        for r, n in zip(rows, lens):
            t0 = time.time()
            ids = tok.encode(fmt(prompt, r))
            # a retried chunk redraws its sample from where the generator stands; the greedy reply and log-probs
            # do not depend on the generator
            greedy = oom_retry(lambda: tok.decode(generate(model, ids, device, 0.0, gen)), f"{r['pilot_id']} greedy")
            sampled = oom_retry(lambda: tok.decode(generate(model, ids, device, 0.7, gen)), f"{r['pilot_id']} sample")
            lps = oom_retry(lambda: option_logprobs(model, tok, ids, device), f"{r['pilot_id']} log-probs")
            f.write(json.dumps({"pilot_id": r["pilot_id"], "vintage": a.vintage, "mode": a.mode, "prompt_file": a.prompt,
                                "realtime": r["vintage"] == a.vintage, "prompt_tokens": n, "greedy": greedy,
                                "sampled_t07": sampled, "option_logprobs": lps,
                                "compute": compute, "torch": torch.__version__,
                                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                                "seconds": round(time.time() - t0, 1)}) + "\n")
            f.flush()
            print(r["pilot_id"], round(time.time() - t0, 1), "s |", greedy[:80].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
