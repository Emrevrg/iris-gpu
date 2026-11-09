"""IRIS-Evrensel: her mimari, her veri tipi iÃ§in kayÄ±psÄ±z, GPU'da aÃ§Ä±lan aÄŸÄ±rlÄ±k sÄ±kÄ±ÅŸtÄ±rma.

TensÃ¶r baÅŸÄ±na iki aday Ã¼retilir, kÃ¼Ã§Ã¼k olan saklanÄ±r (ikisi de bit-kesin):
  A) bÃ¼tÃ¼n bayt akÄ±ÅŸÄ± -> nvCOMP ANS            (int8/uint8, fp8, paketli int4/MXFP4 bloklarÄ±, Ã¶lÃ§ekler)
  B) alan ayrÄ±ÅŸtÄ±rma (kayan nokta):            Ã¼s alanÄ± ayrÄ± dÃ¼zlem -> ANS ; iÅŸaret+mantis ham
       bf16: 8 bit Ã¼s | 1+7      fp16: 5 bit Ã¼s | 1+10     fp32: 8 bit Ã¼s | 1+23
AÃ§ma tamamen GPU'da: sÄ±kÄ±ÅŸÄ±k + ham dÃ¼zlemler pinned RAM'den GPU'ya, nvCOMP ile aÃ§Ä±lÄ±r, torch ile birleÅŸtirilir.
Denetim: her model iÃ§in aÃ§Ä±lan bayt akÄ±ÅŸÄ±nÄ±n SHA256'sÄ± orijinalle aynÄ± olmalÄ±.
python iris_evrensel.py MODEL_DIZINI [--tekrar 3]"""
import argparse, glob, hashlib, json, os, time
import numpy as np, torch
from safetensors import safe_open
from nvidia import nvcomp

ANS = nvcomp.Codec(algorithm="ANS")
ALAN = {torch.bfloat16: (16, 7, 8), torch.float16: (16, 10, 5), torch.float32: (32, 23, 8)}  # (bit, mantis, Ã¼s)
INT = {16: torch.int16, 32: torch.int32}


def sik(u8_gpu):
    return np.asarray(ANS.encode(nvcomp.as_array(u8_gpu)).cpu())


def ac(sik_gpu):
    return torch.as_tensor(ANS.decode(nvcomp.as_array(sik_gpu))).view(torch.uint8)


def alan_ayir(t):
    bit, m, e = ALAN[t.dtype]
    u = t.contiguous().reshape(-1).view(INT[bit]).to(torch.int64) & ((1 << bit) - 1)
    us = ((u >> m) & ((1 << e) - 1)).to(torch.uint8)
    sm = ((u >> (bit - 1)) << m) | (u & ((1 << m) - 1))  # iÅŸaret mantisin Ã¼stÃ¼ne: m+1 bit
    nb = (m + 1 + 7) // 8
    sm_b = torch.stack([((sm >> (8 * i)) & 0xFF).to(torch.uint8) for i in range(nb)], 1).reshape(-1)
    return us, sm_b, nb


def alan_birlestir(us, sm_b, nb, dtype):
    bit, m, e = ALAN[dtype]
    s = sm_b.view(-1, nb).to(torch.int64); sm = torch.zeros(s.shape[0], dtype=torch.int64, device=s.device)
    for i in range(nb): sm |= s[:, i] << (8 * i)
    u = ((sm >> m) << (bit - 1)) | (us.to(torch.int64) << m) | (sm & ((1 << m) - 1))
    u = torch.where(u >= (1 << (bit - 1)), u - (1 << bit), u)  # iÅŸaretli tamsayÄ±ya
    return u.to(INT[bit]).view(dtype)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dizin"); ap.add_argument("--tekrar", type=int, default=3)
    ap.add_argument("--azami_GB", type=float, default=12.0)
    a = ap.parse_args()
    dosyalar = sorted(glob.glob(os.path.join(a.dizin, "**/*.safetensors"), recursive=True))
    paket, orj, tipler, h_ref = [], 0, {}, hashlib.sha256()
    for f in dosyalar:
        with safe_open(f, "pt") as fh:
            for k in sorted(fh.keys()):
                t = fh.get_tensor(k)
                if orj > a.azami_GB * 1e9: break
                b = t.contiguous().view(torch.uint8).reshape(-1) if t.numel() else torch.empty(0, dtype=torch.uint8)
                h_ref.update(b.numpy().tobytes()); orj += b.numel()
                tipler[str(t.dtype)] = tipler.get(str(t.dtype), 0) + b.numel()
                if b.numel() < 65536:
                    paket.append(("ham", t.dtype, t.shape, b.numpy().copy())); continue
                g = b.cuda(); cA = sik(g); aday = ("A", t.dtype, t.shape, cA, b.numel())
                if t.dtype in ALAN:
                    us, smb, nb = alan_ayir(t.cuda())
                    cB = sik(us); boyB = cB.nbytes + smb.numel()
                    if boyB < cA.nbytes: aday = ("B", t.dtype, t.shape, cB, smb.cpu().numpy(), nb)
                    del us, smb
                paket.append(aday); del g
    torch.cuda.empty_cache()
    sik_boy = sum(p[3].nbytes + (p[4].nbytes if p[0] == "B" else 0) for p in paket)
    # pinned tamponlar (diskten okunmuÅŸ hal RAM'de)
    pin = [[torch.from_numpy(x).pin_memory() for x in (p[3:5] if p[0] == "B" else p[3:4])] for p in paket]
    sureler = []
    for _ in range(a.tekrar):
        torch.cuda.synchronize(); t0 = time.perf_counter(); cikti = []
        for p, pp in zip(paket, pin):
            if p[0] == "ham": cikti.append(pp[0].cuda(non_blocking=True))
            elif p[0] == "A": cikti.append(ac(pp[0].cuda(non_blocking=True)))
            else:
                us = ac(pp[0].cuda(non_blocking=True)); smb = pp[1].cuda(non_blocking=True)
                cikti.append(alan_birlestir(us, smb, p[5], p[1]).reshape(-1).view(torch.uint8))
        torch.cuda.synchronize(); sureler.append(time.perf_counter() - t0)
    h = hashlib.sha256()
    for c in cikti: h.update(c.cpu().numpy().tobytes())
    sn = sorted(sureler)[len(sureler) // 2]
    print("IRIS_EVRENSEL", json.dumps({"model": os.path.basename(a.dizin.rstrip("/")), "GB": round(orj / 1e9, 3),
          "tipler_GB": {k: round(v / 1e9, 3) for k, v in tipler.items()}, "oran": round(sik_boy / orj, 4),
          "acma_GBs": round(orj / 1e9 / sn, 2), "bit_kesin": h.hexdigest() == h_ref.hexdigest(),
          "kip_sayisi": {k: sum(1 for p in paket if p[0] == k) for k in ("ham", "A", "B")}}), flush=True)


if __name__ == "__main__":
    main()
