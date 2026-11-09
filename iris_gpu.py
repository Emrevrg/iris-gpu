"""IRIS-GPU: bf16 ağırlıkları düzlemlere ayırıp üs düzlemini GPU'da nvCOMP ile açar.

Biçim (bf16 = 1 işaret + 8 üs + 7 mantis):
  - üs düzlemi (uint8, düşük entropi) -> nvCOMP (ANS / zstd / gdeflate) ile sıkıştırılır
  - işaret+mantis düzlemi (uint8, ~yüksek entropi) -> ham saklanır
Açma: sıkışık üs + ham düzlem -> GPU'ya kopyala -> GPU'da aç -> GPU'da bf16'ya birleştir.
Ölçülen hız = orijinal bayt / (H2D kopya + açma + birleştirme) süresi. Bit-kesinlik SHA ile denetlenir.
"""
import argparse, hashlib, json, os, time
import numpy as np, torch
from safetensors import safe_open
from nvidia import nvcomp


def duzlemlere_ayir(t: torch.Tensor):
    u = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    us = ((u >> 7) & 0xFF).to(torch.uint8)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    return us, sm


def birlestir(us: torch.Tensor, sm: torch.Tensor):
    us16 = us.to(torch.int16); sm16 = sm.to(torch.int16)
    u = ((sm16 & 0x80) << 8) | (us16 << 7) | (sm16 & 0x7F)  # int16'da taşma = doğru işaret biti
    return u.view(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dosyalar", nargs="+")
    ap.add_argument("--algo", default="ANS")
    ap.add_argument("--tekrar", type=int, default=5)
    a = ap.parse_args()

    tensorler = []
    for d in a.dosyalar:
        with safe_open(d, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.dtype == torch.bfloat16 and t.numel() > 4096:
                    tensorler.append(t.reshape(-1))
    tum = torch.cat(tensorler)
    orj_bayt = tum.numel() * 2
    ref_sha = hashlib.sha256(tum.view(torch.int16).numpy().tobytes()).hexdigest()

    us, sm = duzlemlere_ayir(tum)
    codec = nvcomp.Codec(algorithm=a.algo)
    us_gpu = us.cuda()
    sik = codec.encode(nvcomp.as_array(us_gpu))
    sik_host = np.asarray(sik.cpu())  # sıkışık üs düzlemi (diskte duracak hal)
    sm_host = sm.numpy()
    sik_bayt = sik_host.nbytes + sm_host.nbytes
    del us_gpu, sik

    # pinned bellek: diskten okunmuş veri RAM'de hazır kabul edilir
    sik_pin = torch.from_numpy(sik_host).pin_memory()
    sm_pin = torch.from_numpy(sm_host).pin_memory()
    torch.cuda.synchronize()

    sureler = []
    for _ in range(a.tekrar):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        sik_d = sik_pin.cuda(non_blocking=True)
        sm_d = sm_pin.cuda(non_blocking=True)
        us_d = torch.as_tensor(codec.decode(nvcomp.as_array(sik_d)), device="cuda").view(torch.uint8)
        w = birlestir(us_d, sm_d)
        torch.cuda.synchronize(); sureler.append(time.perf_counter() - t0)
    # uçtan uca: sıkışık dosyayı sayfa önbelleğinden kalıcı pinned tampona oku (paralel) + GPU'da aç
    import concurrent.futures as cf
    yol = "/dev/shm/iris_gpu_test.bin"
    with open(yol, "wb") as fh: fh.write(sik_host.tobytes()); fh.write(sm_host.tobytes())
    tampon = torch.empty(sik_bayt, dtype=torch.uint8).pin_memory(); tnp = tampon.numpy()
    def oku(ab):
        a_, b_ = ab
        with open(yol, "rb", buffering=0) as fh: fh.seek(a_); fh.readinto(memoryview(tnp)[a_:b_])
    P = 16; sinir = [(i * sik_bayt // P, (i + 1) * sik_bayt // P) for i in range(P)]
    havuz = cf.ThreadPoolExecutor(P); uc = []
    for _ in range(a.tekrar):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        list(havuz.map(oku, sinir))
        d_ = tampon.cuda(non_blocking=True)
        us_d = torch.as_tensor(codec.decode(nvcomp.as_array(d_[:sik_host.nbytes]))).view(torch.uint8)
        w = birlestir(us_d, d_[sik_host.nbytes:])
        torch.cuda.synchronize(); uc.append(time.perf_counter() - t0)
    os.remove(yol)
    sha = hashlib.sha256(w.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    sn = sorted(sureler)[len(sureler) // 2]
    print(json.dumps({"uctan_uca_dosyadan_GBs": round(orj_bayt / 1e9 / sorted(uc)[len(uc) // 2], 2)}))
    print(json.dumps({"IRIS_GPU": True, "algo": a.algo, "orj_GB": round(orj_bayt / 1e9, 3),
                      "oran": round(sik_bayt / orj_bayt, 4), "acma_GBs": round(orj_bayt / 1e9 / sn, 2),
                      "medyan_sn": round(sn, 4), "bit_kesin": sha == ref_sha}))


if __name__ == "__main__":
    main()
