"""IRIS-GPU vs AirLLM yöntemi: modelin TÜM ağırlıklarını parça parça GPU'ya taşıma süresi.
AirLLM katman-akışlı çıkarımda her token için bütün katmanları safetensors'tan GPU'ya yükler;
bu yüzden token başına alt sınır = tüm modeli bir kez GPU'ya taşıma süresi.
Taban: safetensors load_file(device='cuda') (AirLLM'nin yükleme yolu), sayfa önbelleği sıcak.
IRIS: sıkışık üs (nvCOMP ANS) + ham işaret/mantis, /dev/shm'den pinned'a paralel okuma + GPU'da açma.
Her parça SHA ile bit-kesin denetlenir."""
import glob, hashlib, json, os, sys, time, concurrent.futures as cf
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import load_file
from nvidia import nvcomp

sys.path.insert(0, os.path.dirname(__file__))
from iris_gpu import duzlemlere_ayir, birlestir

dosyalar = sorted(glob.glob(sys.argv[1] + "/*.safetensors"))
codec = nvcomp.Codec(algorithm="ANS")
havuz = cf.ThreadPoolExecutor(16)
top_orj = top_sik = t_taban = t_iris = 0.0; hepsi_kesin = True

for f in dosyalar:
    load_file(f)  # sayfa önbelleğini ısıt
    torch.cuda.synchronize(); t = time.perf_counter(); d = load_file(f, device="cuda"); torch.cuda.synchronize(); t_taban += time.perf_counter() - t
    del d; torch.cuda.empty_cache()

    parcalar = []
    with safe_open(f, "pt") as fh:
        for k in fh.keys():
            x = fh.get_tensor(k)
            if x.dtype == torch.bfloat16: parcalar.append(x.reshape(-1))
            else: top_orj += x.numel() * x.element_size(); top_sik += x.numel() * x.element_size()  # bf16 dışı ham geçer
    tum = torch.cat(parcalar); orj = tum.numel() * 2; top_orj += orj
    ref = hashlib.sha256(tum.view(torch.int16).numpy().tobytes()).hexdigest()
    us, sm = duzlemlere_ayir(tum)
    sik = np.asarray(codec.encode(nvcomp.as_array(us.cuda())).cpu()); smn = sm.numpy()
    yol = "/dev/shm/iris_akis.bin"
    with open(yol, "wb") as w: w.write(sik.tobytes()); w.write(smn.tobytes())
    n = sik.nbytes + smn.nbytes; top_sik += n
    tampon = torch.empty(n, dtype=torch.uint8).pin_memory(); tnp = tampon.numpy()
    def oku(ab):
        with open(yol, "rb", buffering=0) as r: r.seek(ab[0]); r.readinto(memoryview(tnp)[ab[0]:ab[1]])
    sinir = [(i * n // 16, (i + 1) * n // 16) for i in range(16)]
    torch.cuda.synchronize(); t = time.perf_counter()
    list(havuz.map(oku, sinir)); g = tampon.cuda(non_blocking=True)
    usd = torch.as_tensor(codec.decode(nvcomp.as_array(g[:sik.nbytes]))).view(torch.uint8)
    wgt = birlestir(usd, g[sik.nbytes:]); torch.cuda.synchronize(); t_iris += time.perf_counter() - t
    hepsi_kesin &= hashlib.sha256(wgt.view(torch.int16).cpu().numpy().tobytes()).hexdigest() == ref
    os.remove(yol); del tum, us, sm, g, usd, wgt, tampon; torch.cuda.empty_cache()
    print(json.dumps({"parca": os.path.basename(f), "taban_sn": round(t_taban, 2), "iris_sn": round(t_iris, 2)}), flush=True)

print("IRIS_AKIS", json.dumps({"parca_sayisi": len(dosyalar), "model_GB": round(top_orj / 1e9, 2), "sikisik_oran": round(top_sik / top_orj, 4),
      "airllm_yontemi_tum_model_sn": round(t_taban, 2), "iris_gpu_tum_model_sn": round(t_iris, 2),
      "hizlanma": round(t_taban / t_iris, 2), "bit_kesin": hepsi_kesin}))
