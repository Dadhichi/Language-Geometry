"""Run layer_stats on real data for the given layers.  Usage: python run_layers.py 14 0 28 ..."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlib

out = os.path.join(dlib.OUT, "layers")
os.makedirs(out, exist_ok=True)
W = np.load(os.path.join(dlib.DATA, "wu_basis.npy")).astype(np.float32)
weig = np.load(os.path.join(dlib.DATA, "wu_eigs.npy")).astype(np.float64)
ntok = np.load(os.path.join(dlib.DATA, "ntok_dev.npy"))
layers = [int(x) for x in sys.argv[1:]] or dlib.LAYERS


def rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return float("nan")


for l in layers:
    t0 = time.time()
    Xd, Xt = dlib.load("dev", l), dlib.load("devtest", l)
    S, rows, arrs = dlib.layer_stats(Xd, Xt, W, weig, ntok_d=ntok, verbose=True)
    del Xd, Xt
    S["layer"] = l
    for r in rows:
        r["layer"] = l
    json.dump(dict(S=S, rows=rows), open(os.path.join(out, f"L{l}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, f"L{l}.npz"), **arrs)
    print(f"L{l}: {time.time() - t0:.0f}s rss={rss():.0f}MB  f_lang={S['f_lang_dev']:.3f} f_sent={S['f_sent_dev']:.3f} "
          f"inv={S['inv_frac_dev']:.3f} rho_id={S['rho_identity_amb']:.3f} rho_shift={S['rho_shift_amb']:.3f} "
          f"rho_shift_k64={S['rho_shift_k64']:.3f} rho_proc_k64={S['rho_proc_k64']:.3f} W_a={S['W_a']:.3f} W_b={S['W_b']:.3f}",
          flush=True)
