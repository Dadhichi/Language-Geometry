"""Real-data sentence-level belief-simplex pilot, per layer and variant (raw mean pooling / sqrt(ntok)-rescaled)."""
import sys, os, json, time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import flib as F
import pipeline as P

layers = [int(x) for x in sys.argv[1].split(",")]
variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["raw", "sqrtn"]
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 50
Idev, Itest = F.load_instr("dev"), F.load_instr("devtest")
tok = np.load(os.path.join(os.path.dirname(F.HERE), "D_offsets", "tok_sim.npz"))
L = 12
iu = np.triu_indices(L, 1)


def sym(M):
    return 0.5 * (M + M.T)


Bbag_mean = sym(np.concatenate([Idev["B_bag"], Itest["B_bag"]], 2).mean(2))
etok_mean = sym(np.concatenate([Idev["e_tokc"], Itest["e_tokc"]], 2).mean(2))
same_script = (F.SCRIPT[:, None] == F.SCRIPT[None]).astype(float)
TEXT = dict(tok_hist_int=tok["tok_hist_int"], B_bag=Bbag_mean, e_tokc=etok_mean)
GROUPS = {}
m = np.zeros((L, L), bool); m[np.ix_(F.LATIN, F.LATIN)] = True; np.fill_diagonal(m, False); GROUPS["LatLat"] = m
m = np.zeros((L, L), bool); m[7, 8] = m[8, 7] = True; GROUPS["zho-jpn"] = m
GROUPS["rest"] = ~(GROUPS["LatLat"] | GROUPS["zho-jpn"]) & ~np.eye(L, dtype=bool)


def partial_corr(x, y, C):
    X = np.c_[np.ones(len(x)), C]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def script_tests(W, I):
    """barycentric W [L,n,L]; returns partial correlations (controls: log ntok, punct, digit)"""
    out = {}
    han = list(I["classes"]).index("han")
    ctrl = lambda i: np.c_[np.log(I["ntok"][i]), I["punct"][i], I["digit"][i]]
    others = [j for j in range(L) if j not in (7, 8)]
    # jpn: kanji fraction -> pull toward zho, specifically (vs mean pull toward the other 10 vertices)
    y = W[8, :, 7] - W[8, :, others].mean(0)
    out["jpn_kanji->zho"] = partial_corr(I["scr"][8, :, han], y, ctrl(8))
    out["jpn_kanji->zho_raw_wzho"] = partial_corr(I["scr"][8, :, han], W[8, :, 7], ctrl(8))
    # control: kanji fraction -> pull toward eng (should be ~0 or negative)
    out["jpn_kanji->eng_ctrl"] = partial_corr(I["scr"][8, :, han], W[8, :, 0] - W[8, :, others].mean(0), ctrl(8))
    # zho: fraction of its characters also present in the jpn rendering -> pull toward jpn
    y = W[7, :, 8] - W[7, :, others].mean(0)
    out["zho_sharedchar->jpn"] = partial_corr(I["e_char"][7, 8], y, ctrl(7))
    # non-Latin languages: Latin-letter fraction -> pull toward the Latin face vs the other non-Latin vertices
    xs, ys = [], []
    for i in F.NONLAT:
        nl = [j for j in F.NONLAT if j != i]
        y = W[i][:, F.LATIN].mean(1) - W[i][:, nl].mean(1)
        x = I["scr"][i, :, 0].astype(np.float64)
        X = np.c_[np.ones(len(x)), ctrl(i)]
        xs.append(x - X @ np.linalg.lstsq(X, x, rcond=None)[0])
        ys.append(y - X @ np.linalg.lstsq(X, y, rcond=None)[0])
    out["nonLatin_latinfrac->LatinFace"] = float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
    return out


def surface_sharing(Z64, I):
    """out-of-span interaction residual similarity between translations vs their shared-token fraction"""
    u = Z64[..., 11:] - Z64[..., 11:].mean(1, keepdims=True)
    un = u / np.linalg.norm(u, axis=2, keepdims=True)
    rs, rl = [], []
    esym = 0.5 * (I["e_tokc"] + I["e_tokc"].transpose(1, 0, 2))
    for i, j in zip(*iu):
        c = (un[i] * un[j]).sum(1)
        r = np.corrcoef(c, esym[i, j])[0, 1]
        rs.append(r)
        if i in F.LATIN and j in F.LATIN:
            rl.append(r)
    return float(np.mean(rs)), float(np.mean(rl))


def groups_fit(ctx, d_dev, d_test, nm):
    xd, xt_ = ctx.xt["dev"][nm], ctx.xt["test"][nm]
    Dd = [F.dhat(xd * GROUPS[g][:, :, None], ctx.A) for g in GROUPS]
    Dt = [F.dhat(xt_ * GROUPS[g][:, :, None], ctx.A) for g in GROUPS]
    dd, dt = F.centre(d_dev), F.centre(d_test)
    M = F.metric(F.residualize(dd, ctx.Zc["dev"]))
    g, Gam = F.gfit(dd, Dd, ctx.Zc["dev"], M)
    rho, R2, _ = F.heldout(dt, Dt, ctx.Zc["test"], Gam, g, M)
    gt = F.gfit(dt, Dt, ctx.Zc["test"], M)[0]
    return {gn: dict(g_dev=float(g[k]), g_test=float(gt[k]), rho=float(rho[k])) for k, gn in enumerate(GROUPS)}


results = {}
for layer in layers:
    for variant in variants:
        t0 = time.time()
        z = F.load_layer(layer, variant)
        Zd64, Zt64 = z["Z_dev"].astype(np.float64), z["Z_test"].astype(np.float64)
        Zd, Zt = Zd64[..., :11], Zt64[..., :11]
        A = Zd.mean(1)
        n, m_ = Zd.shape[1], Zt.shape[1]
        R = {}
        # ---- geometry
        a_t2 = np.diag(z["gram_a_test"]).sum()
        o2 = float(z["o2_test"].astype(np.float64).sum())
        e2 = o2 - m_ * a_t2
        dt = F.centre(Zt)
        R["o_inspan"] = float((Zt ** 2).sum() / o2)
        R["e_inspan"] = float((dt ** 2).sum() / e2)
        R["e_in64"] = float((F.centre(Zt64) ** 2).sum() / e2)
        R["centroid_share_of_offset"] = float(m_ * a_t2 / o2)
        edges = np.array([np.sum((A[i] - A[j]) ** 2) for i, j in zip(*iu)])
        R["rms_edge"], R["rms_disp"] = float(np.sqrt(edges.mean())), float(np.sqrt((dt ** 2).sum(2).mean()))
        R["disp_over_edge"] = R["rms_disp"] / R["rms_edge"]
        R["min_edge_over_disp"] = float(np.sqrt(edges.min()) / R["rms_disp"])
        R["min_edge_pair"] = "-".join(F.LANGS[k] for k in (iu[0][edges.argmin()], iu[1][edges.argmin()]))
        W = F.bary(Zt, A)
        own = np.stack([W[i, :, i] for i in range(L)])
        R["own_w_median"], R["own_w_p05"], R["own_w_p95"] = map(float, np.percentile(own, [50, 5, 95]))
        R["own_w_by_lang_sd"] = [float(own[i].std()) for i in range(L)]
        R["outside_hull"] = float((W.min(2) < -0.05).mean())
        oth = W.copy()
        for i in range(L):
            oth[i, :, i] = -np.inf
        R["frac_maxother_gt_0.1"] = float((oth.max(2) > 0.1).mean())
        # ---- LDA posterior (64-d, fit dev, score devtest)
        y = np.repeat(np.arange(L), n)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zd64.reshape(-1, 64), y)
        Pt = lda.predict_proba(Zt64.reshape(-1, 64)).reshape(L, m_, L)
        R["lda_acc"] = float((Pt.argmax(2) == np.arange(L)[:, None]).mean())
        R["lda_own_p_mean"] = float(np.mean([Pt[i, :, i].mean() for i in range(L)]))
        conf = Pt.mean(1)
        R["lda_conf_top"] = sorted([(float(conf[i, j]), F.LANGS[i] + ">" + F.LANGS[j]) for i in range(L)
                                    for j in range(L) if i != j], reverse=True)[:5]
        lda11 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zd.reshape(-1, 11), y)
        R["lda11_acc"] = float(lda11.score(Zt.reshape(-1, 11), np.repeat(np.arange(L), m_)))
        # ---- instrument pipeline (with nulls)
        ctx = P.Ctx(Idev, Itest, A)
        out = P.run(ctx, Zd, Zt, nperm=NPERM, seed=layer)
        R["instr"] = {k: {kk: np.asarray(v).tolist() for kk, v in r.items()} for k, r in out.items()}
        out_nc = P.run(ctx, Zd, Zt, sets=(("e_tokc",), ("B_bag",), ("B_pre",)), nperm=0, covs=False)
        R["instr_nocov"] = {k: {kk: np.asarray(v).tolist() for kk, v in r.items()} for k, r in out_nc.items()}
        R["groups"] = {nm: groups_fit(ctx, Zd, Zt, nm) for nm in ("e_tokc", "B_bag", "B_pre")}
        # ---- script-level instruments (barycentric, dev and devtest separately)
        R["script_dev"] = script_tests(F.bary(Zd, A), Idev)
        R["script_test"] = script_tests(W, Itest)
        R["surface_sharing_test"] = surface_sharing(Zt64, Itest)
        # ---- packing
        M = F.metric(F.residualize(F.centre(Zd), ctx.Zc["dev"]))
        De = np.sqrt(((A[:, None] - A[None]) ** 2).sum(2))
        dA = A[:, None] - A[None]
        Dm = np.sqrt(np.einsum("ijk,kl,ijl->ij", dA, M, dA))
        pk = {}
        rng = np.random.RandomState(0)
        for dn, Dx in (("euclid", De), ("mahal", Dm)):
            for tn, S in TEXT.items():
                r0, p0 = F.mantel(-Dx, S, nperm=2000, rng=rng)
                r1, p1 = F.mantel(-Dx, S, nperm=2000, rng=rng, ctrl=same_script)
                pk[f"{dn}~{tn}"] = (r0, p0, r1, p1)
            r2, p2 = F.mantel(-Dx, same_script, nperm=2000, rng=rng)
            pk[f"{dn}~same_script"] = (r2, p2)
        R["packing"] = pk
        R["nearest"] = {F.LANGS[i]: F.LANGS[int(np.argsort(De[i])[1])] for i in range(L)}
        R["seconds"] = time.time() - t0
        results[f"L{layer}_{variant}"] = R
        ins = R["instr"]
        print(f"L{layer:2d} {variant:5s} disp/edge={R['disp_over_edge']:.3f} e_inspan={R['e_inspan']:.3f} "
              f"lda_acc={R['lda_acc']:.3f} | " + " ".join(
                  f"{k}:g={ins[k]['g_dev'][0]:.3f}/{ins[k]['g_test'][0]:.3f},rho={ins[k]['rho'][0]:.3f},"
                  f"z={ins[k]['z_shuf']:.1f}/{ins[k]['z_jperm']:.1f}" for k in ("e_tokc", "e_char", "B_bag", "B_pre"))
              + f" | joint bag,pre g={np.round(ins['B_bag+B_pre']['g_dev'], 3)} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(os.path.join(F.HERE, f"res_{'_'.join(map(str, layers))}.json"), "w"), indent=1)
