"""Emit the markdown tables used in report.md from the CSV/JSON outputs."""
import pandas as pd, json, warnings
warnings.filterwarnings("ignore")


def md(df, idx=True):
    df = df.reset_index() if idx else df
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join((str(int(v)) if isinstance(v, float) and v.is_integer() and v > 1 else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |" for r in df.values)


S = pd.read_csv('by_layer.csv').set_index('layer')
F = pd.read_csv('onefactor.csv').set_index('layer')
D = pd.read_csv('depth.csv').set_index('layer')
t = pd.DataFrame({'f_lang': S.f_lang_dev, 'lang/content': S.lang_over_content_dev, 'shift share of offset': S.inv_frac_dev,
                  'PR(centroids)': S.PR_a, 'lang E in top64 content PCs': S.a_in_content64, 'content E in top64': S.b_in_content64,
                  'max cos(lang, content11)': S.cos_lang_content11_max, 'perlang-linear share of offset resid': 1 - S.hub_resid_ridge_perlang / S.hub_resid_shift,
                  'W_U frac lang': S.W_a, 'W_U frac content': S.W_b, '1-factor expl.': F.frac_offdiag_var_explained,
                  'shape sim vs L14': D.shape_vs_L14, 'ambient cos to next layer': D.next_ident})
print(md(t.round(3)), "\n")
t2 = pd.DataFrame({'rho id (amb)': S.rho_identity_amb, 'rho shift (amb)': S.rho_shift_amb, 'P@1 shift (amb)': S.p1_shift_amb,
                   'rho shift k64': S.rho_shift_trunc_k64, 'rho polar-id k64': S.rho_polar_id_k64, 'rho proc k64': S.rho_proc_k64,
                   'rho ridge k64': S.rho_ridge_k64, 'P@1 polar-id k256': S.p1_polar_id_k256, 'P@1 proc k256': S.p1_proc_k256,
                   'tr(R_proc^T R_id)/k k64': S.proc_vs_identity_k64, 'same k256': S.proc_vs_identity_k256})
print(md(t2.round(3)), "\n")
c = json.load(open('compose.json'))
rows = []
for l in (4, 14, 24):
    for sc in ('real', 'shift', 'gauge'):
        o = c[f'L{l}_{sc}']
        rows.append(dict(layer=l, data={'real': 'real', 'shift': 'synth: ambient identity', 'gauge': 'synth: per-lang rotation'}[sc],
                         c_proc_k64=o['proc_k64']['c'], c_proc_k256=o['proc_k256']['c'], c_ridge_k64=o['ridge_k64']['c'],
                         rho_proc_k64=o['proc_k64']['rho'], rho_polarid_k64=o['polar_id_k64']['rho'],
                         proc_vs_id_k64=o['proc_vs_identity_k64'], proc_vs_id_k256=o['proc_vs_identity_k256']))
print(md(pd.DataFrame(rows).round(3), idx=False))
