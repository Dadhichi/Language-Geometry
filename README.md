# lang-geom
Geometry of cross-language maps inside multilingual LLMs: shift vs gauge vs pair-specific structure by depth.
See `CLAUDE.md` for the full project context, design decisions and next steps.

```
python fit.py --selftest                      # must pass before/after editing fit.py
python extract.py --model Qwen/Qwen2.5-7B --tag qwen25_7b --out /content/drive/MyDrive/lang_geom
python fit.py --data /content/drive/MyDrive/lang_geom/qwen25_7b --pooling mean \
    --fit_split dev --test_splits devtest --k 64,128,256 --pca per_lang --device cuda \
    --out results/qwen25_7b_mean
```
