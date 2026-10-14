# Manuscript sources

`main.tex` uses the Elsevier `elsarticle` class. Every number in the text is written as `\val{key}`; its value comes from `generated/numbers.tex`, which `scripts/r7_paper_assets.py` computes directly from `results/*/reversal/*.csv`. Tables in `generated/tables/` and figures in `generated/figs/` are produced by the same script. A missing key is rendered as a red `??key??`.

```bash
python scripts/r7_paper_assets.py      # from the repository root
cd paper && latexmk -pdf main.tex
```
