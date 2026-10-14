#!/usr/bin/env bash
# 判别关系反转分析全流程（可中断续跑，已完成的部分自动跳过）
# 用法（项目根目录）: bash run_reversal.sh        N_JOBS=16 bash run_reversal.sh 限制并行数
# 协议：dos_v1（主）、binary_v1（对照）；若项目根目录有 NF-*-v3 四个目录，再加 dos_v3（RQ4）
set -euo pipefail
cd "$(dirname "$0")"
NJ=${N_JOBS:--1}
mkdir -p results/logs
ts() { date "+%H:%M:%S"; }

echo "[$(ts)] 单元测试"
python -m pytest -q tests 2>&1 | tee results/logs/reversal_00_tests.log

CFGS="config.yaml configs/binary.yaml"
if [ -d NF-UNSW-NB15-v3 ] && [ -d NF-CICIDS2018-v3 ] && [ -d NF-ToN-IoT-v3 ] && [ -d NF-BoT-IoT-v3 ]; then
  CFGS="${CFGS} configs/dos_v3.yaml"
else
  echo "[$(ts)] 未发现 NF-*-v3 四个目录，跳过 NFv3 复现"
fi

for CFG in ${CFGS}; do
  RUN=$(python -c "import sys; sys.path.insert(0,'.'); from src.common import load_config; print(load_config('${CFG}')['run_name'])")
  echo "[$(ts)] ===== 协议 ${RUN}（${CFG}） ====="
  if [ ! -f "data/processed/${RUN}/meta.json" ]; then
    echo "[$(ts)] 预处理 ${RUN}"
    python scripts/01_preprocess.py --config "${CFG}" 2>&1 | tee "results/logs/reversal_01_preprocess_${RUN}.log"
  fi
  echo "[$(ts)] r1 迁移矩阵 ${RUN}"
  python scripts/r1_transfer.py --config "${CFG}" --n-jobs "${NJ}" 2>&1 | tee "results/logs/reversal_r1_${RUN}.log"
  echo "[$(ts)] r2 特征级分析 ${RUN}"
  python scripts/r2_feature_reversal.py --config "${CFG}" 2>&1 | tee "results/logs/reversal_r2_${RUN}.log"
  echo "[$(ts)] r5 方向对齐诊断 ${RUN}"
  python scripts/r5_alignment.py --config "${CFG}" --n-jobs "${NJ}" 2>&1 | tee "results/logs/reversal_r5_${RUN}.log"
  python scripts/r5_alignment.py --config "${CFG}" --n-jobs "${NJ}" --sens 2>&1 | tee "results/logs/reversal_r5sens_${RUN}.log"
  echo "[$(ts)] r6 少量标签 ${RUN}"
  python scripts/r6_few_label.py --config "${CFG}" --n-jobs "${NJ}" 2>&1 | tee "results/logs/reversal_r6_${RUN}.log"
done

if [ ! -f results/audit.csv ]; then
  echo "[$(ts)] r3 数据审计（NFv2，读一遍全部原始 CSV）"
  python scripts/r3_audit.py 2>&1 | tee results/logs/reversal_r3_audit.log
fi

echo "[$(ts)] r4 汇总报告与图"
python scripts/r4_report.py 2>&1 | tee results/logs/reversal_r4_report.log
echo "[$(ts)] r7 论文数字、表格与图（paper/generated/）"
python scripts/r7_paper_assets.py 2>&1 | tee results/logs/reversal_r7_paper.log
if command -v latexmk >/dev/null 2>&1; then
  (cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex) && echo "[$(ts)] 论文已编译：paper/main.pdf"
fi
echo "[$(ts)] 完成：results/reversal_summary/report.md"
