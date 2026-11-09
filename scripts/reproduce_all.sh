#!/usr/bin/env bash
# Reproduces every table and figure of the paper from ONE configuration (the defaults of
# rdanet/config.py = Table A.1). Each command changes only the fields that define its table row.
# Subject-dependent (SD) results: seeds 0-4 (mean over seeds). LOSO results: seed 0, one run per fold.
#
#   bash scripts/reproduce_all.sh            # run from the repository root
#
# Data directories default to data/BCICIV_2a, data/BCICIV_2b and data/HGD.
# Set D2A / D2B / DHGD to use other locations.
set -euo pipefail
cd "$(dirname "$0")/.."
D2A=${D2A:-data/BCICIV_2a}
D2B=${D2B:-data/BCICIV_2b}
DHGD=${DHGD:-data/HGD}
R="python main.py --out_dir runs"
A="--dataset bci2a --data_dir $D2A"

# ---------------- Tables 1 and 2: SD, BCI-IV 2a; paired tests ------------------------------
$R $A --protocol sd --tag T1_sd_bci2a
python scripts/significance.py baselines runs/T1_sd_bci2a

# ---------------- Tables 3 and 4: SD, BCI-IV 2b and HGD ------------------------------------
$R --dataset bci2b --data_dir $D2B --protocol sd --tag T3_sd_bci2b
$R --dataset hgd   --data_dir $DHGD --protocol sd --tag T4_sd_hgd

# ---------------- Table 5: LOSO, BCI-IV 2a (seed 0) -----------------------------------------
# LOSO-SI: strict subject-independent (no target data; no DA module).
#          Tested on target T+E (Table 5, top) and on target E (Table 9, "w/o DA").
$R $A --protocol loso_si  --seeds 0 --tag T5_loso_si
# LOSO-UDA: transductive; unlabelled target session T for adaptation, tested on target session E.
$R $A --protocol loso_uda --seeds 0 --tag T5_loso_uda

# ---------------- Table 9: effect of DA (target session E) ----------------------------------
python scripts/significance.py da runs/T5_loso_si runs/T5_loso_uda

# ---------------- Table 6: complexity and latency --------------------------------------------
python scripts/complexity.py --attn_impl banded --seq_scan
python scripts/complexity.py --attn_impl masked

# ---------------- Table 7: numerical stability (SD, BCI-IV 2a) -------------------------------
for REG in none tikhonov dual; do
  for MAP in log_euclidean log_cholesky; do
    [ "$REG $MAP" = "dual log_cholesky" ] && continue        # = full model (T1_sd_bci2a)
    $R $A --protocol sd --cov_reg $REG --spd_map $MAP --tag T7_${REG}_${MAP}
  done
done
python scripts/condition_numbers.py --dataset bci2a --data_dir $D2A

# ---------------- Table 8: SD ablations on the three datasets --------------------------------
for DS in "bci2a $D2A" "bci2b $D2B" "hgd $DHGD"; do
  set -- $DS
  for ABL in "use_riemann false" "use_transformer false" "use_cnn false" "fusion concat" "use_aug false"; do
    K=${ABL%% *}; V=${ABL##* }
    $R --dataset $1 --data_dir $2 --protocol sd --$K $V --tag T8_${1}_${K}_${V}
  done
done

# ---------------- Hyperparameter selection (Section 4.3) --------------------------------------
# Selection criterion: mean validation loss, reported in runs/<tag>/summary.txt.
for V in 1e-5 1e-3;     do $R $A --protocol sd --lambda_reg $V --tag S_lreg$V; done
for V in 0.1 0.5 1.0;   do $R $A --protocol loso_uda --seeds 0 --lambda_da $V --tag S_lda$V; done
for V in 20 30 50 60;   do $R $A --protocol sd --emb_size $V --tag S_emb$V;   done
for V in 2 4 8;         do $R $A --protocol sd --depth $V    --tag S_depth$V; done
for V in 4 5 8 20;      do $R $A --protocol sd --heads $V    --tag S_heads$V; done
for V in 3 4 6 7;       do $R $A --protocol sd --window $V   --tag S_win$V;   done
