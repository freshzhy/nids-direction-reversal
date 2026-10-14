import numpy as np
import pandas as pd

from src import preprocess as P

FE = ["IN_BYTES", "PROTOCOL", "MIN_TTL"]


def _csv(tmp_path):
    rows = [
        # ip_src, sport, ip_dst, dport, IN_BYTES, PROTOCOL, MIN_TTL, Label, Attack
        ("1.1.1.1", 1, "2.2.2.2", 80, 100, 6, 64, 0, "Benign"),
        ("1.1.1.9", 7, "2.2.2.2", 81, 100, 6, 64, 0, "Benign"),   # dup after dropping ids
        ("1.1.1.1", 2, "2.2.2.2", 80, 100, 6, 64, 1, "DoS"),      # same features, other label
        ("1.1.1.1", 3, "2.2.2.2", 80, "inf", 6, 64, 1, "DoS"),    # inf -> dropped
        ("1.1.1.1", 4, "2.2.2.2", 80, 5, 17, 32, 1, "DoS"),
        ("1.1.1.1", 5, "2.2.2.2", 80, 7, 17, 32, 1, "Scan"),
        ("1.1.1.1", 6, "2.2.2.2", 80, -3, 17, 32, 0, "Benign"),   # negative -> clipped in log1p
        ("1.1.1.1", 8, "2.2.2.2", 80, "", 17, 32, 0, "Benign"),   # NaN -> dropped
    ]
    df = pd.DataFrame(rows, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR",
                                     "L4_DST_PORT"] + FE + ["Label", "Attack"])
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    return p


def test_scan_dedup_sample_extract(tmp_path):
    p = _csv(tmp_path)
    sc = P.scan(p, FE, "Label", "Attack", chunksize=3, log=lambda m: None)
    assert sc["n_total"] == 8 and sc["n_bad"] == 2
    ds = P.dedup_and_pool(sc, None, pool_cap=10, seed=0)
    st = ds["stats"]
    assert st["rows_after_clean"] == 6
    assert st["rows_dropped_duplicate"] == 1
    assert st["rows_unique"] == 5 and st["benign_unique"] == 2 and st["positive_unique"] == 3
    assert st["label_conflict_groups"] == 1
    assert np.isclose(st["raw_positive_ratio"], 3 / 5)
    assert len(ds["pool_idx"]) == 5 and ds["pool_label"].sum() == 3
    # family task: only "DoS" positive, "Scan" excluded; pool capped per class
    fam = P.dedup_and_pool(sc, ["DoS"], pool_cap=1, seed=0)
    assert fam["stats"]["positive_unique"] == 2 and fam["stats"]["attack_excluded_not_in_task"] == 1
    assert len(fam["pool_idx"]) == 2 and fam["pool_label"].tolist().count(1) == 1
    import pytest
    with pytest.raises(ValueError):
        P.dedup_and_pool(sc, ["NoSuchAttack"], pool_cap=10, seed=0)
    parts = P.extract(p, FE, "Label", "Attack", 3, {"pool": ds["pool_idx"]}, log=lambda m: None)
    out = P.transform(parts["pool"], FE, ["IN_BYTES"], "Label", "Attack", label=ds["pool_label"])
    assert out["Label"].tolist() == ds["pool_label"].tolist()
    assert np.isfinite(out[FE].to_numpy()).all()
    assert out["IN_BYTES"].min() == 0.0                  # -3 clipped then log1p
    assert set(out["PROTOCOL"]) <= {6.0, 17.0}           # categorical untouched
    assert np.isclose(out["IN_BYTES"].max(), np.log1p(100))


def test_sample_positions_bal_and_raw():
    from src.common import sample_positions
    y = np.array([0] * 900 + [1] * 100, dtype=np.int8)
    rng = np.random.default_rng(0)
    b = sample_positions(y, "bal", 500, 0, 0.0, rng)
    assert y[b].sum() == 100 and len(b) == 200                 # limited by positives, stays 1:1
    r = sample_positions(y, "raw", 0, 400, 0.25, rng)
    assert len(r) == 400 and y[r].sum() == 100                 # exact ratio kept
    r2 = sample_positions(y, "raw", 0, 10000, 0.5, rng)       # shrinks to keep 0.5
    assert y[r2].sum() == 100 and len(r2) == 200
    a = sample_positions(y, "bal", 50, 0, 0, np.random.default_rng(1))
    c = sample_positions(y, "bal", 50, 0, 0, np.random.default_rng(2))
    assert set(a) != set(c)                                    # seeds give different samples
