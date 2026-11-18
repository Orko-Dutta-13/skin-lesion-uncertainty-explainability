import numpy as np
import pandas as pd
import pytest

from reliable_derm.data.splits import assert_no_group_overlap, leakage_report, make_splits


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    rows = []
    p = np.array([0.35, 0.25, 0.12, 0.08, 0.08, 0.05, 0.04, 0.03])
    for g in range(900):                                  # 900 lesions with 1-4 images each
        c = int(rng.choice(8, p=p))
        src = str(rng.choice(["HAM", "BCN", "MSK_other"], p=[0.4, 0.4, 0.2]))
        for k in range(int(rng.integers(1, 5)) if src != "MSK_other" else 1):
            rows.append({"image": f"im{len(rows)}", "label": c, "source": src,
                         "lesion_id": f"L{g}" if src != "MSK_other" else np.nan})
    d = pd.DataFrame(rows)
    d["group"] = d["lesion_id"].where(d["lesion_id"].notna(), "img_" + d["image"])
    return make_splits(d, seed=1)


def test_grouped_split_has_no_group_overlap(df):
    assert_no_group_overlap(df, "split")
    assert_no_group_overlap(df, "split_shift")


def test_naive_split_actually_leaks(df):
    # the whole point of the ablation: an image-level split leaks lesions into val/test
    rep = leakage_report(df, "split_naive")
    assert rep["test"]["pct"] > 10
    assert leakage_report(df, "split")["test"]["n_images_sharing_group_with_train"] == 0


def test_every_row_assigned_and_proportions(df):
    assert set(df["split"]) == {"train", "val", "test"}
    frac = df["split"].value_counts(normalize=True)
    assert 0.66 < frac["train"] < 0.76 and 0.10 < frac["val"] < 0.19 and 0.10 < frac["test"] < 0.19


def test_all_classes_in_every_split(df):
    for col in ("split", "split_naive"):
        for s in ("train", "val", "test"):
            assert df[df[col] == s]["label"].nunique() == 8


def test_shift_split_holds_out_exactly_one_source(df):
    assert set(df.loc[df["split_shift"] == "shift_test", "source"]) == {"BCN"}
    assert not (df.loc[df["split_shift"] != "shift_test", "source"] == "BCN").any()


def test_missing_lesion_ids_are_singletons(df):
    miss = df[df["lesion_id"].isna()]
    assert miss["group"].is_unique


def test_ood_split_has_in_distribution_test_and_shift_test(df):
    assert set(df["split_ood"]) == {"train", "val", "test", "shift_test"}
    assert_no_group_overlap(df, "split_ood")
    assert set(df.loc[df["split_ood"] == "shift_test", "source"]) == {"BCN"}
    assert not (df.loc[df["split_ood"] != "shift_test", "source"] == "BCN").any()
    non_bcn = df[df["source"] != "BCN"]
    frac = non_bcn["split_ood"].value_counts(normalize=True)
    assert 0.65 < frac["train"] < 0.77 and 0.09 < frac["val"] < 0.2 and 0.09 < frac["test"] < 0.2
    for s in ("train", "val", "test"):
        assert df[df["split_ood"] == s]["label"].nunique() == 8


def test_ood_script_leaves_existing_columns_untouched(df, tmp_path):
    from reliable_derm.data import add_ood_split as mod

    old = df.drop(columns=["split_ood"])
    p = tmp_path / "master.csv"
    old.to_csv(p, index=False)
    mod.main(["--master-csv", str(p)])
    new = pd.read_csv(p)
    assert set(new.columns) == set(old.columns) | {"split_ood"}
    assert (new["split"] == old["split"].to_numpy()).all()
    assert (p.parent / "master.csv.bak").exists()
