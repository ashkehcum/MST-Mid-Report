\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


import os
import re
import ast
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE        = os.path.join(_SCRIPT_DIR, "MST_Data")
OUT_DIR   = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = {
    "item_only": {
        "data_dir":        os.path.join(BASE, "item_only",     "item_only_data"),
        "obj_bins_file":   os.path.join(BASE, "item_only",     "Set6 bins.txt"),
        "scene_bins_file": os.path.join(BASE, "item_only",     "SetScC bins.txt"),
        "scenes_map_file": os.path.join(BASE, "item_only",     "scenes_mapping.txt"),
        "has_scenes":      True,
    },
    "both": {
        "data_dir":        os.path.join(BASE, "Both_item_task","both_data"),
        "obj_bins_file":   os.path.join(BASE, "Both_item_task","Set6 bins.txt"),
        "scene_bins_file": os.path.join(BASE, "Both_item_task","SetScC bins.txt"),
        "scenes_map_file": os.path.join(BASE, "item_only",     "scenes_mapping.txt"),
        "has_scenes":      True,
    },
    "task_only": {
        "data_dir":        os.path.join(BASE, "task_only",     "task_only_data"),
        "obj_bins_file":   os.path.join(BASE, "task_only",     "Set6 bins_ob.txt"),
        "scene_bins_file": None,
        "scenes_map_file": None,
        "has_scenes":      False,
    },
}


def norm_path(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def load_bins(filepath: str) -> dict:
\
\
\

    if filepath is None or not os.path.exists(filepath):
        return {}
    bins = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                bins[parts[0]] = int(parts[1])
    return bins


def load_scenes_mapping(filepath: str) -> dict:
    if filepath is None or not os.path.exists(filepath):
        return {}
    mapping = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[norm_path(parts[0])] = parts[1].lower()
    return mapping


def get_lure_bin(image_path: str, obj_bins: dict, scene_bins: dict) -> int | None:
\
\
\
\
\

    p = norm_path(image_path)
    m = re.search(r"(\d+)[ab]\.jpg", p, re.IGNORECASE)
    if not m:
        return None
    num_str = str(int(m.group(1)))
    if p.startswith("Objects") or p.startswith("objects"):
        return obj_bins.get(num_str)
    elif p.startswith("Scenes") or p.startswith("scenes"):
        return scene_bins.get(num_str) if scene_bins else None
    return None


def parse_task_csv(filepath: str, scenes_map: dict) -> pd.DataFrame | None:
\
\
\
\
\
\
\
\

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  [WARN] Could not read {filepath}: {e}")
        return None


    mask = (
        df["image_path"].notna()
        & df["image_path"].astype(str).str.contains(r"Objects|Scenes", case=False, regex=True, na=False)
    )
    trials = df[mask].copy().reset_index(drop=True)

    if len(trials) == 0:
        return None

    trials["norm_path"] = trials["image_path"].apply(norm_path)


    trials["event_num"]     = trials.index // 7
    trials["pos_in_event"]  = trials.index % 7
    trials["boundary_position"] = trials["pos_in_event"].map(
        {0: "post", 6: "pre"}
    ).fillna("mid")


    rt_vals  = []
    key_vals = []
    for _, row in trials.iterrows():
        if pd.notna(row.get("key_resp_9.keys")) and pd.notna(row.get("key_resp_9.rt")):
            rt_vals.append(float(row["key_resp_9.rt"]))
            key_vals.append(str(row["key_resp_9.keys"]).strip().lower())
        elif pd.notna(row.get("key_resp_8.keys")) and pd.notna(row.get("key_resp_8.rt")):
            rt_vals.append(3.0 + float(row["key_resp_8.rt"]))
            key_vals.append(str(row["key_resp_8.keys"]).strip().lower())
        else:
            rt_vals.append(np.nan)
            key_vals.append(np.nan)

    trials["rt"]          = rt_vals
    trials["key_pressed"] = key_vals


    def get_correct_key(np_path):
        return scenes_map.get(np_path, np.nan)

    trials["stim_category"] = trials["norm_path"].apply(
        lambda p: "Scenes" if p.startswith("Scenes") else "Objects"
    )
    trials["correct_key"] = trials["norm_path"].apply(get_correct_key)
    trials["is_correct"]  = trials.apply(
        lambda r: (
            (r["key_pressed"] == r["correct_key"])
            if pd.notna(r["correct_key"]) and pd.notna(r["key_pressed"])
            else np.nan
        ),
        axis=1,
    )


    if "encoding_task_accuracy" in df.columns:
        acc_vals = df["encoding_task_accuracy"].dropna()
        csv_accuracy = float(acc_vals.iloc[-1]) if len(acc_vals) > 0 else np.nan
    else:
        csv_accuracy = np.nan
    trials["csv_encoding_accuracy"] = csv_accuracy

    return trials[
        [
            "image_path", "norm_path", "stim_category",
            "event_num", "pos_in_event", "boundary_position",
            "rt", "key_pressed", "correct_key", "is_correct",
            "csv_encoding_accuracy",
        ]
    ]


def parse_test_csv(filepath: str, obj_bins: dict, scene_bins: dict) -> pd.DataFrame | None:
\
\
\
\
\
\

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  [WARN] Could not read {filepath}: {e}")
        return None

    mask = df["image_path"].notna()
    trials = df[mask].copy().reset_index(drop=True)

    if len(trials) == 0:
        return None

    trials["norm_path"] = trials["image_path"].apply(norm_path)


    def classify_stim(p):
        p = norm_path(p)
        if p.lower().startswith("foils"):
            return "foil"
        if p.endswith("a.jpg"):
            return "target"
        if p.endswith("b.jpg"):
            return "lure"
        return "other"

    trials["stim_type"] = trials["image_path"].apply(classify_stim)


    trials["response"] = trials["key_resp_3.keys"].astype(str).str.strip().str.lower()
    trials["rt"]       = pd.to_numeric(trials["key_resp_3.rt"], errors="coerce")


    trials["position"] = trials["position_of_stimuli"].fillna("none").str.strip().str.lower()


    trials["lure_bin"] = trials.apply(
        lambda r: get_lure_bin(r["image_path"], obj_bins, scene_bins)
        if r["stim_type"] == "lure"
        else np.nan,
        axis=1,
    )

    return trials[
        [
            "image_path", "norm_path", "stim_type",
            "position", "response", "rt", "lure_bin",
        ]
    ]


def compute_metrics(test_df: pd.DataFrame) -> dict:
\
\
\
\
\
\
\
\
\
\
\
\
\

    from scipy.stats import norm as _norm

    def _dprime(p_hit, p_fa):

        if np.isnan(p_hit) or np.isnan(p_fa):
            return np.nan
        return float(_norm.ppf(np.clip(p_hit, 0.01, 0.99)) -
                     _norm.ppf(np.clip(p_fa,  0.01, 0.99)))

    metrics = {}

    foils   = test_df[test_df["stim_type"] == "foil"]
    targets = test_df[test_df["stim_type"] == "target"]
    lures   = test_df[test_df["stim_type"] == "lure"]


    p_old_foil  = (foils["response"] == "o").mean() if len(foils) > 0 else np.nan
    p_sim_foil  = (foils["response"] == "s").mean() if len(foils) > 0 else np.nan


    p_old_target = (targets["response"] == "o").mean() if len(targets) > 0 else np.nan
    p_sim_lure   = (lures["response"]   == "s").mean() if len(lures)   > 0 else np.nan

    metrics["n_foils"]   = len(foils)
    metrics["n_targets"] = len(targets)
    metrics["n_lures"]   = len(lures)

    metrics["p_old_foil"]    = p_old_foil
    metrics["p_sim_foil"]    = p_sim_foil
    metrics["p_old_target"]  = p_old_target
    metrics["p_sim_lure"]    = p_sim_lure

    metrics["REC_overall"] = p_old_target - p_old_foil
    metrics["dprime_overall"] = _dprime(p_old_target, p_old_foil)
    metrics["LDI_overall"] = p_sim_lure   - p_sim_foil


    for pos in ["pre", "mid", "post"]:
        tgt_pos  = targets[targets["position"] == pos]
        lure_pos = lures[lures["position"] == pos]

        p_old_t = (tgt_pos["response"]  == "o").mean() if len(tgt_pos)  > 0 else np.nan
        p_sim_l = (lure_pos["response"] == "s").mean() if len(lure_pos) > 0 else np.nan

        metrics[f"n_targets_{pos}"]    = len(tgt_pos)
        metrics[f"n_lures_{pos}"]      = len(lure_pos)
        metrics[f"p_old_target_{pos}"] = p_old_t
        metrics[f"p_sim_lure_{pos}"]   = p_sim_l
        metrics[f"REC_{pos}"]          = p_old_t - p_old_foil
        metrics[f"dprime_{pos}"]       = _dprime(p_old_t, p_old_foil)
        metrics[f"LDI_{pos}"]          = p_sim_l - p_sim_foil


    for b in range(1, 6):
        lure_b  = lures[lures["lure_bin"] == b]
        p_sim_b = (lure_b["response"] == "s").mean() if len(lure_b) > 0 else np.nan
        metrics[f"n_lures_bin{b}"]  = len(lure_b)
        metrics[f"p_sim_bin{b}"]    = p_sim_b
        metrics[f"LDI_bin{b}"]      = p_sim_b - p_sim_foil

    return metrics


def compute_encoding_rt(task_df: pd.DataFrame) -> dict:

    rt_metrics = {}
    rt_metrics["encoding_rt_overall"] = task_df["rt"].mean()
    for pos in ["pre", "mid", "post"]:
        rt_metrics[f"encoding_rt_{pos}"] = task_df[task_df["boundary_position"] == pos]["rt"].mean()
    return rt_metrics


def find_latest_file(files: list[str], pid: str, kind: str) -> str | None:
\
\
\
\

    candidates = sorted([f for f in files if f.startswith(pid) and f"MST_{kind}" in f])
    return candidates[-1] if candidates else None


def process_group(group_name: str, cfg: dict) -> pd.DataFrame:
\
\
\

    print(f"\n{'='*60}")
    print(f"  Processing group: {group_name.upper()}")
    print(f"{'='*60}")

    data_dir    = cfg["data_dir"]
    obj_bins    = load_bins(cfg["obj_bins_file"])
    scene_bins  = load_bins(cfg["scene_bins_file"]) if cfg["scene_bins_file"] else {}
    scenes_map  = load_scenes_mapping(cfg["scenes_map_file"])

    all_files = sorted(os.listdir(data_dir))
    pids = sorted(set(f[:5] for f in all_files if f.endswith(".csv")))

    print(f"  Found {len(pids)} unique participant IDs")

    rows = []
    for pid in pids:
        task_file = find_latest_file(all_files, pid, "task")
        test_file = find_latest_file(all_files, pid, "test")

        if task_file is None or test_file is None:
            print(f"  [SKIP] {pid}: missing {'task' if task_file is None else 'test'} file")
            continue

        task_path = os.path.join(data_dir, task_file)
        test_path = os.path.join(data_dir, test_file)

        task_df = parse_task_csv(task_path, scenes_map)
        test_df = parse_test_csv(test_path, obj_bins, scene_bins)

        if task_df is None or test_df is None:
            print(f"  [SKIP] {pid}: empty data")
            continue

        row = {
            "group":          group_name,
            "participant_id": pid,
            "task_file":      task_file,
            "test_file":      test_file,
            "n_encoding_trials": len(task_df),
            "csv_encoding_accuracy": task_df["csv_encoding_accuracy"].iloc[0],
        }


        scene_trials = task_df[(task_df["stim_category"] == "Scenes") & task_df["is_correct"].notna()]
        row["n_scenes_encoded"]         = len(scene_trials)
        row["scene_encoding_accuracy"]  = scene_trials["is_correct"].mean() if len(scene_trials) > 0 else np.nan


        row.update(compute_encoding_rt(task_df))


        row.update(compute_metrics(test_df))

        rows.append(row)
        print(
            f"  {pid}: REC={row.get('REC_overall', np.nan):.3f}  "
            f"LDI={row.get('LDI_overall', np.nan):.3f}  "
            f"enc_acc(csv)={row.get('csv_encoding_accuracy', np.nan):.3f}"
        )

    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for group_name, cfg in GROUPS.items():
        df = process_group(group_name, cfg)
        all_dfs.append(df)

    participant_df = pd.concat(all_dfs, ignore_index=True)


    out_path = os.path.join(OUT_DIR, "participant_summary.csv")
    participant_df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}  ({len(participant_df)} participants)")


    numeric_cols = participant_df.select_dtypes(include=[np.number]).columns.tolist()
    group_summary = (
        participant_df.groupby("group")[numeric_cols]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    out_path2 = os.path.join(OUT_DIR, "group_summary.csv")
    group_summary.to_csv(out_path2)
    print(f"[SAVED] {out_path2}")


    rt_cols = ["group", "participant_id",
               "encoding_rt_overall", "encoding_rt_pre",
               "encoding_rt_mid",     "encoding_rt_post"]
    enc_rt = participant_df[[c for c in rt_cols if c in participant_df.columns]].copy()
    enc_rt_long = enc_rt.melt(
        id_vars=["group", "participant_id"],
        var_name="position_label", value_name="mean_rt"
    )
    enc_rt_long["position"] = enc_rt_long["position_label"].str.replace("encoding_rt_", "")
    enc_rt_summary = (
        enc_rt_long.groupby(["group", "position"])["mean_rt"]
        .agg(mean="mean", sd="std", n="count")
        .round(4)
        .reset_index()
    )
    out_path3 = os.path.join(OUT_DIR, "encoding_rt_summary.csv")
    enc_rt_summary.to_csv(out_path3, index=False)
    print(f"[SAVED] {out_path3}")


    ldi_bin_cols = ["group", "participant_id"] + [f"LDI_bin{b}" for b in range(1, 6)]
    ldi_bin = participant_df[[c for c in ldi_bin_cols if c in participant_df.columns]].copy()
    ldi_bin_long = ldi_bin.melt(
        id_vars=["group", "participant_id"],
        var_name="bin_label", value_name="LDI"
    )
    ldi_bin_long["lure_bin"] = ldi_bin_long["bin_label"].str.extract(r"(\d)").astype(int)
    ldi_bin_summary = (
        ldi_bin_long.groupby(["group", "lure_bin"])["LDI"]
        .agg(mean_LDI="mean", sd_LDI="std", n="count")
        .round(4)
        .reset_index()
    )
    out_path4 = os.path.join(OUT_DIR, "ldi_by_bin.csv")
    ldi_bin_summary.to_csv(out_path4, index=False)
    print(f"[SAVED] {out_path4}")


    print("\n" + "="*70)
    print("GROUP-LEVEL MEANS  (key metrics)")
    print("="*70)
    key_metrics = [
        "REC_overall", "LDI_overall",
        "REC_pre", "REC_mid", "REC_post",
        "LDI_pre", "LDI_mid", "LDI_post",
        "encoding_rt_pre", "encoding_rt_mid", "encoding_rt_post",
        "csv_encoding_accuracy",
    ]
    for grp, grp_df in participant_df.groupby("group"):
        print(f"\n  {grp.upper()}  (n={len(grp_df)})")
        for col in key_metrics:
            if col in grp_df.columns:
                m = grp_df[col].mean()
                s = grp_df[col].std()
                print(f"    {col:<35}  M={m:.4f}  SD={s:.4f}")


if __name__ == "__main__":
    main()
