from plant_disease.data import grouped_split, image_key, leaf_id


def test_image_key_matches_leaf_map_format():
    assert image_key("0031da2a-8edd-468f-a8b1-106657717a32___RS_HL 0105.JPG") == "rs_hl 0105"
    assert image_key("abc___GH_HL Leaf 259.1.JPG") == "gh_hl leaf 259.1"
    assert image_key("abc___UF.GRC_BS_Lab Leaf 0123 copy 2.jpg") == "uf.grc_bs_lab leaf 0123"


def test_leaf_id_picks_entry_for_the_right_class():
    leaf_map = {"rs_hl 0105": ["Potato___healthy:::1.0", "Tomato___healthy:::7.0"]}
    assert leaf_id("Tomato___healthy", "x___RS_HL 0105.JPG", leaf_map) == "Tomato___healthy:::7.0"


def test_leaf_id_falls_back_to_singleton_group():
    gid = leaf_id("Tomato___healthy", "x___unknown 1.JPG", {})
    assert "single" in gid


def _toy_dataset():
    files, groups = [], []
    for g in range(60):
        for k in range(1 + g % 4):  # groups of 1-4 images
            files.append(f"img_{g}_{k}.jpg")
            groups.append(f"leaf_{g}")
    return files, groups


def test_grouped_split_keeps_each_leaf_in_one_split():
    files, groups = _toy_dataset()
    out = grouped_split(files, groups, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)

    group_of = dict(zip(files, groups))
    seen = {}
    for split, names in out.items():
        for n in names:
            assert seen.setdefault(group_of[n], split) == split

    assert sorted(sum(out.values(), [])) == sorted(files)


def test_grouped_split_sizes_are_close_to_ratios():
    files, groups = _toy_dataset()
    out = grouped_split(files, groups, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    n = len(files)
    for split, ratio in {"val": 0.15, "test": 0.15}.items():
        # can overshoot by at most one group (max 4 images)
        assert ratio * n <= len(out[split]) <= ratio * n + 4


def test_grouped_split_is_deterministic():
    files, groups = _toy_dataset()
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert grouped_split(files, groups, ratios, seed=1) == grouped_split(files, groups, ratios, seed=1)
