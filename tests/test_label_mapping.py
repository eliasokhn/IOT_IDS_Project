"""
Tests for label_mapping.py
"""

import pandas as pd
from src.data.label_mapping import (
    apply_binary, apply_category, apply_fine,
    add_all_label_columns, get_class_names, get_label_col,
    ALL_LABELS, BINARY_CLASS_NAMES, CATEGORY_CLASS_NAMES, FINE_CLASS_NAMES,
    BINARY_MAP, CATEGORY_MAP, FINE_MAP,
)


class TestBinaryMapping:
    def test_benign_maps_to_zero(self):
        s = pd.Series(["BENIGN"])
        assert apply_binary(s).iloc[0] == 0

    def test_all_attacks_map_to_one(self):
        attacks = [l for l in ALL_LABELS if l != "BENIGN"]
        s = pd.Series(attacks)
        assert (apply_binary(s) == 1).all()

    def test_no_nan_in_binary(self):
        s = pd.Series(ALL_LABELS)
        result = apply_binary(s)
        assert result.notna().all()

    def test_binary_has_exactly_two_values(self):
        s = pd.Series(ALL_LABELS)
        assert set(apply_binary(s).unique()) == {0, 1}


class TestCategoryMapping:
    def test_benign_maps_to_zero(self):
        s = pd.Series(["BENIGN"])
        assert apply_category(s).iloc[0] == 0

    def test_ddos_maps_to_one(self):
        s = pd.Series(["DDOS-TCP_FLOOD", "DDOS-SYN_FLOOD"])
        assert (apply_category(s) == 1).all()

    def test_dos_maps_to_two(self):
        s = pd.Series(["DOS-TCP_FLOOD", "DOS-UDP_FLOOD"])
        assert (apply_category(s) == 2).all()

    def test_mirai_maps_to_three(self):
        s = pd.Series(["MIRAI-GREETH_FLOOD", "MIRAI-UDPPLAIN"])
        assert (apply_category(s) == 3).all()

    def test_bruteforce_maps_to_seven(self):
        s = pd.Series(["DICTIONARYBRUTEFORCE"])
        assert apply_category(s).iloc[0] == 7

    def test_all_labels_have_mapping(self):
        s = pd.Series(ALL_LABELS)
        result = apply_category(s)
        assert result.notna().all(), "Some labels have no category mapping"

    def test_exactly_eight_classes(self):
        s = pd.Series(ALL_LABELS)
        assert s.map(CATEGORY_MAP).nunique() == 8


class TestFineMapping:
    def test_all_labels_have_fine_mapping(self):
        s = pd.Series(ALL_LABELS)
        result = apply_fine(s)
        assert result.notna().all()

    def test_exactly_34_fine_classes(self):
        assert len(FINE_CLASS_NAMES) == 34

    def test_fine_indices_are_unique(self):
        assert len(set(FINE_MAP.values())) == len(FINE_MAP)


class TestAddAllLabelColumns:
    def test_adds_three_columns(self):
        df = pd.DataFrame({"label": ALL_LABELS})
        df_out = add_all_label_columns(df)
        assert "label_binary" in df_out.columns
        assert "label_category" in df_out.columns
        assert "label_fine" in df_out.columns

    def test_no_nan_in_any_label_column(self):
        df = pd.DataFrame({"label": ALL_LABELS})
        df_out = add_all_label_columns(df)
        assert df_out["label_binary"].notna().all()
        assert df_out["label_category"].notna().all()
        assert df_out["label_fine"].notna().all()

    def test_original_df_not_modified(self):
        df = pd.DataFrame({"label": ALL_LABELS})
        df_original_cols = list(df.columns)
        _ = add_all_label_columns(df)
        assert list(df.columns) == df_original_cols


class TestGetClassNames:
    def test_binary_returns_two_names(self):
        assert len(get_class_names("binary")) == 2

    def test_8class_returns_eight_names(self):
        assert len(get_class_names("8class")) == 8

    def test_34class_returns_34_names(self):
        assert len(get_class_names("34class")) == 34

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            get_class_names("invalid_task")


class TestGetLabelCol:
    def test_binary_col(self):
        assert get_label_col("binary") == "label_binary"

    def test_8class_col(self):
        assert get_label_col("8class") == "label_category"

    def test_34class_col(self):
        assert get_label_col("34class") == "label_fine"
