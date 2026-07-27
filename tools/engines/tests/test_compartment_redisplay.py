"""Small tests for compartment redisplay input handling."""

from pathlib import Path
import pickle

import pandas as pd

from tools.engines.compartment_redisplay import _load_redisplay_inputs


def test_load_redisplay_inputs_accepts_data_df_metadata(tmp_path: Path):
    """Redisplay should reuse data_df cell metadata when coordinates is absent."""

    pkl_path = tmp_path / "bundle.pkl"
    bundle = {
        "data_df": pd.DataFrame(
            {
                "cell": ["c1", "c1", "c2"],
                "batch": ["batch_1", "batch_1", "batch_2"],
            }
        ),
        "cell_boundary": {"c1": "cell-boundary"},
        "nuclear_boundary": {"c1": "nuclear-boundary"},
    }
    with pkl_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    cell_display_map, cell_boundary, nuclear_boundary = _load_redisplay_inputs(
        pkl_path,
        "batch",
    )

    assert cell_display_map.to_dict("records") == [
        {"cell": "c1", "batch": "batch_1"},
        {"cell": "c2", "batch": "batch_2"},
    ]
    assert cell_boundary == {"c1": "cell-boundary"}
    assert nuclear_boundary == {"c1": "nuclear-boundary"}
