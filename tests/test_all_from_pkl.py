import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.api.compute_all import compute_all_from_pkl


def main():
    pkl_path = "data/Dataset1_merfish_u2os_data_dict.pkl"

    # Bento 这边你已经跑通过，可以不传参数；需要的话在这里加
    bento_kwargs = {}

    # Sprawl 这边很慢，先用你现在跑的配置
    sprawl_kwargs = {
        "metrics": ("peripheral", "central", "punctate", "radial"),
        "processes": 1,
        "num_iterations": 200,  # 你如果正在用1000，这里也可以改成1000（很慢）
        "num_pairs": 4,
    }

    all_df = compute_all_from_pkl(
        pkl_path,
        bento_kwargs=bento_kwargs,
        sprawl_kwargs=sprawl_kwargs,
        out_path=None,   # 想写盘就填路径，比如 "data/all_features.parquet"
    )

    print("shape:", all_df.shape)
    print("columns:", list(all_df.columns))
    print(all_df.head())

    # 期待至少包含 bento 13 + sprawl 4 => 17 列（如果 sprawl 未算完/缺失会少）
    # 这里不写死等于17，避免你先跑快模式时失败；等你确认完整后再改成 assert == 17
    assert all_df.shape[1] >= 13, "Should contain at least Bento 13 features."
    print("✅ compute_all smoke test passed.")


if __name__ == "__main__":
    main()