import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.engines.sprawl_adapter import compute_sprawl_scores_from_pkl

def main():
    out = compute_sprawl_scores_from_pkl(
        "data/Dataset1_merfish_u2os_data_dict.pkl",
        metrics=("peripheral", "central", "punctate", "radial"),
        processes=1,
        num_iterations=200,   # 想更准就改 1000，但会慢很多
        num_pairs=4,
    )
    print(out.shape)
    print(out.columns.tolist())
    print(out.head())

if __name__ == "__main__":
    main()