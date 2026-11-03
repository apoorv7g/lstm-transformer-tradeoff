"""Print parameter counts for every preset at several sequence lengths.

Usage (from repo root):  python -m src.models.summary
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.models import PRESET_NAMES, build_model


def main():
    print(f"{'preset':<20}{'T':>4}{'params':>12}")
    for preset in PRESET_NAMES:
        for t in (10, 20, 40):
            model = build_model(preset, num_frames=t)
            print(f"{preset:<20}{t:>4}{model.count_params():>12,}")


if __name__ == "__main__":
    main()
