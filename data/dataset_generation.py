from __future__ import annotations

from data_pipeline import CLEAN_DATASET_PATH, RAW_DATASET_PATH, ensure_dataset_exists


def main() -> None:
    df = ensure_dataset_exists()
    print(f"Clean dataset ready: {CLEAN_DATASET_PATH} | rows={len(df)}")
    print(f"Raw source dataset: {RAW_DATASET_PATH}")
    print(df.head(10))


if __name__ == "__main__":
    main()
