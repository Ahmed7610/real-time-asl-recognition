from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_DIR / "data" / "dynamic_test"

INDEX_TIP_START = 8 * 3
INDEX_TIP_END = INDEX_TIP_START + 3


MISCLASSIFIED_NONE_FILES = [
    "ahmed_test_none_test_NONE_0001_20260711_142850_228508.npz",
    "ahmed_test_none_test_NONE_0002_20260711_142852_028134.npz",
    "ahmed_test_none_test_NONE_0004_20260711_142856_128534.npz",
    "ahmed_test_none_test_NONE_0005_20260711_142858_131942.npz",
    "ahmed_test_none_test_NONE_0006_20260711_142859_327708.npz",
    "ahmed_test_none_test_NONE_0007_20260711_142900_759009.npz",
    "ahmed_test_none_test_NONE_0008_20260711_142902_127723.npz",
    "ahmed_test_none_test_NONE_0009_20260711_142903_719017.npz",
]


def load_index_path(file_path):
    data = np.load(file_path)

    sequence = np.asarray(
        data["landmarks"],
        dtype=np.float32,
    )

    index_tip = sequence[
        :,
        INDEX_TIP_START:INDEX_TIP_END,
    ]

    return sequence, index_tip


def path_statistics(index_tip):
    xy = index_tip[:, :2]

    movement = np.diff(
        xy,
        axis=0,
    )

    total_path = float(
        np.linalg.norm(
            movement,
            axis=1,
        ).sum()
    )

    horizontal_range = float(
        xy[:, 0].max() - xy[:, 0].min()
    )

    vertical_range = float(
        xy[:, 1].max() - xy[:, 1].min()
    )

    displacement = float(
        np.linalg.norm(
            xy[-1] - xy[0]
        )
    )

    return {
        "frames": len(index_tip),
        "path_length": total_path,
        "horizontal_range": horizontal_range,
        "vertical_range": vertical_range,
        "displacement": displacement,
    }


def plot_sample(file_path, title_prefix):
    _, index_tip = load_index_path(file_path)

    xy = index_tip[:, :2]

    stats = path_statistics(index_tip)

    plt.figure(figsize=(7, 6))

    plt.plot(
        xy[:, 0],
        xy[:, 1],
        marker="o",
        markersize=3,
    )

    plt.scatter(
        xy[0, 0],
        xy[0, 1],
        s=120,
        label="Start",
    )

    plt.scatter(
        xy[-1, 0],
        xy[-1, 1],
        s=120,
        label="End",
    )

    for frame_index in range(
        0,
        len(xy),
        max(1, len(xy) // 8),
    ):
        plt.annotate(
            str(frame_index),
            (
                xy[frame_index, 0],
                xy[frame_index, 1],
            ),
        )

    plt.gca().invert_yaxis()

    plt.xlabel("Index tip X relative to wrist")
    plt.ylabel("Index tip Y relative to wrist")

    plt.title(
        f"{title_prefix}\n"
        f"{file_path.name}\n"
        f"frames={stats['frames']} | "
        f"path={stats['path_length']:.4f} | "
        f"x-range={stats['horizontal_range']:.4f} | "
        f"y-range={stats['vertical_range']:.4f}"
    )

    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    print()
    print("=" * 80)
    print(file_path.name)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    plt.show()


def main():
    none_dir = TEST_DIR / "NONE"
    z_dir = TEST_DIR / "Z"

    print()
    print("=" * 80)
    print("DYNAMIC SAMPLE INSPECTOR")
    print("=" * 80)
    print()
    print("First, the eight NONE samples incorrectly predicted as Z.")

    for filename in MISCLASSIFIED_NONE_FILES:
        file_path = none_dir / filename

        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue

        plot_sample(
            file_path,
            "Actual NONE — predicted as Z",
        )

    z_files = sorted(
        z_dir.glob("*.npz")
    )[:8]

    print()
    print("Now showing eight real Z samples for comparison.")

    for file_path in z_files:
        plot_sample(
            file_path,
            "Actual Z",
        )


if __name__ == "__main__":
    main()
