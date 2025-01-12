from collections import defaultdict
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_cluster_centers(clustering_results):
    clusters = defaultdict(list)
    for key, value in clustering_results.items():
        clusters[value["cluster"]].append(key)

    cluster_id_list = sorted(clusters.keys())

    cluster_centers = {}
    for cluster_id in cluster_id_list:
        mean_x = np.mean([clustering_results[key]["x"]
                         for key in clusters[cluster_id]])
        mean_y = np.mean([clustering_results[key]["y"]
                         for key in clusters[cluster_id]])
        cluster_centers[cluster_id] = (float(mean_x), float(mean_y))
    return cluster_centers


def draw_points(
    output_path: Path,
    xs, ys, colors: list[tuple[float, float, float]],
    cluster_centers: dict[int, tuple[float, float]],
    show_as_x: list | None = None,
    ):
    if show_as_x is None:
        show_as_x = [False] * len(xs)

    plt.figure(figsize=(12, 12))

    for x, y, color, show_as_x in zip(xs, ys, colors, show_as_x):
        plt.scatter(x, y, color=color, marker="x" if show_as_x else "o")

    for cluster_id, (x, y) in cluster_centers.items():
        plt.text(x, y, f"{cluster_id}", fontsize=12, color="black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

