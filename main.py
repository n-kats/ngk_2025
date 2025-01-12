from report_utils import get_cluster_centers, draw_points
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from collections import defaultdict
import numpy as np
import re
from pathlib import Path

from clustering_utils import clustering
from utils import get_day
from framework import load, save, source, indexing, model


@source
def row_markdown(path: Path):
    result = {}
    for md_path in path.glob("*_reports/**/*.md"):
        if md_path.name == "README.md":
            continue
        relative_path = md_path.relative_to(path)
        result[str(relative_path)] = md_path.read_text()

    return result


@indexing(inputs=["row_markdown"])
def title(row_markdown: str):
    lines = row_markdown.split("\n")

    def is_invalid(line):
        if line in [
            "",
            "\ufeff",
            "# タイトル",
            "論文のタイトルを書く",
            "論文のタイトル",
            "#",
            "===",
        ]:
            return True

        if re.match(r"^[\d/]+$", line):
            return True
        return False

    while len(lines) > 0 and is_invalid(lines[0]):
        lines = lines[1:]
    title = lines[0]

    title = re.sub(r"#+", "", title)
    title = re.sub(r"\(http[^\)]+\)", "", title)
    title = title.replace("[arxiv]", "").replace("[\\[arxiv\\]]", "")
    title = title.replace("[", "").replace("]", "").replace("**", "")
    title = re.sub(r"http.+$", "", title)
    title = title.replace("\ufeff", "").strip()

    return title


def embedding_openai_3_large(text: str):
    """c.f. https://openai.com/index/new-embedding-models-and-api-updates/"""
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large",
    )
    return response.data[0].embedding


@indexing(inputs=["title"], by_npz=True)
def title_embedding_openai_3_large(title: str):
    return embedding_openai_3_large(title)


@indexing(inputs=["row_markdown"])
def row_embedding_openai_3_large(row_markdown: str):
    return embedding_openai_3_large(
        row_markdown[:6000]
    )  # 8192トークンが限度のため適当に切る


def prompt_openai_4o(prompt: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def summary_by_openai_4o(text: str):
    prompt = f"""以下の文章を要約してください。返答は要約文のみでお願いします。

{text}
"""
    return prompt_openai_4o(prompt)


@indexing(inputs=["row_markdown"])
def summary_openai_4o(row_markdown: str):
    return summary_by_openai_4o(row_markdown)


@indexing(inputs=["summary_openai_4o"], by_npz=True)
def summary_openai_4o_embedding_openai_3_large(summary_openai_4o: str):
    return embedding_openai_3_large(summary_openai_4o)


@model(inputs=["summary_openai_4o", "summary_openai_4o_embedding_openai_3_large"])
def clustering_summary_openai_4o_embedding_openai_3_large(
    to_summary: dict[str, str], to_embed: dict[str, np.ndarray]
):
    return clustering(to_summary, to_embed)


@model(
    inputs=[
        "summary_openai_4o",
        "clustering_summary_openai_4o_embedding_openai_3_large",
    ]
)
def summary_clustering_summary_openai_4o_embedding_openai_3_large(
    to_summary: dict[str, dict],
    clustering_results: dict[str, dict],
):
    clusters = defaultdict(list)
    for key, value in clustering_results.items():
        clusters[value["cluster"]].append(key)

    cluster_summaies = {}
    for cluster_id, keys in clusters.items():
        prompts = [
            "以下は同じグループと判定したまとめ文章の集まりです。どのようなグループなのかをコンパクトにまとめてください。返答は、～を扱ったグループという形で要約文のみでお願いします。"
        ]
        for i, key in enumerate(keys):
            prompts.append(f"# {i+1}/{len(keys)}")
            prompts.append(f"{to_summary[key]}")
            prompts.append("")

        cluster_summaies[cluster_id] = prompt_openai_4o("\n".join(prompts))
        print(cluster_summaies[cluster_id])

    return cluster_summaies


@model(
    inputs=[
        "summary_openai_4o",
        "clustering_summary_openai_4o_embedding_openai_3_large",
    ]
)
def long_summary_clustering_summary_openai_4o_embedding_openai_3_large(
    to_summary: dict[str, dict],
    clustering_results: dict[str, dict],
):
    clusters = defaultdict(list)
    for key, value in clustering_results.items():
        clusters[value["cluster"]].append(key)

    cluster_summaies = {}
    for cluster_id, keys in clusters.items():
        prompts = [
            "以下は同じグループと判定したまとめ文章の集まりです。どのようなグループなのかが分かるような形で要約してください。出力は、要約文のみにしてください（まとめの部分の文章とするため）。また、このグループは〜という書き出しにしてください。"
        ]
        for i, key in enumerate(keys):
            prompts.append(f"# {i+1}/{len(keys)}")
            prompts.append(f"{to_summary[key]}")
            prompts.append("")

        cluster_summaies[cluster_id] = prompt_openai_4o("\n".join(prompts))
        print(cluster_summaies[cluster_id])

    return cluster_summaies


def report_clustering_result(
    summary_name: str,
    clustering_result_name: str,
    clustering_summary_name: str,
    clustering_long_summary_name: str,
):
    summaries = load(summary_name)
    clustering_results = load(clustering_result_name)
    clustering_summaries = load(clustering_summary_name)
    clustering_long_summaries = load(clustering_long_summary_name)

    clusters = defaultdict(list)
    for key, value in clustering_results.items():
        clusters[value["cluster"]].append(key)

    cluster_id_list = sorted(clusters.keys())

    palette = sns.color_palette("bright", len(cluster_id_list))
    cluster_to_color = dict(zip(cluster_id_list, palette))

    keys = sorted(summaries.keys())
    draw_points(
        Path(f"./_report/{clustering_result_name}.png"),
        xs=[clustering_results[key]["x"] for key in keys],
        ys=[clustering_results[key]["y"] for key in keys],
        colors=[cluster_to_color[clustering_results[key]["cluster"]]
                for key in keys],
        cluster_centers=get_cluster_centers(clustering_results),
    )

    with open(f"./_report/{clustering_result_name}.md", "w") as f:
        print(f"# {clustering_result_name}", file=f)
        print(f"![{clustering_result_name}](./{clustering_result_name}.png)", file=f)
        print("", file=f)

        for cluster_id in cluster_id_list:
            print(
                f"- {cluster_id}: {clustering_summaries[cluster_id]}", file=f)

        for cluster_id in cluster_id_list:
            short = clustering_summaries[cluster_id]
            print(f"<h2>Cluster {cluster_id}({short})</h2>", file=f)
            print(clustering_long_summaries[cluster_id], file=f)

            print("<details>", file=f)
            print("<summary>詳細</summary>", file=f)

            for key in clusters[cluster_id]:
                print(f"<h3>{key}</h3>", file=f)
                print(f"{summaries[key]}", file=f)

            print("</details>", file=f)

    print(f"Reported {clustering_result_name}")


@indexing(inputs=["row_markdown"])
def is_by_n_kats(row_markdown: str):
    return "まとめ @n-kats" in row_markdown


def report_clustering_result_for_n_kats(
    summary_name: str,
    clustering_result_name: str,
    clustering_summary_name: str,
    clustering_long_summary_name: str,
    is_by_n_kats_name: str,
):
    summaries = load(summary_name)
    clustering_results = load(clustering_result_name)
    clustering_summaries = load(clustering_summary_name)
    clustering_long_summaries = load(clustering_long_summary_name)
    is_by_n_kats = load(is_by_n_kats_name)

    clusters = defaultdict(list)
    for key, value in clustering_results.items():
        clusters[value["cluster"]].append(key)

    cluster_id_list = sorted(clusters.keys())

    keys = sorted(summaries.keys())
    xs = [clustering_results[key]["x"] for key in keys]
    ys = [clustering_results[key]["y"] for key in keys]

    cluster_counts = {cluster_id: len(elements)
                      for cluster_id, elements in clusters.items()}
    cluster_n_kats_counts = {cluster_id: sum(
        is_by_n_kats[key] for key in elements) for cluster_id, elements in clusters.items()}
    cluster_scores = {cluster_id: cluster_n_kats_counts[cluster_id] /
                      cluster_counts[cluster_id] for cluster_id in cluster_id_list}

    def get_colors_by_order(order):
        n = len(order)
        return {cluster_id: (0, i / n, 1 - i / n) for i, cluster_id in enumerate(order)}

    cluster_order = sorted(
        cluster_id_list, key=lambda cluster_id: cluster_scores[cluster_id])
    cluster_to_color = get_colors_by_order(cluster_order)
    colors = [cluster_to_color[clustering_results[key]["cluster"]]
              for key in keys]

    Path(f"./_report/n_kats").mkdir(exist_ok=True, parents=True)
    draw_points(
        Path(f"./_report/n_kats/{clustering_result_name}_n_kats.png"),
        xs, ys, colors=colors,
        cluster_centers=get_cluster_centers(clustering_results),
        show_as_x=[not is_by_n_kats[key] for key in keys],
    )

    with open(f"./_report/n_kats/{clustering_result_name}_n_kats.md", "w") as f:
        print(f"# {clustering_result_name}", file=f)
        print(
            f"![{clustering_result_name}](./{clustering_result_name}_n_kats.png)",
            file=f,
        )
        print("", file=f)

        for cluster_id in cluster_id_list:
            print(
                f"- {cluster_id}: {clustering_summaries[cluster_id]}", file=f)

        for cluster_id in cluster_id_list:
            short = clustering_summaries[cluster_id]
            score = cluster_scores[cluster_id]
            print(
                f"<h2>Cluster {cluster_id}(score={score:.2f}, {short})</h2> ", file=f)
            print(clustering_long_summaries[cluster_id], file=f)

            print("<details>", file=f)
            print("<summary>詳細</summary>", file=f)
            for key in clusters[cluster_id]:
                if is_by_n_kats[key]:
                    print(f"<h3>(読んだ){key}</h3>", file=f)
                else:
                    print(f"<h3>{key}</h3>", file=f)
                print(f"{summaries[key]}", file=f)
            print("</details>", file=f)
    print(f"Reported {clustering_result_name}")


def draw_timeseries_result(
    summary_name: str,
    clustering_result_name: str,
):
    summaries = load(summary_name)
    clustering_results = load(clustering_result_name)

    clusters = defaultdict(list)
    for key, value in clustering_results.items():
        clusters[value["cluster"]].append(key)

    cluster_id_list = sorted(clusters.keys())

    keys = sorted(summaries.keys())
    xs = [clustering_results[key]["x"] for key in keys]
    ys = [clustering_results[key]["y"] for key in keys]
    days = [get_day(key) for key in keys]
    days_sorted = sorted(set(days))
    n_days = len(days_sorted)
    day_to_color = {
        day: (0, i / n_days, 1 - i / n_days) for i, day in enumerate(days_sorted)
    }

    draw_points(
        Path(f"./_report/{clustering_result_name}_timeseries.png"),
        xs, ys, colors=[day_to_color[day] for day in days],
        cluster_centers=get_cluster_centers(clustering_results),
    )


def main():
    row_markdown(Path("vendor/surveys"))
    title()
    is_by_n_kats()
    title_embedding_openai_3_large(store_each=True)
    row_embedding_openai_3_large(store_each=True)
    summary_openai_4o(store_each=True)
    summary_openai_4o_embedding_openai_3_large(store_each=True)
    clustering_summary_openai_4o_embedding_openai_3_large()
    summary_clustering_summary_openai_4o_embedding_openai_3_large(
        skip_if_exist=True)
    long_summary_clustering_summary_openai_4o_embedding_openai_3_large(
        skip_if_exist=True)

    report_clustering_result(
        "summary_openai_4o",
        "clustering_summary_openai_4o_embedding_openai_3_large",
        "summary_clustering_summary_openai_4o_embedding_openai_3_large",
        "long_summary_clustering_summary_openai_4o_embedding_openai_3_large",
    )

    report_clustering_result_for_n_kats(
        "summary_openai_4o",
        "clustering_summary_openai_4o_embedding_openai_3_large",
        "summary_clustering_summary_openai_4o_embedding_openai_3_large",
        "long_summary_clustering_summary_openai_4o_embedding_openai_3_large",
        "is_by_n_kats",
    )

    draw_timeseries_result(
        "summary_openai_4o",
        "clustering_summary_openai_4o_embedding_openai_3_large",
    )


if __name__ == "__main__":
    main()
    pass
