import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import SpectralClustering
from janome.tokenizer import Tokenizer

STOP_WORDS = [
    "の",
    "に",
    "は",
    "を",
    "た",
    "が",
    "で",
    "て",
    "と",
    "し",
    "れ",
    "さ",
    "ある",
    "いる",
    "も",
    "する",
    "から",
    "な",
    "こと",
    "として",
    "いく",
    "ない",
]
TOKENIZER = Tokenizer()


def tokenize_japanese(text):
    return [
        token.surface
        for token in TOKENIZER.tokenize(text)
        if token.surface not in STOP_WORDS
    ]


def clustering(to_summary: dict[str, str], to_embed: dict[str, np.ndarray]):
    import bertopic
    import hdbscan
    import umap

    keys = sorted(to_embed.keys())
    n_topics = int(len(keys) ** 0.5)
    embeddings = np.array([to_embed[key] for key in keys])
    docs = [to_summary[key] for key in keys]
    umap_model = umap.UMAP(random_state=42, n_components=2)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2)
    topic_model = bertopic.BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(tokenizer=tokenize_japanese),
        verbose=True,
    )
    topic_model.fit_transform(docs, embeddings=embeddings)
    spectral_model = SpectralClustering(
        n_clusters=n_topics,
        affinity="nearest_neighbors",
        n_neighbors=min(len(docs) - 1, 10),
        random_state=42,
    )
    umap_embeddings = umap_model.fit_transform(embeddings)
    cluster_labels = spectral_model.fit_predict(umap_embeddings)
    result = topic_model.get_document_info(
        docs=docs,
        metadata={
            "id": keys,
            "x": umap_embeddings[:, 0],
            "y": umap_embeddings[:, 1],
        },
    )
    result.columns = [c.lower() for c in result.columns]
    result["cluster-id"] = cluster_labels
    result = result[["id", "x", "y", "probability", "cluster-id"]]
    probablities = result["probability"].to_numpy().tolist()
    return {
        key: {
            "cluster": int(cluster),
            "x": float(emb[0]),
            "y": float(emb[1]),
            "p": float(p),
        }
        for key, cluster, emb, p in zip(
            keys, cluster_labels, umap_embeddings.tolist(), probablities
        )
    }
