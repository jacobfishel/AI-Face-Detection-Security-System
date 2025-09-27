"""
SQLite-backed face embedding utilities.

Provides:
- get_embedding(image)
- store_embedding(name, embedding)
- find_best_match(embedding)
- prompt_label_if_unknown(cropped_face_image)

Notes:
- Requires the `face_recognition` package (dlib-based). Install: pip install face_recognition
- Expects a cropped face image (RGB or BGR numpy array). If BGR (OpenCV), it will be converted.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Tuple, List

import numpy as np


DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_embeddings.sqlite")


def _ensure_db(db_path: str = DEFAULT_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            """
        )
        conn.commit()


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:  # grayscale → stack
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 3:
        # Heuristic: if mean of first channel is much higher than third, assume BGR and convert
        # Safer: allow caller to pass either format. We'll try BGR→RGB conversion by swapping channels.
        # Users commonly pass OpenCV BGR.
        return image[:, :, ::-1]
    return image


def get_embedding(image: np.ndarray) -> np.ndarray:
    """
    Convert a cropped face image into an embedding vector using face_recognition.

    Args:
        image: Cropped face image as numpy array (RGB or BGR), dtype uint8 preferred.

    Returns:
        Numpy array embedding (float32), length typically 128.
    """
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "face_recognition is required. Install with: pip install face_recognition"
        ) from exc

    if image.dtype != np.uint8:
        # Normalize to 0-255 uint8 if needed
        img = image
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        img = image

    # face_recognition expects RGB
    # If the caller already passed RGB, swapping will convert to BGR incorrectly; however
    # most OpenCV sources are BGR. To be explicit, if you know it's RGB pass image[:, :, ::-1] here.
    rgb = _to_rgb(img)

    encodings: List[np.ndarray] = face_recognition.face_encodings(rgb)
    if not encodings:
        raise ValueError("No face encoding could be computed for the provided cropped face image.")

    embedding = np.asarray(encodings[0], dtype=np.float32)
    return embedding


def store_embedding(name: str, embedding: np.ndarray, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Insert embedding + label into SQLite.

    Schema: embeddings(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, embedding BLOB)
    """
    _ensure_db(db_path)
    if not isinstance(embedding, np.ndarray):
        raise TypeError("embedding must be a numpy array")
    emb = embedding.astype(np.float32).tobytes()
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO embeddings (name, embedding) VALUES (?, ?)",
            (name, sqlite3.Binary(emb)),
        )
        conn.commit()
        return int(cur.lastrowid)


def _load_all_embeddings(db_path: str = DEFAULT_DB_PATH) -> List[Tuple[int, str, np.ndarray]]:
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, embedding FROM embeddings")
        rows = cur.fetchall()
    results: List[Tuple[int, str, np.ndarray]] = []
    for rid, name, blob in rows:
        arr = np.frombuffer(blob, dtype=np.float32)
        results.append((int(rid), name, arr))
    return results


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def find_best_match(
    embedding: np.ndarray,
    db_path: str = DEFAULT_DB_PATH,
    metric: str = "euclidean",
    threshold: Optional[float] = None,
) -> Optional[Tuple[str, float]]:
    """
    Compare a new embedding to stored ones and return (best_name, score) if under/over threshold.

    metric:
        - "euclidean" (default): lower is better; typical threshold ~0.6 for face_recognition
        - "cosine": higher is better; reasonable threshold ~0.5-0.6 after normalization
    """
    stored = _load_all_embeddings(db_path)
    if not stored:
        return None

    if metric not in ("euclidean", "cosine"):
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    best_name: Optional[str] = None
    best_score: Optional[float] = None

    for _rid, name, emb in stored:
        if metric == "euclidean":
            score = _euclidean_distance(embedding.astype(np.float32), emb.astype(np.float32))
            if best_score is None or score < best_score:
                best_score = score
                best_name = name
        else:
            score = _cosine_similarity(embedding.astype(np.float32), emb.astype(np.float32))
            if best_score is None or score > best_score:
                best_score = score
                best_name = name

    if best_name is None or best_score is None:
        return None

    # Apply thresholding if provided; else use sensible defaults.
    if threshold is None:
        threshold = 0.6 if metric == "euclidean" else 0.5

    if metric == "euclidean":
        return (best_name, best_score) if best_score <= threshold else None
    else:
        return (best_name, best_score) if best_score >= threshold else None


def prompt_label_if_unknown(
    cropped_face_image: np.ndarray,
    db_path: str = DEFAULT_DB_PATH,
    metric: str = "euclidean",
    threshold: Optional[float] = None,
) -> Tuple[str, np.ndarray, Optional[Tuple[str, float]]]:
    """
    Compute embedding, try to match, and prompt the user to label if unknown.

    Returns: (final_name, embedding, match_info)
    """
    emb = get_embedding(cropped_face_image)
    match = find_best_match(emb, db_path=db_path, metric=metric, threshold=threshold)
    if match is not None:
        return match[0], emb, match

    try:
        name = input("No match found. Enter name to add (or leave blank to skip): ").strip()
    except EOFError:
        name = ""

    if name:
        store_embedding(name, emb, db_path=db_path)
        return name, emb, None
    else:
        return "", emb, None


__all__ = [
    "get_embedding",
    "store_embedding",
    "find_best_match",
    "prompt_label_if_unknown",
    "DEFAULT_DB_PATH",
]


