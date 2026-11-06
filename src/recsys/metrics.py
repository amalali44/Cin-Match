"""Explicit ranking metric definitions independent of optional ML packages."""

import math


def ranking_metrics(recommendations: dict, relevant: dict, k: int) -> dict:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    precisions, recalls = [], []
    for user, positives in relevant.items():
        if not positives:
            continue
        ranked = recommendations.get(user, [])[:k]
        if len(ranked) != len(set(ranked)):
            raise ValueError("Recommendation lists must not contain duplicate movies")
        hits = len(set(ranked) & set(positives))
        precisions.append(hits / k)
        recalls.append(hits / len(positives))
    return {
        "precision_at_k": math.fsum(precisions) / len(precisions) if precisions else None,
        "recall_at_k": math.fsum(recalls) / len(recalls) if recalls else None,
        "evaluated_users": len(precisions),
    }
