"""Session-level prediction monitoring summaries."""

from collections import Counter


def session_metrics(history):
    probabilities = [float(item.get("probability", 0.0)) for item in history]
    confidences = [float(item.get("confidence", max(p, 1 - p))) for item, p in zip(history, probabilities)]
    distribution = Counter(item.get("Risk", "Unknown") for item in history)
    return {
        "count": len(history),
        "average_risk": sum(probabilities) / len(probabilities) if probabilities else 0.0,
        "distribution": {name: distribution.get(name, 0) for name in ("Low", "Moderate", "High")},
        "trend": probabilities[-10:],
        "confidences": confidences,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }

