"""Rule-based ensemble re-ranker for local tourism recommendations.

This module implements the requirement document's ML core without a trained
model: three expert-defined decision trees are averaged as a bagging step, then
two shallow correction stumps are applied as a boosting step.
"""

from __future__ import annotations

import math
from typing import Any


ETA = 0.3
FAIRNESS_CHAIN_THRESHOLD = 0.1
FAIRNESS_CHAIN_DELTA = -0.15
ACCESSIBILITY_DELTA = 0.10

PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    "FREE": 0,
    "INEXPENSIVE": 1,
    "MODERATE": 2,
    "EXPENSIVE": 3,
    "VERY_EXPENSIVE": 4,
}

FEATURE_IMPORTANCES = {
    "local_factor": 0.30,
    "distance_meters": 0.20,
    "rating": 0.18,
    "price_level": 0.12,
    "is_open_now": 0.10,
    "category_match": 0.10,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _as_int_flag(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "open", "opened"})
    return int(bool(value))


def _parse_price_level(value: Any, default: int = 2) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        value = PRICE_LEVEL_MAP.get(value.strip().upper(), value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def tinh_local_factor(quan: dict) -> float:
    """Compute the local fairness factor from internal metadata."""
    diem = 0.0

    if quan.get("ho_kinh_doanh_dia_phuong"):
        diem += 0.40
    if quan.get("chu_la_ngu_dan"):
        diem += 0.25
    if quan.get("nghe_truyen_thong"):
        diem += 0.20
    if quan.get("dung_lao_dong_yeu_the"):
        diem += 0.15

    return round(_clamp(diem), 3)


def extract_features(quan: dict) -> dict:
    """Normalize a raw place dict into the six-feature space from the req file."""
    opening_hours = quan.get("currentOpeningHours") or quan.get("regularOpeningHours") or {}
    accessibility = quan.get("accessibilityOptions") or {}

    price_level = quan.get("price_level", quan.get("priceLevel"))
    open_now = quan.get("is_open_now", opening_hours.get("openNow"))
    wheelchair = quan.get(
        "wheelchair_accessible",
        accessibility.get("wheelchairAccessibleEntrance", False),
    )

    if "local_factor" in quan and quan.get("local_factor") is not None:
        local_factor = _as_float(quan.get("local_factor"), 0.0)
    else:
        local_factor = tinh_local_factor(quan)

    return {
        "rating": _clamp(_as_float(quan.get("rating"), 3.0), 1.0, 5.0),
        "distance_meters": max(0.0, _as_float(quan.get("distance_meters"), 9999.0)),
        "price_level": max(0, min(_parse_price_level(price_level), 4)),
        "is_open_now": _as_int_flag(open_now),
        "local_factor": _clamp(local_factor),
        "category_match": _clamp(_as_float(quan.get("category_match"), 0.5)),
        "wheelchair_accessible": bool(wheelchair),
    }


def tree1_locality(quan: dict) -> float:
    """Tree 1: locality-first scoring."""
    features = extract_features(quan)
    local = features["local_factor"]
    mo_cua = features["is_open_now"]

    if local > 0.6 and mo_cua == 1:
        return 0.9
    if local > 0.6:
        return 0.7
    if local > 0.3:
        return 0.5
    return 0.2


def tree2_proximity(quan: dict) -> float:
    """Tree 2: proximity-first scoring."""
    features = extract_features(quan)
    distance = features["distance_meters"]
    rating = features["rating"]
    local = features["local_factor"]

    if distance < 300:
        return 0.9
    if distance < 800:
        return _clamp(0.65 + (rating - 3.0) * 0.1)
    if distance < 2000:
        return _clamp(0.4 + local * 0.2)
    return 0.15


def tree3_quality(quan: dict) -> float:
    """Tree 3: quality and budget scoring."""
    features = extract_features(quan)
    rating = features["rating"]
    price = features["price_level"]
    local = features["local_factor"]

    if rating >= 4.5 and price <= 2:
        return _clamp(0.85 + local * 0.15)
    if rating >= 4.0 and price <= 1:
        return 0.75
    if rating >= 3.5:
        return _clamp(0.5 + (2 - price) * 0.05)
    return 0.2


def bagging(quan: dict, weights: dict | None = None) -> float:
    """Average the three expert trees, with optional UI preference weights."""
    if weights is None:
        weights = {"locality": 1.0, "proximity": 1.0, "quality": 1.0}

    tree_scores = {
        "locality": tree1_locality(quan),
        "proximity": tree2_proximity(quan),
        "quality": tree3_quality(quan),
    }
    total_weight = sum(_as_float(weights.get(name), 1.0) for name in tree_scores)
    if total_weight <= 0:
        total_weight = 3.0
        weights = {"locality": 1.0, "proximity": 1.0, "quality": 1.0}

    score = sum(
        tree_scores[name] * _as_float(weights.get(name), 1.0)
        for name in tree_scores
    ) / total_weight
    return _clamp(score)


def boosting_details(quan: dict, s_bag: float) -> dict:
    """Apply two sequential boosting correction stumps and return trace data."""
    features = extract_features(quan)

    raw_delta1 = FAIRNESS_CHAIN_DELTA if features["local_factor"] < FAIRNESS_CHAIN_THRESHOLD else 0.0
    weighted_delta1 = ETA * raw_delta1
    f1 = s_bag + weighted_delta1

    raw_delta2 = ACCESSIBILITY_DELTA if features["wheelchair_accessible"] else 0.0
    weighted_delta2 = ETA * raw_delta2
    final_score = _clamp(f1 + weighted_delta2)

    return {
        "delta1_fairness": round(weighted_delta1, 3),
        "delta2_access": round(weighted_delta2, 3),
        "final_score": round(final_score, 3),
    }


def boosting(quan: dict, s_bag: float) -> float:
    """Backward-compatible wrapper returning only the final boosted score."""
    return boosting_details(quan, s_bag)["final_score"]


def build_score_breakdown(
    quan: dict,
    rank: int | None = None,
    weights: dict | None = None,
    preference: str | None = None,
) -> dict:
    """Create the explainability payload consumed by API/UI responses."""
    features = extract_features(quan)
    t1 = round(tree1_locality(quan), 3)
    t2 = round(tree2_proximity(quan), 3)
    t3 = round(tree3_quality(quan), 3)
    s_bag = round(bagging(quan, weights=weights), 3)
    boosted = boosting_details(quan, s_bag)
    preference_key = normalize_preference(preference)
    preference_bonus = compute_preference_bonus(features, preference_key)
    final_score = round(_clamp(boosted["final_score"] + preference_bonus), 3)

    breakdown = {
        "tree1_locality": t1,
        "tree2_proximity": t2,
        "tree3_quality": t3,
        "s_bag": s_bag,
        "delta1_fairness": boosted["delta1_fairness"],
        "delta2_access": boosted["delta2_access"],
        "final_score": final_score,
        "rank": rank,
        "rf_score": s_bag,
        "gbm_score": boosted["final_score"],
        "local_factor": round(features["local_factor"], 3),
        "category_match": round(features["category_match"], 3),
        "preference": preference_key,
        "preference_bonus": round(preference_bonus, 3),
        "feature_importances": FEATURE_IMPORTANCES.copy(),
    }
    return breakdown


def rerank_places(
    danh_sach_quan: list,
    top_k: int | None = None,
    weights: dict | None = None,
    budget_filter: str = "any",
    accessibility_required: bool = False,
    preference: str | None = None,
) -> list:
    """Score and sort places using bagging plus boosting.

    Existing callers can keep using rerank_places(list). Optional filters support
    the req-level budget/accessibility controls when the API layer is ready.
    """
    filtered_places = []
    for quan in danh_sach_quan:
        features = extract_features(quan)
        quan["local_factor"] = round(features["local_factor"], 3)
        quan["accessibility_score"] = 1.0 if features["wheelchair_accessible"] else 0.0

        if accessibility_required and not features["wheelchair_accessible"]:
            continue
        if not _matches_budget(features["price_level"], budget_filter):
            continue

        breakdown = build_score_breakdown(quan, weights=weights, preference=preference)
        quan["score_breakdown"] = breakdown
        quan["final_score"] = breakdown["final_score"]
        quan["rf_score"] = breakdown["rf_score"]
        quan["gbm_score"] = breakdown["gbm_score"]
        filtered_places.append(quan)

    ranked = sorted(
        filtered_places,
        key=lambda q: (
            q["final_score"],
            q.get("local_factor", 0.0),
            -extract_features(q)["distance_meters"],
        ),
        reverse=True,
    )

    if top_k is not None:
        ranked = ranked[:top_k]

    for index, quan in enumerate(ranked, start=1):
        quan["rank"] = index
        quan["score_breakdown"]["rank"] = index

    return ranked



def normalize_preference(preference: str | None) -> str:
    """Map UI/user wording into a stable reranking preference key."""
    if not preference:
        return "balanced"

    text = preference.strip().lower()
    cheap_keywords = {"cheap", "budget", "low", "inexpensive", "gia re", "giá rẻ", "tiết kiệm", "tiet kiem", "rẻ"}
    local_keywords = {"local", "locality", "traditional", "dia phuong", "địa phương", "bản địa", "ban dia", "truyền thống", "truyen thong"}
    near_keywords = {"near", "nearest", "distance", "gan", "gần", "khoảng cách", "khoang cach"}

    if any(keyword in text for keyword in cheap_keywords):
        return "cheap"
    if any(keyword in text for keyword in local_keywords):
        return "local"
    if any(keyword in text for keyword in near_keywords):
        return "near"
    return "balanced"


def compute_preference_bonus(features: dict, preference: str) -> float:
    """Return a transparent bonus for the explicit user preference."""
    if preference == "cheap":
        price = features["price_level"]
        if price <= 0:
            return 0.22
        if price == 1:
            return 0.18
        if price == 2:
            return 0.0
        return -0.10

    if preference == "local":
        return features["local_factor"] * 0.25

    if preference == "near":
        distance = features["distance_meters"]
        if distance < 300:
            return 0.14
        if distance < 800:
            return 0.08
        if distance < 2000:
            return 0.03
        return 0.0

    return 0.0
def _matches_budget(price_level: int, budget_filter: str) -> bool:
    filter_value = (budget_filter or "any").lower()
    if filter_value in {"any", "all"}:
        return True
    if filter_value in {"free"}:
        return price_level == 0
    if filter_value in {"low", "cheap", "budget", "inexpensive"}:
        return price_level <= 1
    if filter_value in {"medium", "moderate"}:
        return price_level <= 2
    if filter_value in {"high", "expensive"}:
        return price_level >= 3
    return True


if __name__ == "__main__":
    danh_sach = [
        {
            "ten": "Quan Ba Nam",
            "ho_kinh_doanh_dia_phuong": True,
            "chu_la_ngu_dan": True,
            "nghe_truyen_thong": False,
            "dung_lao_dong_yeu_the": False,
            "is_open_now": 1,
            "distance_meters": 500,
            "rating": 4.2,
            "price_level": 1,
            "wheelchair_accessible": True,
        },
        {
            "ten": "Quan Hai San ABC",
            "ho_kinh_doanh_dia_phuong": False,
            "chu_la_ngu_dan": False,
            "nghe_truyen_thong": False,
            "dung_lao_dong_yeu_the": False,
            "is_open_now": 1,
            "distance_meters": 400,
            "rating": 4.5,
            "price_level": 2,
            "wheelchair_accessible": False,
        },
    ]

    for quan in rerank_places(danh_sach):
        print(f"\n--- Hang {quan['rank']}: {quan['ten']} ---")
        print(f"  local_factor : {quan['local_factor']}")
        print(f"  final_score  : {quan['final_score']}")
        print(f"  breakdown    : {quan['score_breakdown']}")
