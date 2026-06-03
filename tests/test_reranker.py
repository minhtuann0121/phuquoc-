import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reranker import rerank_places, extract_features


class RerankerTests(unittest.TestCase):
    def test_rerank_places_returns_req_breakdown_and_rank(self):
        places = [
            {
                "ten": "Quan dia phuong",
                "ho_kinh_doanh_dia_phuong": True,
                "chu_la_ngu_dan": True,
                "nghe_truyen_thong": True,
                "dung_lao_dong_yeu_the": False,
                "is_open_now": 1,
                "distance_meters": 250,
                "rating": 4.2,
                "price_level": 1,
                "wheelchair_accessible": False,
                "category_match": 0.8,
            },
            {
                "ten": "Chuoi lon",
                "ho_kinh_doanh_dia_phuong": False,
                "chu_la_ngu_dan": False,
                "nghe_truyen_thong": False,
                "dung_lao_dong_yeu_the": False,
                "is_open_now": 1,
                "distance_meters": 350,
                "rating": 4.8,
                "price_level": 3,
                "wheelchair_accessible": True,
                "category_match": 0.9,
            },
        ]

        ranked = rerank_places(copy.deepcopy(places))

        self.assertEqual(ranked[0]["ten"], "Quan dia phuong")
        self.assertEqual([p["rank"] for p in ranked], [1, 2])

        breakdown = ranked[0]["score_breakdown"]
        expected_keys = {
            "tree1_locality",
            "tree2_proximity",
            "tree3_quality",
            "s_bag",
            "delta1_fairness",
            "delta2_access",
            "final_score",
            "rank",
            "rf_score",
            "gbm_score",
            "local_factor",
            "feature_importances",
        }
        self.assertTrue(expected_keys.issubset(breakdown))
        self.assertEqual(ranked[0]["final_score"], breakdown["final_score"])
        self.assertEqual(breakdown["rank"], 1)

    def test_accessibility_bonus_is_recorded_as_weighted_delta(self):
        place = {
            "ten": "Accessible local place",
            "local_factor": 0.5,
            "is_open_now": 1,
            "distance_meters": 1000,
            "rating": 4.0,
            "price_level": 2,
            "wheelchair_accessible": True,
        }

        ranked = rerank_places([place])

        self.assertEqual(ranked[0]["score_breakdown"]["delta2_access"], 0.03)

    def test_extract_features_normalizes_missing_and_alias_fields(self):
        place = {
            "rating": None,
            "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
            "currentOpeningHours": {"openNow": True},
            "accessibilityOptions": {"wheelchairAccessibleEntrance": True},
            "ho_kinh_doanh_dia_phuong": True,
        }

        features = extract_features(place)

        self.assertEqual(features["rating"], 3.0)
        self.assertEqual(features["price_level"], 1)
        self.assertEqual(features["is_open_now"], 1)
        self.assertIs(features["wheelchair_accessible"], True)
        self.assertEqual(features["local_factor"], 0.4)

    def test_cheap_preference_can_outrank_stronger_local_factor(self):
        places = [
            {
                "ten": "Quan Mam Ba Nam",
                "local_factor": 1.0,
                "is_open_now": 1,
                "distance_meters": 150,
                "rating": 4.2,
                "price_level": 2,
                "wheelchair_accessible": True,
            },
            {
                "ten": "Cafe Bien Ham Ninh",
                "local_factor": 0.55,
                "is_open_now": 1,
                "distance_meters": 500,
                "rating": 4.0,
                "price_level": 1,
                "wheelchair_accessible": False,
            },
        ]

        ranked = rerank_places(copy.deepcopy(places), preference="cheap")

        self.assertEqual(ranked[0]["ten"], "Cafe Bien Ham Ninh")
        self.assertGreater(ranked[0]["score_breakdown"]["preference_bonus"], 0)
        self.assertEqual(ranked[0]["score_breakdown"]["preference"], "cheap")

    def test_local_preference_keeps_stronger_local_factor_first(self):
        places = [
            {
                "ten": "Quan Mam Ba Nam",
                "local_factor": 1.0,
                "is_open_now": 1,
                "distance_meters": 150,
                "rating": 4.2,
                "price_level": 2,
                "wheelchair_accessible": True,
            },
            {
                "ten": "Cafe Bien Ham Ninh",
                "local_factor": 0.55,
                "is_open_now": 1,
                "distance_meters": 500,
                "rating": 4.0,
                "price_level": 1,
                "wheelchair_accessible": False,
            },
        ]

        ranked = rerank_places(copy.deepcopy(places), preference="local")

        self.assertEqual(ranked[0]["ten"], "Quan Mam Ba Nam")
        self.assertGreater(ranked[0]["score_breakdown"]["preference_bonus"], ranked[1]["score_breakdown"]["preference_bonus"])
        self.assertEqual(ranked[0]["score_breakdown"]["preference"], "local")


if __name__ == "__main__":
    unittest.main()
