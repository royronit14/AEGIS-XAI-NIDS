def enrich_explanation_with_drift(
    explanation: dict,
    drift_df,
    top_k: int = 5
):
    if drift_df is None or drift_df.empty:
        explanation["drift_context"] = {
            "drifting_features": [],
            "note": "No drift context available"
        }
        return explanation

    drift_map = {
        row["feature"]: row["status"]
        for _, row in drift_df.iterrows()
    }

    drifting = []
    stable = []

    for item in explanation["top_contributors"][:top_k]:
        feature = item["feature"]
        status = drift_map.get(feature, "no_drift")

        if status != "no_drift":
            drifting.append({
                "feature": feature,
                "impact": item["impact"],
                "drift_status": status
            })
        else:
            stable.append(feature)

    explanation["drift_context"] = {
        "drifting_features": drifting,
        "stable_features": stable,
        "note": (
            "Some influential features are drifting"
            if drifting
            else "Top features are stable"
        )
    }

    return explanation
