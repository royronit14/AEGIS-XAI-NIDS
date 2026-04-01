def summarize_drift(drift_df):
    total_features = len(drift_df)

    severe = drift_df[drift_df["status"] == "severe_drift"]
    moderate = drift_df[drift_df["status"] == "moderate_drift"]

    severe_ratio = len(severe) / total_features
    moderate_ratio = len(moderate) / total_features

    if severe_ratio > 0.3:
        drift_level = "high"
    elif severe_ratio > 0.1 or moderate_ratio > 0.3:
        drift_level = "medium"
    else:
        drift_level = "low"

    return {
        "drift_level": drift_level,
        "severe_count": len(severe),
        "moderate_count": len(moderate),
        "total_features": total_features,
        "severe_ratio": round(severe_ratio, 3),
        "moderate_ratio": round(moderate_ratio, 3),
    }
