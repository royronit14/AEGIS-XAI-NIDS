class DriftConfig:
    PSI_LOW = 0.1
    PSI_HIGH = 0.25

class AlertConfig:
    DRIFT_WEIGHT = 0.65
    CONF_WEIGHT = 0.35

class RateLimitConfig:
    REQUESTS = 20
    WINDOW = 60  # seconds
