"""Default constants for trading configuration."""

# Default model settings
DEFAULT_MODEL_PROVIDER = "openrouter"
DEFAULT_AGENT_MODEL = "deepseek/deepseek-chat"

# Default trading parameters
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_MAX_LEVERAGE = 3.0
DEFAULT_MAX_POSITIONS = 5
DEFAULT_CAP_FACTOR = 1.5
DEFAULT_DECIDE_INTERVAL = 60  # seconds
DEFAULT_FEE_BPS = 10.0  # 0.1%

# Feature grouping constants
FEATURE_GROUP_BY_KEY = "group_by_key"
FEATURE_GROUP_BY_INTERVAL_PREFIX = "interval_"
FEATURE_GROUP_BY_MARKET_SNAPSHOT = "market_snapshot"
