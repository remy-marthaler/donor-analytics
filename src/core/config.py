
import os
from dotenv import load_dotenv

load_dotenv()

# Toggle between real API and mock API (default: mock for dev)
USE_MOCK_API = os.getenv("USE_MOCK_API", "true").lower() in {"1","true","yes","y"}

# Real API base URL and token (if you switch off mock)
FBOX_BASE = "https://api.fundraisingbox.com/v1"
