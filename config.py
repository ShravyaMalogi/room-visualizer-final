import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_FOLDER = os.path.join(BASE_DIR, "static", "IMG")
DATA_FOLDER = os.path.join(BASE_DIR, "static", "data")

TEXTURE_LIBRARY = os.path.join(BASE_DIR, "test_images", "textures")
WALL_TEXTURES = os.path.join(TEXTURE_LIBRARY, "walls")
FLOOR_TEXTURES = os.path.join(TEXTURE_LIBRARY, "floors")
SINGLE_PIECE_TEXTURES = os.path.join(TEXTURE_LIBRARY, "single_piece")

# -------------------------------
# Roboflow config
# -------------------------------

# fallback values for testing
ROBOFLOW_API_KEY = os.getenv(
    "ROBOFLOW_API_KEY",
    "QS2vslJ7TfTmSTjxsUHi"
)

ROBOFLOW_MODEL_ID = os.getenv(
    "ROBOFLOW_MODEL_ID",
    "room-surface-segmentation/1"
)