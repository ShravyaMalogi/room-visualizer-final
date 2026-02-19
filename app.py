from flask import Flask, render_template, request, redirect, jsonify, send_from_directory
from PIL import Image
import os
import numpy as np
import uuid

from texture_mapping import image_resize
from wall_segmentation.segmentation import build_model

from config import *
from services.texture_service import (
    apply_wall_texture,
    apply_wall_single_texture,
    apply_floor_texture
)

app = Flask(__name__)

os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

print("Loading wall model...")
wall_model = build_model()


@app.route("/")
def main():
    return redirect("/room")


@app.route("/prediction", methods=["POST"])
def predict_image_room():

    file = request.files.get("image")
    if not file:
        return jsonify(success=False, error="No image"), 400

    uid = uuid.uuid4().hex
    room_path = os.path.join(IMG_FOLDER, f"{uid}_room.jpg")

    img = Image.open(file.stream).convert("RGB")
    img_np = np.asarray(img)

    if img_np.shape[0] > 600:
        img_np = image_resize(img_np, height=600)

    Image.fromarray(img_np).save(room_path)

    return jsonify(success=True, redirect_url=f"/room/{uid}")


@app.route("/room")
@app.route("/room/<uid>")
def room(uid=None):

    textures = {
        "wall": sorted(os.listdir(WALL_TEXTURES)),
        "floor": sorted(os.listdir(FLOOR_TEXTURES)),
        "single": sorted(os.listdir(SINGLE_PIECE_TEXTURES))
    }

    room_image = ""
    if uid:
        textured = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")
        if os.path.isfile(textured):
            room_image = f"/static/IMG/{uid}_textured.jpg"
        else:
            room_image = f"/static/IMG/{uid}_room.jpg"

    return render_template(
        "index.html",
        room=room_image,
        textures=textures,
        uid=uid,
        enable_textures=bool(uid)
    )


@app.route("/result_textured", methods=["POST"])
def result_textured():

    data = request.get_json()
    uid = data["uid"]
    tex_type = data["type"]
    tex_name = data["texture"]

    try:
        if tex_type == "wall":
            apply_wall_texture(uid, tex_name, wall_model)

        elif tex_type == "single":
            apply_wall_single_texture(uid, tex_name, wall_model)

        elif tex_type == "floor":
            apply_floor_texture(uid, tex_name)

        else:
            return jsonify(state="error", msg="Invalid type")

    except Exception as e:
        return jsonify(state="error", msg=str(e))

    return jsonify(
        state="success",
        room_path=f"/static/IMG/{uid}_textured.jpg"
    )


@app.route("/reset_texture", methods=["POST"])
def reset_texture():
    data = request.get_json()
    uid = data.get("uid")

    if not uid:
        return jsonify(success=False, error="No UID provided"), 400

    textured_path = os.path.join(IMG_FOLDER, f"{uid}_textured.jpg")

    # Delete the textured version if it exists
    if os.path.isfile(textured_path):
        try:
            os.remove(textured_path)
        except Exception as e:
            return jsonify(success=False, error=str(e)), 500

    # Return the original room image
    return jsonify(
        success=True,
        room_path=f"/static/IMG/{uid}_room.jpg"
    )


@app.route("/textures/<path:filename>")
def serve_texture(filename):
    return send_from_directory(TEXTURE_LIBRARY, filename)


if __name__ == "__main__":
    app.run(port=9000, debug=True)
