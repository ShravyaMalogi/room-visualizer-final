document.addEventListener("DOMContentLoaded", () => {

  console.log("texture_click.js loaded");

  const texturePanel = document.querySelector(".texture-panel");
  const imageResult = document.getElementById("imageResult");
  const loading = document.getElementById("loadingMessage");
  const resetBtn = document.getElementById("resetBtn");
  const exportBtn = document.getElementById("exportBtn");

  /* ---------------------------------
     FILTER BUTTON LOGIC
  --------------------------------- */
  document.querySelectorAll(".filter").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.type;

      // activate button
      document.querySelectorAll(".filter").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // show only matching textures
      document.querySelectorAll(".texture-card").forEach(card => {
        card.style.display = (card.dataset.type === type) ? "block" : "none";
      });
    });
  });

    /* ---------------------------------
      TEXTURE CLICK (WALL / FLOOR)
    --------------------------------- */
  document.addEventListener("click", async (e) => {

    const card = e.target.closest(".texture-card");
    if (!card) return;

    if (!window.currentUID) {
      alert("Session expired. Please upload again.");
      return;
    }

    if (texturePanel.classList.contains("disabled")) {
      alert("Please upload a room image first.");
      return;
    }

    const img = card.querySelector(".texture-thumb");
    const textureName = img?.dataset.texture;
    const textureType = card.dataset.type;

    if (!textureName || !textureType) {
      console.error("Missing texture data");
      return;
    }

    if (loading) loading.style.display = "block";

    try {
      const response = await fetch("/result_textured", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          uid: window.currentUID,
          texture: textureName,
          type: textureType
        })
      });

      const data = await response.json();

      if (!response.ok || data.state !== "success") {
        throw new Error(data.msg || "Texture application failed");
      }

      // force image refresh
      imageResult.src = data.room_path + "?t=" + Date.now();

    } catch (err) {
      console.error("Texture error:", err);
      alert(err.message);
    } finally {
      if (loading) loading.style.display = "none";
    }
  });

  /* ---------------------------------
     RESET BUTTON
  --------------------------------- */
  resetBtn?.addEventListener("click", async () => {
    if (!window.currentUID) return;

    try {
      const res = await fetch("/reset_texture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uid: window.currentUID })
      });

      const data = await res.json();
      if (data.success) {
        imageResult.src = data.room_path + "?t=" + Date.now();
      }
    } catch (err) {
      console.error("Reset failed:", err);
    }
  });

  /* ---------------------------------
     EXPORT BUTTON
  --------------------------------- */
  exportBtn?.addEventListener("click", () => {
    if (!window.currentUID) return;

    const link = document.createElement("a");
    link.href = `/static/IMG/${window.currentUID}_textured.jpg`;
    link.download = "textured_room.jpg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

});
