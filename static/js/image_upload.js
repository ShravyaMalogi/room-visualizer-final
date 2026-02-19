document.addEventListener("DOMContentLoaded", () => {

  const uploadInput = document.getElementById("upload");
  const loading = document.getElementById("loadingMessage");
  const texturePanel = document.querySelector(".texture-panel");

  if (!uploadInput) return;

  uploadInput.addEventListener("change", async () => {
    if (!uploadInput.files || uploadInput.files.length === 0) return;

    const file = uploadInput.files[0];

    // SAFETY: ensure it's an image
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      uploadInput.value = "";
      return;
    }

    // Show loading
    if (loading) loading.style.display = "block";

    // Disable input to prevent double upload
    uploadInput.disabled = true;

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/prediction", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Upload failed");
      }

      // 🔥 Redirect safely AFTER server finishes processing
      window.location.href = data.redirect_url;

    } catch (err) {
      console.error("Upload error:", err);
      alert("Upload failed. Please try again.");

      uploadInput.disabled = false;
      if (loading) loading.style.display = "none";
    }
  });

});
