document.getElementById("open-camera").addEventListener("click", () => {
    alert("Camera functionality will be added later.");
});

document.getElementById("upload-image").addEventListener("click", () => {
    document.getElementById("file-input").click();
});

document.getElementById("file-input").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById("preview");
            preview.src = e.target.result;
            preview.style.display = "block";
            predictAnimal(file);
        };
        reader.readAsDataURL(file);
    }
});

function predictAnimal(file) {
    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("prediction").innerText = `Recognized Animal: ${data.animal}`;
    })
    .catch(error => {
        console.error("Error:", error);
    });
}