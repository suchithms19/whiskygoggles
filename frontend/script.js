document.addEventListener("DOMContentLoaded", () => {
  const fileUpload = document.getElementById("file-upload")
  const uploadContainer = document.getElementById("upload-container")
  const uploadContent = document.getElementById("upload-content")
  const previewContainer = document.getElementById("preview-container")
  const previewImage = document.getElementById("preview-image")
  const removeButton = document.getElementById("remove-image")
  const submitButton = document.getElementById("submit-button")
  const resultsSection = document.getElementById("results-section")
  const bottlesContainer = document.querySelector(".bottles-container")
  const loadingOverlay = document.querySelector(".loading-overlay")

  let selectedFile = null;

  // Handle file selection
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0]
    handleFileSelect(file)
  })

  function handleFileSelect(file) {
    if (file && file.type.match("image.*")) {
      selectedFile = file
      const reader = new FileReader()

      reader.onload = (e) => {
        previewImage.src = e.target.result
        uploadContent.style.display = "none"
        previewContainer.style.display = "block"
        submitButton.disabled = false
      }

      reader.readAsDataURL(file)
    }
  }

  // Handle drag and drop
  ;["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    uploadContainer.addEventListener(eventName, preventDefaults, false)
  })

  function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
  }
  ;["dragenter", "dragover"].forEach((eventName) => {
    uploadContainer.addEventListener(eventName, highlight, false)
  })
  ;["dragleave", "drop"].forEach((eventName) => {
    uploadContainer.addEventListener(eventName, unhighlight, false)
  })

  function highlight() {
    uploadContainer.classList.add("highlight")
  }

  function unhighlight() {
    uploadContainer.classList.remove("highlight")
  }

  uploadContainer.addEventListener("drop", handleDrop, false)

  function handleDrop(e) {
    const dt = e.dataTransfer
    const file = dt.files[0]
    handleFileSelect(file)
  }

  // Remove image
  removeButton.addEventListener("click", () => {
    previewImage.src = "#"
    fileUpload.value = ""
    selectedFile = null
    previewContainer.style.display = "none"
    uploadContent.style.display = "flex"
    submitButton.disabled = true
    resultsSection.style.display = "none"
  })

  // Submit button
  submitButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    submitButton.disabled = true;
    loadingOverlay.classList.add('active');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/api/recognize', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        // Clear existing results
        bottlesContainer.innerHTML = '';

        // Add new results
        data.results.forEach(result => {
          const bottleHtml = `
            <div class="bottle">
              <div class="bottle-image">
                <img src="${result.image_url}" alt="${result.name}" onerror="this.src='bottle1.svg'">
              </div>
              <div class="bottle-info">
                <h3>${result.name}</h3>
                <div class="details">
                  <p><strong>Type:</strong> ${result.spirit_type || 'N/A'}</p>
                  <p><strong>Size:</strong> ${result.size || 'N/A'}</p>
                  <p><strong>ABV:</strong> ${result.abv ? result.abv + '%' : 'N/A'} ${result.proof ? `(${result.proof} proof)` : ''}</p>
                  
                  <div class="prices-section">
                    <h4>Prices</h4>
                    <ul class="price-list">
                      <li>
                        <span>Average MSRP</span>
                        <span>${result.avg_msrp ? '$' + result.avg_msrp : 'N/A'}</span>
                      </li>
                      <li>
                        <span>Fair Price</span>
                        <span>${result.fair_price ? '$' + result.fair_price : 'N/A'}</span>
                      </li>
                      <li>
                        <span>Shelf Price</span>
                        <span>${result.shelf_price ? '$' + result.shelf_price : 'N/A'}</span>
                      </li>
                    </ul>
                  </div>

                  <div class="stats-section">
                    <h4>Stats</h4>
                    <ul class="stats-list">
                      <li>Total Score: ${result.total_score || 'N/A'}</li>
                      <li>Ranking: ${result.ranking ? '#' + result.ranking : 'N/A'}</li>
                      <li>Wishlist: ${result.wishlist_count}</li>
                      <li>Votes: ${result.vote_count}</li>
                    </ul>
                  </div>

                  <div>
                    <span class="confidence">Match Confidence: ${result.confidence.toFixed(1)}%</span>
                    <span class="text-score">Text Match: ${(result.text_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          `;
          bottlesContainer.insertAdjacentHTML('beforeend', bottleHtml);
        });

        resultsSection.style.display = "block";
        resultsSection.scrollIntoView({ behavior: "smooth" });
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert('Error processing image. Please try again.');
      console.error('Error:', error);
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Recognize Spirit";
      loadingOverlay.classList.remove('active');
    }
  });
})
