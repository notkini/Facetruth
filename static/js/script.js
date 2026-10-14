document.addEventListener('DOMContentLoaded', () => {
  const dropArea = document.getElementById('drop-area');
  const fileInput = document.getElementById('fileInput');
  const fileName = document.getElementById('file-name');
  const preview = document.getElementById('file-preview');
  const form = document.getElementById('upload-form');
  if (!dropArea || !fileInput) return;

  let previewUrl;
  const showFile = file => {
    fileName.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : '';
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = file && file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    if (preview) {
      preview.src = previewUrl || '';
      preview.hidden = !previewUrl;
    }
  };
  fileInput.addEventListener('change', () => showFile(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach(eventName =>
    dropArea.addEventListener(eventName, event => {
      event.preventDefault();
      dropArea.classList.add('dragover');
    })
  );
  ['dragleave', 'drop'].forEach(eventName =>
    dropArea.addEventListener(eventName, event => {
      event.preventDefault();
      dropArea.classList.remove('dragover');
    })
  );
  dropArea.addEventListener('drop', event => {
    const files = event.dataTransfer.files;
    if (files.length) {
      fileInput.files = files;
      showFile(files[0]);
    }
  });
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    if (fileInput.files.length && button) {
      button.disabled = true;
      button.querySelector('span:first-child').textContent = 'Analyzing…';
      const loading = document.getElementById('analysis-loading');
      if (loading) loading.hidden = false;
    }
  });
  const results = document.getElementById('results');
  if (results)
    requestAnimationFrame(() => results.scrollIntoView({ behavior: 'smooth', block: 'start' }));
});
