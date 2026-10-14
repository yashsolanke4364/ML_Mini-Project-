const API_BASE_URL = String(
  window.MNIST_CONFIG?.apiBaseUrl || window.location.origin
).replace(/\/+$/, '');
const PREDICT_API_URL = `${API_BASE_URL}/api/predict`;
const HEALTH_API_URL = `${API_BASE_URL}/api/health`;
const EVALUATION_API_URL = `${API_BASE_URL}/api/evaluation`;
const EVALUATION_ARTIFACTS_URL = `${EVALUATION_API_URL}/artifacts`;
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
const MAX_HISTORY_ITEMS = 6;
const BRUSH_SIZE_SCALE = 2;

const canvas = document.getElementById('drawingCanvas');
const context = canvas.getContext('2d');
const drawTab = document.getElementById('drawTab');
const uploadTab = document.getElementById('uploadTab');
const drawPane = document.getElementById('drawPane');
const uploadPane = document.getElementById('uploadPane');
const drawTool = document.getElementById('drawTool');
const eraseTool = document.getElementById('eraseTool');
const clearButton = document.getElementById('clearBtn');
const predictButton = document.getElementById('predictBtn');
const predictionStatus = document.getElementById('predictionStatus');
const backendStatus = document.getElementById('backendStatus');
const backendStatusText = document.getElementById('backendStatusText');
const imageUpload = document.getElementById('imageUpload');
const uploadDropzone = document.getElementById('uploadDropzone');
const uploadError = document.getElementById('uploadError');
const previewCard = document.getElementById('previewCard');
const imagePreview = document.getElementById('imagePreview');
const previewName = document.getElementById('previewName');
const previewDimensions = document.getElementById('previewDimensions');
const brushSize = document.getElementById('brushSize');
const brushSizeValue = document.getElementById('brushSizeValue');
const canvasHint = document.getElementById('canvasHint');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const predictionResult = document.getElementById('predictionResult');
const predictedDigit = document.getElementById('predictedDigit');
const resultDigit = document.getElementById('resultDigit');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceFill = document.getElementById('confidenceFill');
const chartFallback = document.getElementById('chartFallback');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');

let isDrawing = false;
let hasCanvasInk = false;
let activeTool = 'draw';
let activeInput = 'draw';
let uploadedImage = null;
let uploadedImageUrl = null;
let probabilityChart = null;
let history = [];
let evaluationCharts = [];
const NOTEBOOK_MODELS = [
  {
    name: 'Baseline ANN',
    architecture: 'Flatten → Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Softmax(10)',
    reason: 'A dense baseline provides a reference without explicit spatial feature extraction.',
  },
  {
    name: 'CNN Config 1 (Standard)',
    architecture: 'Conv2D(32, 3×3, same, ReLU) → MaxPool(2×2) → Conv2D(64, 3×3, same, ReLU) → MaxPool(2×2) → Dense(128, ReLU) → Dropout(0.5) → Softmax(10)',
    reason: 'Two convolution and pooling blocks learn local image features with a compact CNN.',
  },
  {
    name: 'CNN Config 2 (Deep + BN)',
    architecture: 'Two Conv2D(32, 3×3, same, ReLU) + BatchNorm layers → MaxPool(2×2) → Dropout(0.25) → Two Conv2D(64, 3×3, same, ReLU) + BatchNorm layers → MaxPool(2×2) → Dropout(0.25) → Dense(128, ReLU) + BatchNorm → Dropout(0.5) → Softmax(10)',
    reason: 'Additional convolution depth and batch normalization test whether a deeper feature extractor improves held-out results.',
  },
];

function setPredictionStatus(message, state = 'ready') {
  const icon = state === 'error' ? '!' : state === 'loading' ? '' : '\u2713';
  predictionStatus.className = `prediction-status${state === 'ready' ? '' : ` ${state}`}`;
  predictionStatus.replaceChildren();

  const iconElement = document.createElement('span');
  iconElement.className = 'status-check';
  iconElement.setAttribute('aria-hidden', 'true');
  iconElement.textContent = icon;

  const messageElement = document.createElement('span');
  messageElement.textContent = message;
  predictionStatus.append(iconElement, messageElement);
}

function setBackendStatus(state, message) {
  backendStatus.dataset.state = state;
  backendStatusText.textContent = message;
}

function appendDetail(list, label, value) {
  const term = document.createElement('dt');
  term.textContent = label;
  const description = document.createElement('dd');
  description.textContent = value;
  list.append(term, description);
}

function renderDatasetDetails(dataset) {
  const details = document.getElementById('datasetDetails');
  details.replaceChildren();
  appendDetail(details, 'Dataset', dataset?.name || 'MNIST handwritten digits (Keras dataset)');
  appendDetail(details, 'Images', `${dataset?.image_shape?.[0] || 28} × ${dataset?.image_shape?.[1] || 28} grayscale`);
  appendDetail(details, 'Classes', 'Digits 0–9');
  appendDetail(details, 'Original partitions', '60,000 training images (split into 50,000 train + 10,000 validation) and 10,000 test images');
  appendDetail(details, 'Training / validation / test', dataset
    ? `${dataset.training_samples} / ${dataset.validation_samples} / ${dataset.test_samples}`
    : '50,000 / 10,000 / 10,000');
  appendDetail(details, 'Preprocessing', (dataset?.preprocessing || [
    'Convert pixels to float32 and divide by 255, scaling values to [0, 1].',
    'Add a single grayscale channel (28 × 28 × 1).',
    'Create a stratified 10,000-image validation split from the 60,000-image training partition.',
    'Keep the original 10,000-image test partition separate for final evaluation.',
  ]).join(' '));
}

function renderHyperparameters(hyperparameters = {}) {
  const details = document.getElementById('hyperparameterDetails');
  details.replaceChildren();
  const earlyStopping = hyperparameters.early_stopping || {};
  [
    ['Random seed', hyperparameters.seed ?? 42],
    ['Optimizer', hyperparameters.optimizer || 'Adam'],
    ['Learning rate', hyperparameters.learning_rate ?? 0.001],
    ['Loss', hyperparameters.loss || 'Sparse categorical cross-entropy'],
    ['Batch size', hyperparameters.batch_size ?? 128],
    ['Epoch limit', hyperparameters.maximum_epochs ?? 20],
    ['Early stopping', earlyStopping.monitor
      ? `${earlyStopping.monitor}, patience ${earlyStopping.patience}, restore best weights: ${earlyStopping.restore_best_weights}`
      : 'Monitor validation loss; patience 5; restore best weights'],
  ].forEach(([label, value]) => appendDetail(details, label, String(value)));

  const reasons = document.getElementById('hyperparameterReasons');
  reasons.replaceChildren();
  (hyperparameters.reasons || [
    'Adam with a 0.001 starting learning rate provides an adaptive optimizer baseline.',
    'Batch size 128 balances gradient updates with the amount of data processed per step.',
    'Sparse categorical cross-entropy matches integer labels and the ten-class softmax output.',
    'Early stopping monitors validation loss and restores the best weights to limit overfitting.',
    'A fixed seed makes data splitting and training comparisons more repeatable.',
  ]).forEach((reason) => {
    const item = document.createElement('li');
    item.textContent = reason;
    reasons.append(item);
  });
}

function renderModelArchitectures(comparison = []) {
  const cards = document.getElementById('modelArchitectureCards');
  cards.replaceChildren();
  const architectures = new Map(comparison.map((model) => [model.name, model]));
  NOTEBOOK_MODELS.forEach((notebookModel) => {
    const model = architectures.get(notebookModel.name);
    const card = document.createElement('article');
    card.className = 'architecture-card';
    const title = document.createElement('h4');
    title.textContent = notebookModel.name;
    const architecture = document.createElement('p');
    architecture.textContent = model?.architecture || notebookModel.architecture;
    const reason = document.createElement('p');
    reason.className = 'architecture-reason';
    reason.textContent = model?.reason || notebookModel.reason;
    card.append(title, architecture, reason);
    cards.append(card);
  });
}

function formatMetric(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : 'Unavailable';
}

function makeEvaluationChart(canvasId, configuration) {
  const canvasElement = document.getElementById(canvasId);
  if (!window.Chart || !canvasElement) return false;
  evaluationCharts.push(new Chart(canvasElement, configuration));
  return true;
}

function renderComparison(comparison, selectedModel) {
  const modelsByName = new Map((comparison || []).map((model) => [model.name, model]));
  comparison = NOTEBOOK_MODELS.map((model) => ({
    name: model.name,
    ...(modelsByName.get(model.name) || {}),
  }));
  const rows = document.getElementById('comparisonRows');
  rows.replaceChildren();
  const metricKeys = [
    ['validation_accuracy_at_best_epoch', 'Validation accuracy'],
    ['test_accuracy', 'Test accuracy'],
    ['macro_precision', 'Macro precision'],
    ['macro_recall', 'Macro recall'],
    ['macro_f1_score', 'Macro F1'],
  ];

  comparison.forEach((model) => {
    const row = document.createElement('tr');
    if (model.name === selectedModel) row.className = 'selected-model-row';
    const nameCell = document.createElement('th');
    nameCell.scope = 'row';
    nameCell.textContent = model.name;
    row.append(nameCell);
    metricKeys.forEach(([key]) => {
      const cell = document.createElement('td');
      cell.textContent = formatMetric(model[key]);
      row.append(cell);
    });
    rows.append(row);
  });

  const hasComparisonMetrics = comparison.some((model) =>
    metricKeys.some(([key]) => Number.isFinite(model[key])));
  document.getElementById('comparisonChartWrap').hidden = !hasComparisonMetrics;
  if (!hasComparisonMetrics) return false;

  const chartMetricKeys = metricKeys;
  const labels = chartMetricKeys.map(([, label]) => label);
  const colors = ['#2865d8', '#27a776', '#e4a638', '#d64b5b'];
  const comparisonChartLoaded = makeEvaluationChart('comparisonChart', {
    type: 'bar',
    data: {
      labels,
      datasets: comparison.map((model, index) => ({
        label: model.name,
        data: chartMetricKeys.map(([key]) => Number.isFinite(model[key]) ? model[key] * 100 : null),
        backgroundColor: `${colors[index % colors.length]}cc`,
        borderRadius: 5,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Score (%)' } } },
      plugins: { legend: { position: 'bottom' } },
    },
  });
  return comparisonChartLoaded;
}

function renderTrainingCurves(histories) {
  const palette = ['#2865d8', '#27a776', '#d64b5b'];
  const names = Object.keys(histories);
  const accuracyDatasets = [];
  const lossDatasets = [];
  names.forEach((name, index) => {
    const historyData = histories[name];
    const color = palette[index % palette.length];
    [['accuracy', accuracyDatasets, 'train'], ['val_accuracy', accuracyDatasets, 'validation'],
      ['loss', lossDatasets, 'train'], ['val_loss', lossDatasets, 'validation']]
      .forEach(([key, destination, split]) => {
        if (!Array.isArray(historyData[key]) || historyData[key].length === 0) return;
        destination.push({
          label: `${name} · ${split}`,
          data: historyData[key],
          borderColor: color,
          backgroundColor: color,
          borderDash: split === 'validation' ? [5, 4] : [],
          pointRadius: 2,
          tension: 0.2,
        });
      });
  });

  const accuracyChartLoaded = makeEvaluationChart('accuracyCurveChart', {
    type: 'line',
    data: { labels: Array.from({ length: Math.max(0, ...accuracyDatasets.map((item) => item.data.length)) }, (_, i) => i + 1), datasets: accuracyDatasets },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: 'bottom' } } },
  });
  const lossChartLoaded = makeEvaluationChart('lossCurveChart', {
    type: 'line',
    data: { labels: Array.from({ length: Math.max(0, ...lossDatasets.map((item) => item.data.length)) }, (_, i) => i + 1), datasets: lossDatasets },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { position: 'bottom' } } },
  });
  return accuracyChartLoaded && lossChartLoaded;
}

function renderSelectedMetrics(result) {
  const metrics = result.selected_metrics || {};
  const container = document.getElementById('selectedMetrics');
  container.replaceChildren();
  [
    ['Test accuracy', metrics.test_accuracy],
    ['Macro precision', metrics.macro_precision],
    ['Macro recall', metrics.macro_recall],
    ['Macro F1-score', metrics.macro_f1_score],
  ].forEach(([label, value]) => {
    const card = document.createElement('div');
    card.className = 'metric-card';
    const heading = document.createElement('span');
    heading.textContent = label;
    const number = document.createElement('strong');
    number.textContent = formatMetric(value);
    card.append(heading, number);
    container.append(card);
  });
  document.getElementById('selectedModelLabel').textContent = result.selected_model
    ? `Selected by validation: ${result.selected_model}`
    : 'Selected model unavailable';

  const leaders = (result.comparison || [])
    .filter((model) => Number.isFinite(model.test_accuracy))
    .sort((a, b) => b.test_accuracy - a.test_accuracy);
  const selectedComparison = (result.comparison || []).find((model) => model.name === result.selected_model);
  let interpretation = leaders.length === 1 && result.selected_model
    ? `The notebook selected ${result.selected_model} using validation accuracy (${formatMetric(selectedComparison?.validation_accuracy_at_best_epoch)}). Its recorded test accuracy is ${formatMetric(metrics.test_accuracy)} and its macro F1-score is ${formatMetric(metrics.macro_f1_score)}. Test metrics for the other architectures are unavailable in this run's output. ` +
      'Macro scores average each digit class equally, so they show class-balanced performance rather than letting frequent easy classes dominate.'
    : leaders.length > 1 && result.selected_model
      ? `On the held-out test set, ${leaders[0].name} has the highest measured accuracy (${formatMetric(leaders[0].test_accuracy)}). The notebook selected ${result.selected_model} using validation accuracy; its macro F1-score is ${formatMetric(metrics.macro_f1_score)}. Macro scores average each digit class equally, so they show class-balanced performance rather than letting frequent easy classes dominate.`
    : 'A findings interpretation is unavailable because the training output does not include the model comparison or selected-model test results.';
  const matrix = result.confusion_matrix;
  if (Array.isArray(matrix) && matrix.length > 1) {
    let largestConfusion = { count: 0, actual: null, predicted: null };
    matrix.forEach((row, actual) => row.forEach((count, predicted) => {
      if (actual !== predicted && count > largestConfusion.count) {
        largestConfusion = { count, actual, predicted };
      }
    }));
    if (largestConfusion.count > 0) {
      interpretation += ` The most frequent off-diagonal error for the selected model was true ${largestConfusion.actual} predicted as ${largestConfusion.predicted} (${largestConfusion.count} test images).`;
    }
  }
  document.getElementById('evaluationInterpretation').textContent = interpretation;
}

function renderConfusionMatrix(matrix) {
  const table = document.getElementById('confusionMatrix');
  table.replaceChildren();
  if (!Array.isArray(matrix) || matrix.length === 0) {
    table.hidden = true;
    document.getElementById('confusionMatrixUnavailable').hidden = false;
    return;
  }
  table.hidden = false;
  document.getElementById('confusionMatrixUnavailable').hidden = true;
  const maximum = Math.max(1, ...matrix.flat());
  const header = document.createElement('tr');
  const corner = document.createElement('th');
  corner.textContent = 'True \\ Predicted';
  header.append(corner);
  matrix.forEach((_, digit) => {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = String(digit);
    header.append(cell);
  });
  const head = document.createElement('thead');
  head.append(header);
  const body = document.createElement('tbody');
  matrix.forEach((rowValues, rowIndex) => {
    const row = document.createElement('tr');
    const label = document.createElement('th');
    label.scope = 'row';
    label.textContent = String(rowIndex);
    row.append(label);
    rowValues.forEach((value) => {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      cell.style.backgroundColor = `rgba(40, 101, 216, ${0.06 + 0.72 * value / maximum})`;
      cell.style.color = value / maximum > 0.55 ? '#fff' : '#172238';
      row.append(cell);
    });
    body.append(row);
  });
  table.append(head, body);
}

function loadEvaluationPlot(imageId, fallbackId, filename) {
  const image = document.getElementById(imageId);
  const fallback = document.getElementById(fallbackId);
  image.hidden = true;
  fallback.hidden = true;
  image.onload = () => {
    image.hidden = false;
    fallback.hidden = true;
  };
  image.onerror = () => {
    image.hidden = true;
    fallback.hidden = false;
  };
  image.src = `${EVALUATION_ARTIFACTS_URL}/${filename}?v=${Date.now()}`;
}

async function loadEvaluation() {
  const status = document.getElementById('evaluationStatus');
  const message = document.getElementById('evaluationMessage');
  renderDatasetDetails();
  renderHyperparameters();
  renderModelArchitectures();
  renderComparison(NOTEBOOK_MODELS.map((model) => ({ name: model.name })));
  try {
    const response = await fetch(EVALUATION_API_URL);
    const result = await response.json();
    if (!response.ok || result.available !== true) {
      throw new Error(result.message || 'Evaluation results are not available yet.');
    }

    renderDatasetDetails(result.dataset);
    renderHyperparameters(result.hyperparameters);
    renderModelArchitectures(result.comparison);
    const comparisonChartAvailable = renderComparison(result.comparison, result.selected_model);
    const curvesAvailable = renderTrainingCurves(result.training_histories);
    renderSelectedMetrics(result);
    renderConfusionMatrix(result.confusion_matrix);
    loadEvaluationPlot('trainingCurvesImage', 'trainingCurvesUnavailable', 'training_curves.png');
    loadEvaluationPlot('confusionMatrixImage', 'confusionMatrixUnavailable', 'confusion_matrix.png');
    loadEvaluationPlot('predictionExamplesImage', 'predictionExamplesUnavailable', 'predictions_analysis.png');
    document.getElementById('evaluationResults').hidden = false;
    document.getElementById('evaluationChartNotice').hidden = comparisonChartAvailable && curvesAvailable;
    status.textContent = 'Notebook results loaded';
  } catch (error) {
    message.textContent = error.message || 'Evaluation results could not be loaded.';
    message.hidden = false;
    status.textContent = 'Evaluation results unavailable';
  }
}

async function checkBackend() {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(HEALTH_API_URL, { signal: controller.signal });
    let health;
    try {
      health = await response.json();
    } catch {
      throw new Error('The API returned an invalid health response.');
    }

    if (response.ok && health.model_loaded === true) {
      setBackendStatus('connected', 'Backend connected');
    } else if (health.model_loaded === false) {
      setBackendStatus('unavailable', 'Model unavailable');
    } else {
      setBackendStatus('unavailable', 'Backend unavailable');
    }
  } catch {
    setBackendStatus('offline', 'Backend offline');
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function fillCanvasBlack() {
  context.save();
  context.globalCompositeOperation = 'source-over';
  context.fillStyle = '#000000';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.restore();
}

function resizeCanvas() {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;

  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const nextWidth = Math.round(bounds.width * pixelRatio);
  const nextHeight = Math.round(bounds.height * pixelRatio);
  if (canvas.width === nextWidth && canvas.height === nextHeight) return;

  const previous = document.createElement('canvas');
  previous.width = canvas.width;
  previous.height = canvas.height;
  previous.getContext('2d').drawImage(canvas, 0, 0);

  canvas.width = nextWidth;
  canvas.height = nextHeight;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.lineWidth = Number(brushSize.value) * pixelRatio;
  fillCanvasBlack();
  if (previous.width && previous.height) {
    context.drawImage(previous, 0, 0, previous.width, previous.height, 0, 0, canvas.width, canvas.height);
  }
}

function resetCanvas() {
  fillCanvasBlack();
  hasCanvasInk = false;
  canvasHint.classList.remove('is-hidden');
}

function pointerPosition(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * (canvas.width / bounds.width),
    y: (event.clientY - bounds.top) * (canvas.height / bounds.height),
  };
}

function applyBrushStyle() {
  const pixelRatio = canvas.width / canvas.getBoundingClientRect().width;
  context.lineWidth = Number(brushSize.value) * pixelRatio;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.strokeStyle = activeTool === 'erase' ? '#000000' : '#ffffff';
}

function startDrawing(event) {
  if (activeInput !== 'draw' || event.button > 0) return;
  isDrawing = true;
  canvas.setPointerCapture(event.pointerId);
  const point = pointerPosition(event);
  applyBrushStyle();
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineTo(point.x + 0.01, point.y + 0.01);
  context.stroke();
  hasCanvasInk = true;
  canvasHint.classList.add('is-hidden');
  event.preventDefault();
}

function continueDrawing(event) {
  if (!isDrawing) return;
  const point = pointerPosition(event);
  context.lineTo(point.x, point.y);
  context.stroke();
  event.preventDefault();
}

function stopDrawing(event) {
  if (!isDrawing) return;
  isDrawing = false;
  context.closePath();
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function setActiveTab(tab) {
  const isDraw = tab === 'draw';
  activeInput = isDraw ? 'draw' : 'upload';
  drawTab.classList.toggle('active', isDraw);
  uploadTab.classList.toggle('active', !isDraw);
  drawTab.setAttribute('aria-selected', String(isDraw));
  uploadTab.setAttribute('aria-selected', String(!isDraw));
  drawPane.hidden = !isDraw;
  uploadPane.hidden = isDraw;
  setPredictionStatus(isDraw ? 'Ready when you are' : 'Choose an image to continue.');
  window.requestAnimationFrame(resizeCanvas);
}

function setActiveTool(tool) {
  activeTool = tool;
  const drawing = tool === 'draw';
  drawTool.classList.toggle('active', drawing);
  eraseTool.classList.toggle('active', !drawing);
  drawTool.setAttribute('aria-pressed', String(drawing));
  eraseTool.setAttribute('aria-pressed', String(!drawing));
  canvas.setAttribute(
    'aria-label',
    drawing
      ? 'Black drawing canvas. Draw a single white handwritten digit from zero to nine.'
      : 'Black drawing canvas. Erase with the black brush.',
  );
}

function clearResults() {
  predictionResult.hidden = true;
  resultPlaceholder.hidden = false;
  predictedDigit.textContent = '-';
  resultDigit.textContent = '-';
  confidenceValue.textContent = '--%';
  confidenceFill.style.width = '0%';
  confidenceBar.setAttribute('aria-valuenow', '0');
  updateChart(Array(10).fill(0));
}

function clearInput() {
  resetCanvas();
  uploadedImage = null;
  imageUpload.value = '';
  uploadError.hidden = true;
  uploadError.textContent = '';
  previewCard.hidden = true;
  imagePreview.removeAttribute('src');
  if (uploadedImageUrl) {
    URL.revokeObjectURL(uploadedImageUrl);
    uploadedImageUrl = null;
  }
  setActiveTab('draw');
  clearResults();
  setPredictionStatus('Ready when you are');
}

function showUploadError(message) {
  uploadError.textContent = message;
  uploadError.hidden = false;
  setPredictionStatus(message, 'error');
}

function validateImageFile(file) {
  const extension = file.name.split('.').pop().toLowerCase();
  const supportedType = ['image/png', 'image/jpeg'].includes(file.type);
  const supportedExtension = ['png', 'jpg', 'jpeg'].includes(extension);

  if (!supportedType && !supportedExtension) {
    throw new Error('Choose a PNG, JPG or JPEG image.');
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error('Image is too large. Choose a file smaller than 5 MB.');
  }
  if (file.size === 0) {
    throw new Error('The selected image is empty.');
  }
}

function handleImageFile(file) {
  if (!file) return;
  uploadError.hidden = true;
  uploadError.textContent = '';

  try {
    validateImageFile(file);
  } catch (error) {
    imageUpload.value = '';
    showUploadError(error.message);
    return;
  }

  if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
  uploadedImageUrl = URL.createObjectURL(file);
  uploadedImage = file;
  imagePreview.src = uploadedImageUrl;
  previewName.textContent = file.name;
  previewDimensions.textContent = `${(file.size / 1024).toFixed(0)} KB`;
  previewCard.hidden = false;
  setActiveTab('upload');
  setPredictionStatus('Image ready to predict.');
}

function fileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') resolve(reader.result);
      else reject(new Error('Could not read the selected image.'));
    });
    reader.addEventListener('error', () => reject(new Error('Could not read the selected image.')));
    reader.readAsDataURL(file);
  });
}

function makeHistoryThumbnail(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener('load', () => {
      const thumb = document.createElement('canvas');
      thumb.width = 90;
      thumb.height = 90;
      const thumbContext = thumb.getContext('2d');
      thumbContext.fillStyle = '#000000';
      thumbContext.fillRect(0, 0, thumb.width, thumb.height);
      const scale = Math.min(74 / image.naturalWidth, 74 / image.naturalHeight);
      const width = image.naturalWidth * scale;
      const height = image.naturalHeight * scale;
      thumbContext.drawImage(image, (90 - width) / 2, (90 - height) / 2, width, height);
      resolve(thumb.toDataURL('image/jpeg', 0.72));
    }, { once: true });
    image.addEventListener('error', () => reject(new Error('Could not create a history thumbnail.')), { once: true });
    image.src = source;
  });
}

function renderHistory() {
  historyCount.textContent = `${history.length} ${history.length === 1 ? 'prediction' : 'predictions'}`;
  historyList.replaceChildren();

  if (!history.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.innerHTML = '<span aria-hidden="true">&#8635;</span><p>Your predictions will show up here.</p>';
    historyList.append(empty);
    return;
  }

  history.forEach((item) => {
    const button = document.createElement('button');
    button.className = 'history-item';
    button.type = 'button';
    button.setAttribute('aria-label', `Review prediction ${item.prediction} with ${item.confidence.toFixed(1)} percent confidence`);

    const thumbnail = document.createElement('img');
    thumbnail.className = 'history-thumb';
    thumbnail.src = item.thumbnail;
    thumbnail.alt = '';

    const info = document.createElement('span');
    info.className = 'history-info';
    const digit = document.createElement('span');
    digit.className = 'history-digit';
    digit.textContent = `Digit ${item.prediction}`;
    const meta = document.createElement('span');
    meta.className = 'history-meta';
    meta.textContent = `${item.confidence.toFixed(1)}% confidence \u00b7 ${item.time}`;
    info.append(digit, meta);
    button.append(thumbnail, info);
    button.addEventListener('click', () => showResult(item));
    historyList.append(button);
  });
}

function validatePrediction(data) {
  if (
    !data ||
    !Number.isInteger(data.prediction) ||
    data.prediction < 0 ||
    data.prediction > 9 ||
    typeof data.confidence_percentage !== 'number' ||
    !Number.isFinite(data.confidence_percentage) ||
    data.confidence_percentage < 0 ||
    data.confidence_percentage > 100 ||
    !Array.isArray(data.probabilities_list) ||
    data.probabilities_list.length !== 10 ||
    data.probabilities_list.some(
      (value) => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1,
    )
  ) {
    throw new Error('The backend returned incomplete prediction data. Restart the backend and try again.');
  }
}

function updateChart(probabilities) {
  if (!window.Chart) {
    chartFallback.hidden = false;
    return;
  }

  chartFallback.hidden = true;
  if (!probabilityChart) {
    const chartContext = document.getElementById('probabilityChart').getContext('2d');
    probabilityChart = new Chart(chartContext, {
      type: 'bar',
      data: {
        labels: Array.from({ length: 10 }, (_, digit) => String(digit)),
        datasets: [{
          label: 'Probability',
          data: probabilities.map((value) => value * 100),
          backgroundColor: probabilities.map((_, digit) => digit === 0 ? '#3678ef' : '#dce7fa'),
          hoverBackgroundColor: '#3678ef',
          borderRadius: 4,
          borderSkipped: false,
          barPercentage: 0.67,
          categoryPercentage: 0.78,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            displayColors: false,
            callbacks: {
              label: (context) => ` ${context.parsed.x.toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            grid: { color: '#edf1f7', drawTicks: false },
            border: { display: false },
            ticks: {
              color: '#96a1b1',
              font: { family: 'DM Sans', size: 9 },
              stepSize: 25,
              callback: (value) => `${value}%`,
            },
          },
          y: {
            grid: { display: false },
            border: { display: false },
            ticks: { color: '#718097', font: { family: 'DM Sans', size: 10, weight: '600' } },
          },
        },
      },
    });
  } else {
    probabilityChart.data.datasets[0].data = probabilities.map((value) => value * 100);
  }

  const highestDigit = probabilities.indexOf(Math.max(...probabilities));
  probabilityChart.data.datasets[0].backgroundColor = probabilities.map(
    (_, digit) => digit === highestDigit ? '#3678ef' : '#dce7fa',
  );
  probabilityChart.update();
}

function showResult(item) {
  resultPlaceholder.hidden = true;
  predictionResult.hidden = false;
  predictedDigit.textContent = String(item.prediction);
  resultDigit.textContent = String(item.prediction);
  confidenceValue.textContent = `${item.confidence.toFixed(2)}%`;
  confidenceFill.style.width = `${Math.min(100, Math.max(0, item.confidence))}%`;
  confidenceBar.setAttribute('aria-valuenow', String(Math.round(item.confidence)));
  updateChart(item.probabilities);
  setPredictionStatus(`Showing prediction for digit ${item.prediction}.`, 'success');
}

async function addHistoryItem(result, imageData) {
  const item = {
    prediction: result.prediction,
    confidence: result.confidence_percentage,
    probabilities: result.probabilities_list,
    thumbnail: await makeHistoryThumbnail(imageData),
    time: new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date()),
  };
  history = [item, ...history].slice(0, MAX_HISTORY_ITEMS);
  renderHistory();
  showResult(item);
}

async function predictDigit() {
  if (activeInput === 'draw' && !hasCanvasInk) {
    setPredictionStatus('Draw a digit on the canvas before predicting.', 'error');
    return;
  }
  if (activeInput === 'upload' && !uploadedImage) {
    showUploadError('Choose a PNG, JPG or JPEG image before predicting.');
    return;
  }

  predictButton.disabled = true;
  predictButton.classList.add('is-loading');
  predictButton.querySelector('.button-label').textContent = 'Analyzing...';
  setPredictionStatus('Sending your image to the trained model...', 'loading');

  try {
    const imageData = activeInput === 'upload'
      ? await fileAsDataUrl(uploadedImage)
      : canvas.toDataURL('image/png');
    const response = await fetch(PREDICT_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`The backend returned an invalid response (HTTP ${response.status}).`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || `Prediction failed (HTTP ${response.status}).`);
    }

    validatePrediction(data);
    await addHistoryItem(data, imageData);
    setPredictionStatus('Prediction received from your model.', 'success');
  } catch (error) {
    console.error('Prediction request failed:', error);
    const message = error instanceof TypeError
      ? `Could not connect to ${API_BASE_URL}. Confirm the Flask API is running and its CORS allows this page.`
      : error.message || 'Prediction failed. Please try again.';
    setPredictionStatus(message, 'error');
    await checkBackend();
  } finally {
    predictButton.disabled = false;
    predictButton.classList.remove('is-loading');
    predictButton.querySelector('.button-label').textContent = 'Predict digit';
  }
}

drawTab.addEventListener('click', () => setActiveTab('draw'));
uploadTab.addEventListener('click', () => setActiveTab('upload'));
drawTool.addEventListener('click', () => setActiveTool('draw'));
eraseTool.addEventListener('click', () => setActiveTool('erase'));
clearButton.addEventListener('click', clearInput);
predictButton.addEventListener('click', predictDigit);

canvas.addEventListener('pointerdown', startDrawing);
canvas.addEventListener('pointermove', continueDrawing);
canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointercancel', stopDrawing);
canvas.addEventListener('lostpointercapture', () => {
  isDrawing = false;
});

brushSize.addEventListener('input', () => {
  brushSizeValue.textContent = `${brushSize.value} px`;
  applyBrushStyle();
});

imageUpload.addEventListener('change', () => handleImageFile(imageUpload.files[0]));
document.getElementById('removeUpload').addEventListener('click', () => {
  uploadedImage = null;
  imageUpload.value = '';
  previewCard.hidden = true;
  imagePreview.removeAttribute('src');
  uploadError.hidden = true;
  if (uploadedImageUrl) {
    URL.revokeObjectURL(uploadedImageUrl);
    uploadedImageUrl = null;
  }
  setPredictionStatus('Choose an image to continue.');
});

uploadDropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadDropzone.classList.add('drag-over');
});
uploadDropzone.addEventListener('dragleave', () => uploadDropzone.classList.remove('drag-over'));
uploadDropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadDropzone.classList.remove('drag-over');
  handleImageFile(event.dataTransfer.files[0]);
});

const canvasResizeObserver = new ResizeObserver(resizeCanvas);
canvasResizeObserver.observe(canvas);
window.addEventListener('resize', resizeCanvas);

setActiveTool('draw');
renderHistory();
resizeCanvas();
resetCanvas();
updateChart(Array(10).fill(0));
checkBackend();
loadEvaluation();
window.setInterval(checkBackend, 15000);
