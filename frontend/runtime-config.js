(() => {
  const host = window.location.hostname;
  const isLocal = host === 'localhost' || host === '127.0.0.1';
  window.MNIST_CONFIG = Object.freeze({
    apiBaseUrl: isLocal ? `${window.location.protocol}//${host}:5000` : '',
  });
})();
