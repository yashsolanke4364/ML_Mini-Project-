const fs = require('node:fs');
const path = require('node:path');

const apiBaseUrl = process.env.MNIST_API_BASE_URL?.trim().replace(/\/+$/, '');
if (!apiBaseUrl) {
  throw new Error('Set MNIST_API_BASE_URL to the deployed HTTPS Flask API URL.');
}

let parsedUrl;
try {
  parsedUrl = new URL(apiBaseUrl);
} catch {
  throw new Error('MNIST_API_BASE_URL must be a valid absolute HTTPS URL.');
}
if (
  parsedUrl.protocol !== 'https:' ||
  parsedUrl.pathname !== '/' ||
  parsedUrl.search ||
  parsedUrl.hash ||
  parsedUrl.username ||
  parsedUrl.password
) {
  throw new Error('MNIST_API_BASE_URL must be an HTTPS origin without a path, query, or fragment.');
}

const configPath = path.join(__dirname, '..', 'frontend', 'runtime-config.js');
const config = `window.MNIST_CONFIG = Object.freeze({ apiBaseUrl: ${JSON.stringify(apiBaseUrl)} });\n`;
fs.writeFileSync(configPath, config, 'utf8');
console.log(`Configured static frontend for ${parsedUrl.origin}`);
