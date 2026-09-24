#!/usr/bin/env node
/**
 * Capture transparent PNG frames from the Three.js Ophanim scene via headless Chrome.
 */
const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const puppeteer = require('puppeteer-core');

const ROOT = '/workspace/ophanim';
const PORT = 8765;
const DURATION = 10;
const FPS = 24;
const N = DURATION * FPS;

const variant = process.argv[2] || 'computador';
const previewOnly = process.argv.includes('--preview');
const size =
  variant === 'celular'
    ? { w: 1080, h: 1920 }
    : { w: 1920, h: 1080 };

const outDir = path.join(ROOT, 'frames', variant);
fs.mkdirSync(outDir, { recursive: true });

function startServer() {
  const mime = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
  };
  const server = http.createServer((req, res) => {
    let urlPath = decodeURIComponent(req.url.split('?')[0]);
    if (urlPath === '/') urlPath = '/src/scene.html';
    const filePath = path.join(ROOT, urlPath.replace(/^\//, ''));
    // also allow /node_modules from ROOT
    const alt =
      urlPath.startsWith('/node_modules/')
        ? path.join(ROOT, urlPath.replace(/^\//, ''))
        : filePath;
    const target = fs.existsSync(alt) ? alt : filePath;
    fs.readFile(target, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('not found ' + urlPath);
        return;
      }
      const ext = path.extname(target);
      res.writeHead(200, { 'Content-Type': mime[ext] || 'application/octet-stream' });
      res.end(data);
    });
  });
  return new Promise((resolve) => server.listen(PORT, '127.0.0.1', () => resolve(server)));
}

async function main() {
  const server = await startServer();
  const browser = await puppeteer.launch({
    executablePath: '/usr/local/bin/google-chrome',
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--hide-scrollbars',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: size.w, height: size.h, deviceScaleFactor: 1 });
  const url = `http://127.0.0.1:${PORT}/src/scene.html?w=${size.w}&h=${size.h}&variant=${variant}`;
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => window.__READY__ === true, { timeout: 120000 });

  const frames = previewOnly
    ? [0, Math.floor(0.25 * N), Math.floor(0.5 * N), Math.floor(0.75 * N)]
    : [...Array(N).keys()];

  const t0 = Date.now();
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const dataUrl = await page.evaluate(async (fi) => {
      return window.renderFrame(fi);
    }, f);
    const b64 = dataUrl.split(',')[1];
    const name = previewOnly
      ? `preview_${[0, 25, 50, 75][i]}.png`
      : `frame_${String(f).padStart(4, '0')}.png`;
    fs.writeFileSync(path.join(outDir, name), Buffer.from(b64, 'base64'));
    if (i % 10 === 0 || previewOnly) {
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`frame ${f} (${i + 1}/${frames.length}) ${dt}s`);
    }
  }
  console.log('CAPTURE_DONE', variant, ((Date.now() - t0) / 1000).toFixed(1) + 's');
  await browser.close();
  server.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
