// server.js - Projeto completo demo com protecoes
const express = require('express');
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const crypto = require('crypto');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { open } = require('sqlite');
const sqlite3 = require('sqlite3');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
app.use(bodyParser.json());
app.use(cookieParser());
app.use(express.static('static'));

// CONFIG - substituir em produção via env vars
const JWT_SECRET = process.env.JWT_SECRET || 'troque_para_uma_chave_muito_forte';
const HMAC_SECRET = process.env.HMAC_SECRET || 'troque_esta_chave_hmac_para_gerar_codigos';
const SITE_HASH = process.env.SITE_HASH || 'site-version-hash-001';
const PORT = process.env.PORT || 3000;
const DB_FILE = './data.db';
const allowedDomains = (process.env.ALLOWED_DOMAINS || 'localhost:3000,seudominio.com').split(',');

// Middleware: headers de segurança globais
app.use((req, res, next) => {
  // Anti-iframe
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; connect-src 'self'; media-src 'self'; frame-ancestors 'none';");
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// DB init
let db;
(async () => {
  db = await open({ filename: DB_FILE, driver: sqlite3.Database });
  await db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      name TEXT,
      email TEXT UNIQUE,
      password_hash TEXT
    );
    CREATE TABLE IF NOT EXISTS courses (
      id TEXT PRIMARY KEY,
      title TEXT,
      description TEXT,
      file_path TEXT
    );
    CREATE TABLE IF NOT EXISTS purchases (
      id TEXT PRIMARY KEY,
      user_id TEXT,
      course_id TEXT,
      purchased_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      user_id TEXT,
      device_hash TEXT,
      ip TEXT,
      user_agent TEXT,
      created_at INTEGER
    );
  `);

  // cria um curso demo se não existir
  const exists = await db.get(`SELECT id FROM courses LIMIT 1`);
  if (!exists) {
    const cid = uuidv4();
    await db.run(`INSERT INTO courses (id,title,description,file_path) VALUES (?,?,?,?)`,
      [cid, 'Curso Demo', 'Curso de exemplo protegido', './media/curso-demo.mp4']);
    console.log('curso demo criado com id', cid);
  }
})();

// utilitarios
function generateToken(user) {
  return jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' });
}
function getWeekKey() {
  // Retorna string ano + numero da semana ISO (simplificado)
  const now = new Date();
  const year = now.getUTCFullYear();
  const start = new Date(Date.UTC(year, 0, 1));
  const diff = Math.floor((now - start) / 86400000);
  const week = Math.ceil((diff + start.getUTCDay() + 1) / 7);
  return `${year}w${week}`;
}
function generateWeeklyCodeForUser(userId) {
  const weekKey = getWeekKey();
  const h = crypto.createHmac('sha256', HMAC_SECRET)
    .update(userId + '|' + weekKey)
    .digest('hex');
  return h.slice(0, 8).toUpperCase();
}
function domainAllowed(host) {
  if (!host) return false;
  return allowedDomains.includes(host);
}
function makeAccessJwt(payload, expiresInSeconds = 600) {
  const token = jwt.sign({ ...payload, exp: Math.floor(Date.now() / 1000) + expiresInSeconds }, JWT_SECRET);
  return token;
}

// auth middleware
async function authMiddleware(req, res, next) {
  const auth = req.headers.authorization || req.cookies.token;
  if (!auth) return res.status(401).json({ error: 'Não autenticado' });
  const token = auth.startsWith('Bearer ') ? auth.split(' ')[1] : auth;
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    const user = await db.get('SELECT id,name,email FROM users WHERE id = ?', payload.id);
    if (!user) return res.status(401).json({ error: 'Usuário não encontrado' });
    req.user = user;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Token inválido' });
  }
}

// helper device fingerprint (simples): combine userAgent + accept-language + resolution (client provides)
function deviceHashFromClient(reqBody, userAgent) {
  const seed = (reqBody.deviceId || '') + '|' + (reqBody.screen || '') + '|' + userAgent;
  return crypto.createHash('sha256').update(seed).digest('hex').slice(0, 16);
}

// Routes

// Register
app.post('/api/register', async (req, res) => {
  const { name, email, password } = req.body;
  if (!email || !password) return res.status(400).json({ error: 'email e senha obrigatórios' });
  const password_hash = await bcrypt.hash(password, 10);
  const id = uuidv4();
  try {
    await db.run('INSERT INTO users (id,name,email,password_hash) VALUES (?,?,?,?)', [id, name||'', email, password_hash]);
    res.json({ ok: true });
  } catch (err) {
    res.status(400).json({ error: 'Email já cadastrado' });
  }
});

// Login - cria session (limite de dispositivos)
app.post('/api/login', async (req, res) => {
  const { email, password, deviceId, screen } = req.body;
  const user = await db.get('SELECT * FROM users WHERE email = ?', email);
  if (!user) return res.status(400).json({ error: 'Credenciais inválidas' });
  const match = await bcrypt.compare(password, user.password_hash);
  if (!match) return res.status(400).json({ error: 'Credenciais inválidas' });

  // device hash
  const device_hash = deviceHashFromClient({ deviceId, screen }, req.headers['user-agent'] || '');
  // count devices active
  const devices = await db.all('SELECT * FROM sessions WHERE user_id = ?', user.id);
  const MAX_DEVICES = 2;
  // se device novo e ultrapassa, remove o mais antigo
  const found = devices.find(d => d.device_hash === device_hash);
  if (!found && devices.length >= MAX_DEVICES) {
    // opcional: recusar novo device invés de remover
    // vamos remover o mais antigo
    const oldest = devices.sort((a,b)=>a.created_at - b.created_at)[0];
    await db.run('DELETE FROM sessions WHERE id = ?', oldest.id);
  }
  const sessionId = uuidv4();
  await db.run('INSERT INTO sessions (id,user_id,device_hash,ip,user_agent,created_at) VALUES (?,?,?,?,?,?)',
    [sessionId, user.id, device_hash, req.ip, req.headers['user-agent'] || '', Date.now()]);
  const token = generateToken(user);
  res.cookie('token', token, { httpOnly: true });
  res.json({ token });
});

// List courses
app.get('/api/courses', async (req, res) => {
  const cursos = await db.all('SELECT id,title,description FROM courses');
  res.json(cursos);
});

// Simulate buy (in production: create purchase via webhook after payment)
app.post('/api/buy', authMiddleware, async (req, res) => {
  const { courseId } = req.body;
  const purchaseId = uuidv4();
  await db.run('INSERT INTO purchases (id, user_id, course_id, purchased_at) VALUES (?,?,?,?)',
    [purchaseId, req.user.id, courseId, Date.now()]);
  res.json({ ok: true, purchaseId });
});

// Provide weekly code to authenticated user
app.get('/api/my-weekly-code', authMiddleware, async (req, res) => {
  const code = generateWeeklyCodeForUser(req.user.id);
  res.json({ code });
});

// Enter code -> returns accessUrl (signed & bound to domain + user)
app.post('/api/enter-code', authMiddleware, async (req, res) => {
  const { courseId, code } = req.body;

  // check domain header (origin or host)
  const host = req.headers.host;
  if (!domainAllowed(host)) return res.status(403).json({ error: 'Domínio não autorizado' });

  // site-hash check
  if (req.headers['x-site-hash'] !== SITE_HASH) {
    return res.status(403).json({ error: 'Versão do site não autorizada' });
  }

  // verify purchase
  const purchased = await db.get('SELECT * FROM purchases WHERE user_id = ? AND course_id = ?', [req.user.id, courseId]);
  if (!purchased) return res.status(403).json({ error: 'Curso não comprado' });

  // verify code corresponds to user
  const expected = generateWeeklyCodeForUser(req.user.id);
  if (code !== expected) return res.status(403).json({ error: 'Código inválido ou expirado' });

  // create short-lived access token containing domain + userId + courseId
  const accessToken = makeAccessJwt({ userId: req.user.id, courseId, domain: host }, 10 * 60); // 10 min
  const accessUrl = `/media-serve/${courseId}?token=${accessToken}`;
  res.json({ accessUrl });
});

// Serve media - validates short JWT and domain and purchase
app.get('/media-serve/:courseId', async (req, res) => {
  const token = req.query.token;
  if (!token) return res.status(401).send('token ausente');
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    const { userId, courseId, domain } = payload;
    // domain must match host
    if (domain !== req.headers.host) return res.status(401).send('dominio invalido');
    if (courseId !== req.params.courseId) return res.status(403).send('curso mismatch');
    // checar compra
    const purchased = await db.get('SELECT * FROM purchases WHERE user_id = ? AND course_id = ?', [userId, courseId]);
    if (!purchased) return res.status(403).send('curso nao comprado');

    // server-side file
    const course = await db.get('SELECT * FROM courses WHERE id = ?', courseId);
    if (!course) return res.status(404).send('curso nao encontrado');
    const filePath = path.resolve(course.file_path);
    if (!fs.existsSync(filePath)) return res.status(404).send('arquivo nao encontrado');

    // Headers para dificultar download direto
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', 'inline; filename="stream.mp4"');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Pragma', 'no-cache');

    const stat = fs.statSync(filePath);
    const fileSize = stat.size;
    const range = req.headers.range;
    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = (end - start) + 1;
      const stream = fs.createReadStream(filePath, { start, end });
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': 'video/mp4'
      });
      stream.pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Length': fileSize,
        'Accept-Ranges': 'bytes'
      });
      fs.createReadStream(filePath).pipe(res);
    }

  } catch (err) {
    return res.status(401).send('token invalido ou expirado');
  }
});

// optional: revoke session (logout)
app.post('/api/logout', authMiddleware, async (req, res) => {
  // client should send deviceId to remove specific session
  const { deviceId } = req.body;
  const device_hash = deviceHashFromClient({ deviceId }, req.headers['user-agent'] || '');
  await db.run('DELETE FROM sessions WHERE user_id = ? AND device_hash = ?', [req.user.id, device_hash]);
  res.json({ ok: true });
});

// root UI (static)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'index.html'));
});

app.listen(PORT, () => console.log(`Servidor rodando em http://localhost:${PORT}`));