# Wallpapers animados — Ophanim / Fusão Visual e Conceitual

Cena 3D reproduzível (Three.js + captura headless) composta sobre a arte escura.
Anéis cobertos de olhos giram em eixos mundiais distintos; o olho central move a íris e pisca.
Textos/diagramas permanecem estáticos via máscaras.

## Entregáveis (`final/`)

| Arquivo | Descrição |
|---------|-----------|
| `wallpaper_computador_1920x1080.mp4` | 16:9, 10s, 24fps, H.264, sem áudio |
| `wallpaper_celular_1080x1920.mp4` | 9:16, 10s, 24fps, H.264, sem áudio |
| `preview/computador/frame_{0,25,50,75}pct.png` | Quadros de verificação |
| `preview/celular/frame_{0,25,50,75}pct.png` | Quadros de verificação |
| `src/` | Cena, scripts de captura/composição |

## Requisitos

- Node.js 18+
- Google Chrome / Chromium (WebGL SwiftShader)
- Python 3 + Pillow, NumPy, SciPy
- FFmpeg
- (Opcional) Blender 4.2+ em `../tools/blender` — script alternativo `scripts/render_scene.py`

## Reproduzir

```bash
cd ophanim
npm install
python3 scripts/prepare_assets.py
python3 scripts/inpaint_plates.py

# Captura dos overlays 3D (transparente)
node scripts/capture_frames.js computador
node scripts/capture_frames.js celular

# Compõe na arte + gera MP4
python3 scripts/composite.py both
```

Prévia rápida (4 quadros):

```bash
node scripts/capture_frames.js computador --preview
python3 scripts/composite.py computador --preview
```

## Movimento (loop 10s)

- 4 toros com textura de olhos; rotação por **eixo mundial** (X/Y/Z) com ±1 volta inteira → sem salto no loop.
- Íris: offset periódico senoidal.
- Piscadas em ~30% e ~70% do ciclo (envelopes gaussianos periódicos).
- Máscaras em `masks/` protegem textos; o buraco central troca os anéis estáticos da arte pelo render 3D.

## Notas sobre assets de entrada

Neste ambiente só estavam disponíveis as artes anexadas (horizontal/vertical).
Elas foram redimensionadas para `arte_computador_1920x1080.png` e `arte_celular_1080x1920.png`.
Máscaras foram geradas automaticamente (texto branco + painéis laterais).
O vídeo `referencia_movimento.mp4` não estava na pasta; o movimento 3D segue o comportamento de esfera armilar descrito (eixos distintos, passagem frente/atrás do olho).
