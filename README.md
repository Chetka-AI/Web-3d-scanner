# Scan3D — Mobilne skanowanie 3D

Aplikacja webowa (PWA) do skanowania obiektów z kamery smartfona i generowania modeli 3D.

## Funkcje

### Tryb AI (domyślny)
- **Depth Anything V2** — sieć neuronowa do estymacji głębi z pojedynczych zdjęć
- **Transformers.js** — inference AI bezpośrednio w przeglądarce (WASM/WebGPU)
- **Gęsta chmura punktów** — tysiące kolorowych punktów 3D z map głębi
- **Automatyczne oddzielanie tła** — analiza histogramu głębi

### Tryb klasyczny (fallback)
- **Detekcja cech Harris** — narożniki i deskryptory patch
- **Dopasowywanie NCC** — sparowane cechy między widokami
- **Triangulacja geometryczna** — punkty 3D z par dopasowanych cech

### Skanowanie na żywo (nowy moduł)
- **Real-time scanning** — ciągłe przechwytywanie klatek z kamery
- **Pose fusion** — IMU + tracking obrazu stabilizują pozycję kamery
- **Keyframe integration** — tylko dobre klatki trafiają do modelu
- **Confidence voxels** — voxel dostaje wsparcie / karę zamiast binarnego on/off
- **World-space map** — punkty są akumulowane w stałym układzie odniesienia
- **Podgląd 3D PiP** — miniatura mapy fusion nakładana na obraz kamery
- **HUD** — FPS, dopasowania, keyframe'y, confidence mapy i status nagrywania

### Wspólne
- **Podgląd kamery** — WebRTC, przełączanie przód/tył, orientacja urządzenia
- **Galeria** — miniaturki z kątami, podgląd, usuwanie
- **Wizualizacja 3D** — Three.js, OrbitControls, auto-rotacja, tryb chmura/siatka
- **Eksport** — OBJ i PLY z kolorami wierzchołków
- **Wczytywanie plików** — upload zdjęć bez kamery
- **PWA** — offline, instalacja na ekranie głównym

## Technologie

- HTML5 / CSS3 / JavaScript (vanilla, zero bundlerów)
- **Transformers.js v3** — AI w przeglądarce (Hugging Face)
- **Depth Anything V2 Small** (int8, ~27MB) — monokularna estymacja głębi
- WebRTC (Camera API)
- Device Orientation API
- Three.js r128 (wizualizacja 3D)
- Service Worker (offline PWA)

## Uruchomienie

Wystarczy serwer HTTP, np.:

```bash
npx serve .
# lub
python3 -m http.server 8000
```

Otwórz na telefonie: `http://<IP>:8000`

> Kamera wymaga HTTPS lub `localhost`. Na serwerze deweloperskim użyj tunelu (np. ngrok) lub flagi Chrome `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

## Jak używać

1. Otwórz aplikację na smartfonie
2. Upewnij się, że **Tryb AI** jest włączony (domyślnie tak)
3. W zakładce **Skanuj** uruchom kamerę (lub wczytaj zdjęcia z plików)
4. Wykonaj 12–24 zdjęć obiektu z różnych kątów (obchodząc go dookoła)
5. Przejdź do **Galeria** i kliknij **Przetwórz (model 3D)**
6. Przy pierwszym użyciu AI model (~27 MB) zostanie pobrany i zapisany w cache
7. Poczekaj na rekonstrukcję w zakładce **Buduj**
8. Obejrzyj model w zakładce **Model 3D** (obracaj, przybliżaj, przełączaj tryby)
9. Eksportuj do OBJ lub PLY

## Struktura projektu

```
├── index.html          # Główny plik HTML
├── manifest.json       # PWA manifest
├── sw.js               # Service Worker
├── css/
│   └── style.css       # Style (mobile-first, dark theme)
├── js/
│   ├── ai-engine.js         # Moduł AI (Transformers.js, Depth Anything V2)
│   ├── app.js               # Kontroler aplikacji, nawigacja
│   ├── camera.js            # Moduł kamery (WebRTC)
│   ├── processing.js        # Silnik rekonstrukcji 3D (AI + klasyczny)
│   ├── realtime-scanner-v2.js  # Stabilizowane skanowanie w czasie rzeczywistym
│   └── viewer.js            # Podgląd 3D (Three.js)
└── icons/
    ├── favicon.svg
    ├── icon-192.png
    └── icon-512.png
```

## Licencja

MIT
