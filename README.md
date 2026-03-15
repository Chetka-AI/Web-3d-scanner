# Scan3D — Mobilne skanowanie 3D

Aplikacja webowa (PWA) do skanowania obiektów z kamery smartfona i generowania modeli 3D.

## Funkcje

- **Podgląd kamery** — dostęp do tylnej/przedniej kamery przez WebRTC
- **Przechwytywanie zdjęć** — zapis zdjęć z informacją o orientacji urządzenia
- **Galeria** — podgląd, usuwanie zdjęć
- **Rekonstrukcja 3D** — detekcja cech (Harris), dopasowywanie, triangulacja, generowanie chmury punktów i siatki
- **Podgląd 3D** — interaktywna wizualizacja Three.js (obrót, zoom, pan)
- **Eksport** — OBJ i PLY z kolorami wierzchołków
- **PWA** — działa offline, można zainstalować na ekranie głównym

## Technologie

- HTML5 / CSS3 / JavaScript (vanilla)
- WebRTC (Camera API)
- Device Orientation API
- Three.js (wizualizacja 3D)
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
2. W zakładce **Skanuj** uruchom kamerę
3. Wykonaj 12–24 zdjęć obiektu z różnych kątów (obiektyw obchodzi obiekt dookoła)
4. Przejdź do **Galeria** i kliknij **Przetwórz**
5. Poczekaj na rekonstrukcję w zakładce **Buduj**
6. Obejrzyj model w zakładce **Model 3D**
7. Eksportuj do OBJ lub PLY

## Struktura projektu

```
├── index.html          # Główny plik HTML
├── manifest.json       # PWA manifest
├── sw.js               # Service Worker
├── css/
│   └── style.css       # Style (mobile-first, dark theme)
├── js/
│   ├── app.js          # Kontroler aplikacji, nawigacja
│   ├── camera.js       # Moduł kamery (WebRTC)
│   ├── processing.js   # Silnik rekonstrukcji 3D
│   └── viewer.js       # Podgląd 3D (Three.js)
└── icons/
    ├── favicon.svg
    ├── icon-192.png
    └── icon-512.png
```

## Licencja

MIT
