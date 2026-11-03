# docs/slides

The progress review deck and everything needed to rebuild it.

| File | What it is |
|---|---|
| `Progress_Review_2026-10-23_CacheMemory.pdf` | The finished deck, as presented on 23 Oct (it still describes the earlier ear dataset) |
| `slides.html` | The deck source: one `<section class="slide">` per slide |
| `build.sh` | Turns `slides.html` into the PDF using headless Microsoft Edge |
| `bg.jpg`, `logo.png` | Slide background and university logo |
| `Cantarell-Bold.ttf`, `JetBrainsMono.ttf` | Fonts (open licence), bundled so the build works offline |

Rebuild (Git Bash, from the repo root):

```bash
bash docs/slides/build.sh
```

To add a slide, copy an existing `<section class="slide">` block in `slides.html` and change its
title and content. For a new deck (for example final results), copy this folder and edit the copy.
