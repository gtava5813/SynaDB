# SynaDB Website

Simple landing page for [synadb.ai](https://synadb.ai).

## Local Preview

```bash
cd website
python -m http.server 8000
# Open http://localhost:8000
```

## Deployment

This folder is served via GitHub Pages. To enable:

1. Go to repo Settings → Pages
2. Source: Deploy from branch
3. Branch: `main`, folder: `/website`
4. Save

The site will be available at `https://synadb.ai` (with custom domain) or `https://gtava5813.github.io/SynaDB/website/`.

## Structure

```
website/
├── index.html    # Landing page
├── style.css     # Styles
└── README.md     # This file
```

Assets (logo, favicon) are referenced from `../assets/`.
