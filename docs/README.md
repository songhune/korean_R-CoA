# KLSBench Website

This directory contains the GitHub Pages website for KLSBench.

## Structure

```
docs/
├── index.html          # Main page
├── css/
│   └── style.css      # Styles
├── js/
│   └── main.js        # Interactive components
└── samples/           # Sample data (generated)
    ├── sample_classification.json
    ├── sample_retrieval.json
    ├── sample_nli.json
    ├── sample_translation.json
    ├── sample_punctuation.json
    └── summary.json
```

## Local Development

To test the website locally:

1. Generate sample data:
   ```bash
   python3 create_samples.py
   ```

2. Start a local server:
   ```bash
   cd docs
   python3 -m http.server 8000
   ```

3. Open http://localhost:8000 in your browser

## Deployment

This site is automatically deployed via GitHub Pages from the `docs/` directory.

To enable GitHub Pages:

1. Go to repository Settings
2. Navigate to Pages section
3. Under "Source", select "Deploy from a branch"
4. Select branch: `main` (or your default branch)
5. Select folder: `/docs`
6. Click Save

The site will be available at: `https://klsbench.github.io/`

## Updating Sample Data

To update the sample data displayed on the website:

```bash
# From the project root
python3 create_samples.py
```

This will regenerate all sample JSON files in `docs/samples/`.
