# Deploy to Google Cloud Run - Simple Instructions

## Step 1: Open Terminal

Open your Terminal app.

## Step 2: Navigate to Project

```bash
cd /Users/chuck/Documents/antigravity_workspace
```

## Step 3: Deploy

```bash
gcloud builds submit --region=us-central1 --config=cloudbuild.yaml .
```

## Step 4: Wait

This will take 30-40 minutes. You'll see:
1. "Creating temporary archive..." (uploading)
2. "Building..." (building container)
3. "Deploying..." (deploying to Cloud Run)

## Step 5: Get Your URL

When done, you'll see:
```
Service URL: https://austin-rag-api-XXXXXXX.us-central1.run.app
```

Copy that URL and use it in your GitHub Pages site.

---

## If It Fails

Paste the error message and we'll troubleshoot.

## What This Does

- Uploads your code to Google Cloud
- Builds Docker container with Python 3.10
- Deploys to Cloud Run
- Sets up API with your OpenAI key
- Gives you a public URL

That's it!
