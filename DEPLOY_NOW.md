# Deploy to Google Cloud Run - Complete Instructions

## Prerequisites

Before deploying, you need to:
1. Have `gcloud` CLI installed and authenticated
2. Have your ChromaDB database files ready
3. Have your OpenAI API key

---

## Step 1: Set Up Secret Manager (One-Time Setup)

First, store your OpenAI API key securely in Google Secret Manager:

```bash
# Create the secret
echo -n "YOUR_OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding openai-api-key \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

Replace `YOUR_OPENAI_API_KEY` with your actual key and `PROJECT_NUMBER` with your GCP project number.

---

## Step 2: Upload ChromaDB to Cloud Storage

Since ChromaDB files are too large for git, upload them to Cloud Storage:

```bash
# Create a bucket (one-time)
gsutil mb -l us-central1 gs://austin-planning-rag-data

# Upload your chroma_db folder
gsutil -m cp -r ./chroma_db gs://austin-planning-rag-data/
```

---

## Step 3: Update Dockerfile for GCS (Alternative)

If using Cloud Storage, update your Dockerfile to download the database at startup, OR include the chroma_db folder in your build context.

**Option A: Include in build context**
```bash
# Make sure chroma_db is in the same directory as Dockerfile
ls -la chroma_db/
```

**Option B: Download from GCS at startup** (requires modifying main.py)

---

## Step 4: Deploy

```bash
# Navigate to your project directory
cd /path/to/your/project

# Submit the build
gcloud builds submit --region=us-central1 --config=cloudbuild.yaml .
```

---

## Step 5: Wait

This will take 10-20 minutes. You'll see:
1. "Creating temporary archive..." (uploading)
2. "Building..." (building container)
3. "Deploying..." (deploying to Cloud Run)

---

## Step 6: Get Your URL

When done, you'll see:
```
Service URL: https://austin-rag-api-XXXXXXX.us-central1.run.app
```

Test it:
```bash
curl https://austin-rag-api-XXXXXXX.us-central1.run.app/health
```

---

## Troubleshooting

### "Pool timeout" errors
This has been fixed! The code now uses async HTTP clients with proper connection pooling.

### "ChromaDB not initialized"
Your chroma_db folder wasn't included in the build. Make sure it's in the build context.

### "Secret not found"
Run the Secret Manager setup in Step 1.

### View logs
```bash
gcloud run logs read austin-rag-api --region=us-central1
```

---

## What This Deployment Does

- Builds Docker container with Python 3.11
- Uses async OpenAI client with connection pooling (no more timeouts!)
- Securely loads API key from Secret Manager
- Deploys to Cloud Run with 4GB RAM, 2 CPUs
- Auto-scales from 0 to 10 instances
- Gives you a public HTTPS URL
