# 🚀 Deployment Guide - Streamlit Cloud

This guide will help you deploy your Mini Dataset Generator to Streamlit Cloud for free.

## 📋 Prerequisites

1. **GitHub Account** - Sign up at [github.com](https://github.com)
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Code

✅ Already done! Your project includes:
- `.gitignore` - Excludes unnecessary files
- `.streamlit/config.toml` - Streamlit configuration
- `requirements.txt` - All dependencies
- `app.py` - Main application

### Step 2: Push to GitHub

1. **Initialize Git** (if not already done):
```powershell
cd "D:\My Products\mini_dataset_genarator"
git init
```

2. **Add all files**:
```powershell
git add .
```

3. **Commit**:
```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
git commit -m "Initial commit: Mini Dataset Generator"
```

4. **Create GitHub Repository**:
   - Go to [github.com/new](https://github.com/new)
   - Name: `mini-dataset-generator`
   - Description: "Web-based YOLO dataset generator with augmentation"
   - Keep it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (we already have one)
   - Click "Create repository"

5. **Push to GitHub**:
```powershell
git remote add origin https://github.com/YOUR-USERNAME/mini-dataset-generator.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign in with GitHub"
   - Authorize Streamlit

2. **Create New App**:
   - Click "New app" button
   - Repository: `YOUR-USERNAME/mini-dataset-generator`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose custom name (e.g., `mini-dataset-gen`)

3. **Click "Deploy"**:
   - Streamlit will install dependencies
   - Build process takes ~2-3 minutes
   - Your app will be live at: `https://your-custom-name.streamlit.app`

### Step 4: Update README

After deployment, update the badge URL in `README.md`:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-actual-url.streamlit.app)
```

## 🎯 Your App is Live!

Share your app URL with anyone:
- `https://your-custom-name.streamlit.app`

## 🔄 Updating Your App

Every time you push changes to GitHub, Streamlit Cloud automatically redeploys:

```powershell
git add .
git commit -m "Update: description of changes"
git push
```

Wait 1-2 minutes for changes to appear.

## 📊 Monitoring

- View logs: Click "Manage app" → "Logs" in Streamlit Cloud
- Analytics: See usage stats in your dashboard
- Reboot: Click "Reboot app" if needed

## ⚠️ Important Notes

1. **Storage**: Files created in `output/` folder are temporary (cleared on reboot)
2. **Memory**: Free tier has 1GB RAM limit
3. **Usage**: Unlimited users, but rate-limited for heavy traffic
4. **Privacy**: App is public unless you upgrade to paid tier

## 🆘 Troubleshooting

**Build fails?**
- Check `requirements.txt` for correct package names
- View logs in Streamlit Cloud dashboard

**App crashes?**
- Large image uploads may exceed memory
- Check error logs in Streamlit Cloud

**Can't find your repo?**
- Make sure repository is public
- Refresh GitHub connection in Streamlit settings

## 🎉 Next Steps

1. Share your app URL on social media
2. Add custom domain (Teams plan)
3. Monitor usage analytics
4. Collect user feedback
5. Add to your portfolio/resume!

---

**Questions?** Check [Streamlit Documentation](https://docs.streamlit.io/streamlit-community-cloud)
