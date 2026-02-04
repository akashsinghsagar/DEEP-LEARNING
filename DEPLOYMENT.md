# Deployment Guide for Gradient Descent Visualizer

## Files Required for Deployment ✅
- `gradient_descent_app.py` - Main application file
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `.streamlit/config.toml` - Streamlit configuration

## Deploy to Streamlit Cloud (Recommended)

### Step 1: Prepare Your Repository
1. Create a GitHub repository (if you haven't already)
2. Push these files to your repository:
   - `gradient_descent_app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to: `gradient_descent_app.py`
6. Click "Deploy"

Your app will be live in a few minutes at: `https://[your-app-name].streamlit.app`

## Alternative: Deploy to Heroku

### Requirements
- Heroku account
- Heroku CLI installed

### Files Needed (Create these)

**Procfile**:
```
web: sh setup.sh && streamlit run gradient_descent_app.py
```

**setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Deployment Steps
```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-gradient-descent-app

# Deploy
git push heroku main

# Open your app
heroku open
```

## Local Testing

Before deploying, test your app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run gradient_descent_app.py
```

The app will open at `http://localhost:8501`

## Environment Variables

No environment variables are needed for this application.

## Troubleshooting

### Common Issues:

1. **Module not found error**: Make sure all dependencies are listed in `requirements.txt`
2. **Port binding error**: Streamlit Cloud handles ports automatically
3. **Memory issues**: The app is lightweight and shouldn't have memory issues

### Support
- Streamlit Community: [discuss.streamlit.io](https://discuss.streamlit.io)
- Documentation: [docs.streamlit.io](https://docs.streamlit.io)

## App Features
- Interactive gradient descent visualization
- Multiple function types (Quadratic, Cubic, Sin Wave)
- Linear regression with gradient descent
- Real-time parameter adjustment
- Iteration history tracking

## Performance Notes
- The app is optimized for quick rendering
- Plotly provides interactive charts
- No database or external API calls required
