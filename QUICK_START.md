# 🚀 Quick Start Guide - Sentiment Analysis App

## One-Minute Setup

### 1️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```bash
streamlit run app.py
```

### 3️⃣ Open in Browser
Navigate to: **http://localhost:8501**

---

## 5-Minute First Use

### For CSV Files:
1. Create a CSV with a column named `text`
2. Upload the CSV file
3. View results in the table
4. Download results as CSV

### For PDF Files:
1. Upload any PDF document
2. Text is extracted and split into sentences
3. Each sentence is analyzed
4. Download results as CSV

---

## Example CSV Format

Create a file `reviews.csv`:
```csv
text
"This product is amazing! I love it."
"Terrible quality, very disappointed."
"Best purchase I've made, highly recommend!"
"Waste of money, would not recommend."
```

---

## Key Features at a Glance

| Feature | Description |
|---------|-------------|
| 📁 File Upload | CSV with "text" column or any PDF |
| 🔍 Text Processing | Auto cleans, removes stopwords, normalizes |
| 🧠 LSTM Model | Pre-trained binary sentiment classifier |
| 📊 Visualization | Sentiment distribution chart |
| 📥 Download | Export results as CSV |
| ⚙️ Filters | Confidence threshold adjustment |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fitz'"
**Solution:**
```bash
pip install PyMuPDF
```

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:**
```bash
pip install tensorflow keras
```

### Issue: CSV has no "text" column
**Solution:** Ensure your CSV has exactly a column named `text` (lowercase, exact spelling)

### Issue: Model not loading
**Solution:**
```bash
# Delete cached model to retrain
rm sentiment_model.h5
rm tokenizer.pkl

# Run app again - it will retrain on dummy data
streamlit run app.py
```

---

## What Happens on First Run?

1. **NLTK Data Download** (~30 seconds)
   - Downloads tokenizer and stopwords

2. **Model Training** (~20 seconds)
   - Trains LSTM on 10 dummy sentences
   - Saves to `sentiment_model.h5`

3. **Tokenizer Creation** (~5 seconds)
   - Creates Keras tokenizer
   - Saves to `tokenizer.pkl`

4. **Ready to Use** 🎉
   - On next runs, loads cached model (instant startup)

---

## Model Prediction Logic

- **Score > 0.5** → "Positive" ✅
- **Score ≤ 0.5** → "Negative" ❌

Each prediction includes a confidence percentage.

---

## Application Flow

```
Upload File
    ↓
Extract Content (CSV/PDF)
    ↓
Preprocess Text
    ↓
Convert to Sequences
    ↓
LSTM Prediction
    ↓
Display Results
    ↓
Download CSV
```

---

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ (for TensorFlow)
- **Disk**: 1GB+ (for dependencies)
- **Internet**: For first-run NLTK downloads

---

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `sentiment_model.h5` | Trained LSTM model (auto-generated) |
| `tokenizer.pkl` | Keras tokenizer (auto-generated) |
| `sentiment_results.csv` | Downloaded analysis results |

---

## Need Help?

1. **Review** the [full documentation](./SENTIMENT_ANALYSIS_README.md)
2. **Check** error messages in the app
3. **Verify** your input files follow the format
4. **Reinstall** packages if issues persist

---

**Start analyzing sentiments now! 🎉**
