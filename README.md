# nlp_text-processing_mod3
flexisaf gen AI and ds mod 3 task + streamlit interface

# Financial Text Preprocessing Pipeline

NLP preprocessing system for financial text data with interactive web interface.

## Features

Implements three core text preprocessing techniques:

- **URL Removal**: Strips hyperlinks from social media posts and web content
- **Punctuation Removal**: Cleans text by removing all punctuation marks
- **Lemmatization**: Reduces words to base dictionary form using POS tagging

## Tech Stack

- Python 3.11
- Streamlit for web interface
- NLTK for natural language processing
- Pandas for batch data handling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Standalone Script

```bash
python text_processing_pipeline.py
```

Processes sample financial texts and outputs results to `preprocessed_financial_text.csv`.

### Web Interface

```bash
streamlit run streamlit_app.py
```

Access the interactive interface at `http://localhost:8501`

## Features

**Interactive Processing:**
- Real-time text preprocessing with toggleable techniques
- Sample financial texts for quick demonstration
- Side-by-side comparison of original vs processed text

**Batch Processing:**
- Upload CSV files for bulk preprocessing
- Select specific text columns to process
- Download processed results

**Statistics:**
- Character count tracking
- Text reduction percentage
- Processing technique summary

## Project Structure

```
project/
├── streamlit_app.py     # Interactive web interface
├── text_processing_pipeline.py              # Standalone preprocessing script
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version specification
└── README.md
```

## Use Cases

- Clean financial news headlines for sentiment analysis models
- Preprocess social media data for trading signal generation
- Prepare text from 10-K filings for risk classification
- Batch process financial documents for NLP pipelines

## Output

**Standalone Script:**
- `preprocessed_financial_text.csv`: Processed text with statistics

**Web Interface:**
- Interactive results display
- Downloadable CSV for batch processing

## Requirements

- Python 3.11+
- Streamlit 1.29.0+
- NLTK 3.8.1+
- Pandas 2.0.0+

## Deployment

Deployed on Streamlit Community Cloud. Push to GitHub and deploy through https://streamlit.io/cloud

**Required files for deployment:**
- `requirements.txt`
- `runtime.txt` (specifies Python 3.11)
- `streamlit_app.py`

## Technical Details

**Preprocessing Pipeline:**
- Regex-based URL extraction and removal
- Translation table for punctuation stripping
- POS-tagged lemmatization for accuracy

**Performance:**
- Cached preprocessor for efficiency
- Automatic NLTK data downloads on first run
- Handles large batch files through pandas
