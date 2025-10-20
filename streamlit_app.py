#d541

"""
main text processing pipeline + streamlit app for interactiveness
"""

#imports
import streamlit as st, re, string, nltk, pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd


# download required nltk data quietly
@st.cache_resource
def download_nltk_data():
    """downloads required nltk data packages"""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)


download_nltk_data()


class TextPreprocessor:
    """text preprocessing pipeline"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = r'https?://\S+|www\.\S+'

    def get_wordnet_pos(self, word):
        """map pos tag to wordnet pos tag for better lemmatization"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def remove_urls(self, text):
        """removes urls from text"""
        cleaned = re.sub(self.url_pattern, '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def remove_punctuation(self, text):
        """removes punctuation from text"""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def lemmatize_text(self, text):
        """applies lemmatization to text"""
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = []
        for token in tokens:
            pos = self.get_wordnet_pos(token)
            lemmatized = self.lemmatizer.lemmatize(token.lower(), pos)
            lemmatized_tokens.append(lemmatized)
        return ' '.join(lemmatized_tokens)

    def preprocess_pipeline(self, text, remove_url=True, remove_punct=True, lemmatize=True):
        """full preprocessing pipeline"""
        processed = text
        steps_applied = []

        if remove_url:
            processed = self.remove_urls(processed)
            steps_applied.append("URL Removal")

        if remove_punct:
            processed = self.remove_punctuation(processed)
            steps_applied.append("Punctuation Removal")

        if lemmatize:
            processed = self.lemmatize_text(processed)
            steps_applied.append("Lemmatization")

        return processed, steps_applied


# initialize preprocessor
@st.cache_resource
def get_preprocessor():
    return TextPreprocessor()


preprocessor = get_preprocessor()

# sample financial texts for demo
sample_texts = {
    "Tech Earnings": "Apple Inc. Reports Record Q4 Earnings, Stock Surges 5%! https://cnbc.com/apple-earnings",
    "Fed Policy": "BREAKING: Federal Reserve raises interest rates by 0.25% - Markets react negatively!!!",
    "Tesla News": "Tesla's New Factory in Texas is OPERATIONAL... Production ramping up! Check details: https://reuters.com/tesla-texas",
    "Recession Warning": "Goldman Sachs predicts recession in 2024??? Read more at https://bloomberg.com/gs-prediction",
    "Crypto Crash": "Bitcoin crashes below $20K! Crypto winter continues... https://coindesk.com/btc-crash"
}

# streamlit app layout
st.set_page_config(page_title="Financial Text Preprocessing", layout="wide")

st.title("Financial Text Preprocessing Tool")
st.markdown("preprocessing pipeline for financial news and social media text")

# sidebar for preprocessing options
st.sidebar.header("preprocessing options")
remove_url = st.sidebar.checkbox("Remove URLs", value=True)
remove_punct = st.sidebar.checkbox("Remove Punctuation", value=True)
lemmatize = st.sidebar.checkbox("Apply Lemmatization", value=True)

# main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("input text")

    # option to use sample or custom text
    use_sample = st.checkbox("use sample financial text")

    if use_sample:
        selected_sample = st.selectbox("select sample", list(sample_texts.keys()))
        input_text = sample_texts[selected_sample]
        st.text_area("sample text", input_text, height=150, disabled=True)
    else:
        input_text = st.text_area(
            "enter your financial text here",
            "Paste financial news headlines, social media posts, or any text you want to preprocess...",
            height=150
        )

with col2:
    st.subheader("processed output")

    if input_text and input_text.strip():
        # process the text
        processed_text, steps = preprocessor.preprocess_pipeline(
            input_text,
            remove_url=remove_url,
            remove_punct=remove_punct,
            lemmatize=lemmatize
        )

        # show processed text
        st.text_area("processed text", processed_text, height=150, disabled=True)

        # show statistics
        st.markdown("**statistics:**")
        col_stat1, col_stat2, col_stat3 = st.columns(3)

        with col_stat1:
            st.metric("original length", f"{len(input_text)} chars")

        with col_stat2:
            st.metric("processed length", f"{len(processed_text)} chars")

        with col_stat3:
            reduction = ((len(input_text) - len(processed_text)) / len(input_text) * 100)
            st.metric("reduction", f"{reduction:.1f}%")

        # show applied steps
        if steps:
            st.markdown(f"**applied techniques:** {', '.join(steps)}")
    else:
        st.info("enter text in the input area to see preprocessing results")

# technique explanations
st.markdown("---")
st.subheader("preprocessing techniques explained")

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    with st.expander("URL Removal"):
        st.write("""
        removes hyperlinks from text using regex patterns.
        important for cleaning social media posts and web-scraped content.

        **example:**
        - before: check this https://example.com/article
        - after: check this
        """)

with exp_col2:
    with st.expander("Punctuation Removal"):
        st.write("""
        strips all punctuation marks from text.
        useful for text analysis and machine learning preprocessing.

        **example:**
        - before: markets react negatively!!!
        - after: markets react negatively
        """)

with exp_col3:
    with st.expander("Lemmatization"):
        st.write("""
        reduces words to their base dictionary form.
        uses part-of-speech tagging for accurate results.

        **example:**
        - before: factories are producing vehicles
        - after: factory be produce vehicle
        """)

# batch processing section
st.markdown("---")
st.subheader("batch processing")

uploaded_file = st.file_uploader("upload csv file with text column", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # let user select text column
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        selected_column = st.selectbox("select text column", text_columns)

        if st.button("process batch"):
            with st.spinner("processing..."):
                # apply preprocessing to selected column
                df['processed_text'] = df[selected_column].apply(
                    lambda x: preprocessor.preprocess_pipeline(
                        str(x), remove_url, remove_punct, lemmatize
                    )[0]
                )

                # show results
                st.success(f"processed {len(df)} rows")
                st.dataframe(df[[selected_column, 'processed_text']].head(10))

                # download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "download processed data",
                    csv,
                    "preprocessed_data.csv",
                    "text/csv"
                )
    except Exception as e:
        st.error(f"error processing file: {str(e)}")

# footer
st.markdown("---")
st.markdown("*built for financial text analysis and nlp preprocessing*")