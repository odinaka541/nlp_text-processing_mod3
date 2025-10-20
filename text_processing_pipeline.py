# d541

# imports
import re, string, pandas as pd, nltk, warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

warnings.filterwarnings('ignore')

# download required nltk data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


def create_sample_financial_data():
    """
    creates sample financial text data for preprocessing
    mix of news headlines and social media style posts
    """
    financial_texts = [
        "Apple Inc. Reports Record Q4 Earnings, Stock Surges 5%! https://cnbc.com/apple-earnings",
        "BREAKING: Federal Reserve raises interest rates by 0.25% - Markets react negatively!!!",
        "Tesla's New Factory in Texas is OPERATIONAL... Production ramping up! Check details: https://reuters.com/tesla-texas",
        "Goldman Sachs predicts recession in 2024??? Read more at https://bloomberg.com/gs-prediction",
        "Amazon announces massive LAYOFFS affecting 10,000 employees. #TechLayoffs https://techcrunch.com/amzn",
        "Bitcoin crashes below $20K! Crypto winter continues... https://coindesk.com/btc-crash",
        "Microsoft's AI investments paying off - Azure revenue UP 30%",
        "JPMorgan CEO warns of economic headwinds!!! Full interview: https://cnbc.com/jpmorgan-ceo",
        "Meta platforms launches new VR headset, priced at $1,500. Details: https://theverge.com/meta-vr",
        "Oil prices surge to $95/barrel amid supply concerns... https://reuters.com/oil-prices",
        "Netflix subscriber growth EXCEEDS expectations!!! Stock jumps 12% https://variety.com/netflix",
        "Deutsche Bank faces regulatory scrutiny over trading practices https://ft.com/deutsche-bank",
        "Nvidia's GPU shortage continues - Gaming industry impacted!!! https://pcgamer.com/nvidia",
        "Warren Buffett increases stake in Occidental Petroleum https://wsj.com/buffett-occidental",
        "Chinese markets tumble as property sector concerns MOUNT... https://scmp.com/china-markets",
        "Ford announces $3.5B battery plant investment. Read: https://autoblog.com/ford-battery",
        "Inflation data comes in HOTTER than expected! Fed's next move??? https://bloomberg.com/inflation",
        "PayPal explores cryptocurrency integration for payments https://coindesk.com/paypal-crypto",
        "Boeing deliveries increase in Q3, recovery underway... https://reuters.com/boeing-deliveries",
        "Zoom video communications reports declining revenue - Work from home era ending???"
    ]

    # create dataframe with sample data
    df = pd.DataFrame({
        'text': financial_texts,
        'source': ['news' if i % 2 == 0 else 'social_media' for i in range(len(financial_texts))]
    })

    print(f"created sample dataset with {len(df)} financial texts")
    return df


class TextPreprocessor:
    """
    text preprocessing pipeline for financial text data
    implements lemmatization, url removal, and punctuation removal
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = r'https?://\S+|www\.\S+'  # pattern to match urls

    def get_wordnet_pos(self, word):
        """
        map pos tag to wordnet pos tag
        helps with better lematization results
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    # technique 2
    def remove_urls(self, text):
        """
        removes urls from text using regex pattern
        important for cleaning social media financial posts
        """
        cleaned = re.sub(self.url_pattern, '', text)
        # clean up extra spaces left after url removal
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    # technique 3
    def remove_punctuation(self, text):
        """
        removes all punctuation marks from text
        keeps text clean for further analysis
        """
        # create translation table for punctuation removal
        translator = str.maketrans('', '', string.punctuation)
        cleaned = text.translate(translator)
        return cleaned

    # technique 1
    def lemmatize_text(self, text):
        """
        applies lemmatization to reduce words to base form
        uses pos tagging for better accuracy
        """
        # tokenize the text first
        tokens = nltk.word_tokenize(text)

        # lemmatize each token with proper pos tag
        lemmatized_tokens = []
        for token in tokens:
            pos = self.get_wordnet_pos(token)
            lemmatized = self.lemmatizer.lemmatize(token.lower(), pos)
            lemmatized_tokens.append(lemmatized)

        # join tokens back into string
        return ' '.join(lemmatized_tokens)

    def preprocess_pipeline(self, text, remove_url=True, remove_punct=True, lemmatize=True):
        """
        full preprocessing pipeline applying all techniques
        can toggle individual steps as needed
        """
        processed = text

        if remove_url:
            processed = self.remove_urls(processed)

        if remove_punct:
            processed = self.remove_punctuation(processed)

        if lemmatize:
            processed = self.lemmatize_text(processed)

        return processed


def demonstrate_preprocessing(df, preprocessor):
    """
    demonstrates each preprocessing technique with before/after examples
    shows the impact of each step
    """
    print("\n" + "=" * 80)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 80)

    # technique 1: url removal
    print("\n[1] URL REMOVAL")
    print("-" * 80)
    sample_with_url = df[df['text'].str.contains('https')].iloc[0]['text']
    print(f"BEFORE: {sample_with_url}")
    print(f"AFTER:  {preprocessor.remove_urls(sample_with_url)}")

    # technique 2: punctuation removal
    print("\n[2] PUNCTUATION REMOVAL")
    print("-" * 80)
    sample_with_punct = df[df['text'].str.contains('!!!|\\?\\?\\?')].iloc[0]['text']
    print(f"BEFORE: {sample_with_punct}")
    # first remove urls then punctuation for clean demo
    temp = preprocessor.remove_urls(sample_with_punct)
    print(f"AFTER:  {preprocessor.remove_punctuation(temp)}")

    # technique 3: lemmatization
    print("\n[3] LEMMATIZATION")
    print("-" * 80)
    sample_text = "Tesla's factories are producing more vehicles than expected"
    print(f"BEFORE: {sample_text}")
    print(f"AFTER:  {preprocessor.lemmatize_text(sample_text)}")

    print("\n" + "=" * 80)


def process_full_dataset(df, preprocessor):
    """
    applies full preprocessing pipeline to entire dataset
    creates comparison of original vs processed text
    """
    print("\nProcessing full dataset...")

    # apply preprocessing to all texts
    df['processed_text'] = df['text'].apply(
        lambda x: preprocessor.preprocess_pipeline(x,
                                                   remove_url=True,
                                                   remove_punct=True,
                                                   lemmatize=True)
    )

    # calculate some basic statistics
    df['original_length'] = df['text'].apply(len)
    df['processed_length'] = df['processed_text'].apply(len)
    df['length_reduction'] = ((df['original_length'] - df['processed_length']) /
                              df['original_length'] * 100)

    print(f"\nDataset statistics:")
    print(f"  Average original length: {df['original_length'].mean():.1f} characters")
    print(f"  Average processed length: {df['processed_length'].mean():.1f} characters")
    print(f"  Average length reduction: {df['length_reduction'].mean():.1f}%")

    return df


def save_results(df):
    """saves preprocessing results to csv file"""
    output_file = 'preprocessed_financial_text.csv'

    # select relevant columns for output
    output_df = df[['text', 'processed_text', 'source', 'length_reduction']]
    output_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

    # show a few examples from the output
    print("\nSample results:")
    print("-" * 80)
    for idx in range(min(3, len(output_df))):
        print(f"\nExample {idx + 1}:")
        print(f"Original:  {output_df.iloc[idx]['text'][:100]}...")
        print(f"Processed: {output_df.iloc[idx]['processed_text'][:100]}...")
        print(f"Reduction: {output_df.iloc[idx]['length_reduction']:.1f}%")


def main():
    """main execution function"""

    print("Starting text preprocessing pipeline for financial data")
    print("Techniques: URL Removal, Punctuation Removal, Lemmatization\n")

    # create sample financial data
    df = create_sample_financial_data()

    # initialize preprocessor
    preprocessor = TextPreprocessor()

    # demonstrate each technique individually
    demonstrate_preprocessing(df, preprocessor)

    # process full dataset with all techniques
    processed_df = process_full_dataset(df, preprocessor)

    # save results
    save_results(processed_df)

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()