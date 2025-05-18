import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import nltk

#Download VADER lexicon
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

@st.cache_data
def load_data():
    df = pd.read_csv("pre_df_export.csv")  # Adjust the path if needed
    return df

#Load models
@st.cache_resource
def load_models():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    loaded_rf = joblib.load('best_rf_pipeline.joblib')
    loaded_lr = joblib.load('best_lr_pipeline.joblib')
    return tfidf_vectorizer, loaded_rf, loaded_lr

#Initialize resources
pre_df = load_data()
tfidf_vectorizer, loaded_rf, loaded_lr = load_models()
analyzer = SentimentIntensityAnalyzer()

#To Check Sentyment
def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def sentiment_summary(keyword, df):
    df_filtered = df[df['ready_text'].str.lower().str.contains(keyword.lower(), na=False)].copy()
    
    if df_filtered.empty:
        st.warning(f"Tidak ada tweet yang mengandung kata '{keyword}'")
        return

    #Predict sentiment using loaded models
    df_filtered['lr_sentiment'] = loaded_lr.predict(df_filtered['ready_text'])
    df_filtered['rf_sentiment'] = loaded_rf.predict(df_filtered['ready_text'])

    #VADER sentiment scores input
    df_filtered['compound'] = df_filtered['ready_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df_filtered['vader_sentiment'] = df_filtered['compound'].apply(classify_sentiment)

    #Sentiment labels for plotting
    sentiment_labels = ['positive', 'neutral', 'negative']

    # Prepare summary dataframe
    summary = pd.DataFrame({
        'VADER': df_filtered['vader_sentiment'].value_counts(),
        'LogisticRegression': df_filtered['lr_sentiment'].value_counts(),
        'RandomForest': df_filtered['rf_sentiment'].value_counts()
    }).reindex(sentiment_labels).fillna(0).astype(int)

    #Display summary
    st.subheader(f"Sentiment untuk kata: '{keyword}'")
    st.dataframe(summary)

    #Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    summary.T.plot(kind='bar', ax=ax)
    ax.set_title(f"Kalkulasi Sentiment untuk kata '{keyword}' dengan menggunakan Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Angka Tweet")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    #Pie charts
    st.subheader("Distribusi Sentimen oleh Model di kancah Pie Charts")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, model in enumerate(['VADER', 'LogisticRegression', 'RandomForest']):
        counts = summary[model]
        axs[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
                   colors=['#99ff99', '#ffcc99', '#66b3ff'])
        axs[i].set_title(f"{model} Sentiment Distribution")
    st.pyplot(fig)

    #Stacked bar chart
    st.subheader("Bar Chart: Komparasi Sentimen dengan Model")
    stacked_df = summary.T
    stacked_df.plot(kind='bar', stacked=True, figsize=(10, 6),
                    color=['#99ff99', '#ffcc99', '#66b3ff'])
    plt.title(f"Kalkulasi Sentimen untuk kata '{keyword}' dengan menggunakan Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=0)
    st.pyplot(plt.gcf())

    #Agreement-based accuracy with VADER
    st.subheader("Akurasi Model (Dengan VADER)")
    lr_matches = df_filtered['lr_sentiment'] == df_filtered['vader_sentiment']
    rf_matches = df_filtered['rf_sentiment'] == df_filtered['vader_sentiment']
    lr_accuracy = lr_matches.mean() * 100
    rf_accuracy = rf_matches.mean() * 100

    acc_fig, acc_ax = plt.subplots(figsize=(6, 4))
    acc_ax.bar(['Logistic Regression', 'Random Forest'], [lr_accuracy, rf_accuracy],
               color=['#FFB347', '#87CEEB'])
    acc_ax.set_ylabel('Accuracy (%)')
    acc_ax.set_ylim(0, 100)
    acc_ax.set_title('Machine Learning Model Agreement with VADER')
    for i, acc in enumerate([lr_accuracy, rf_accuracy]):
        acc_ax.text(i, acc + 1, f"{acc:.2f}%", ha='center')
    st.pyplot(acc_fig)

    #Sample tweets for each method
    st.subheader("Sample Tweets dengan VADER Sentiment dan Machine Learning Model")
    models_and_cols = {
        'VADER': 'vader_sentiment',
        'Logistic Regression': 'lr_sentiment',
        'Random Forest': 'rf_sentiment'
    }

    for model_name, col in models_and_cols.items():
        st.markdown(f"## {model_name}")
        for sentiment in ['positive', 'neutral', 'negative']:
            st.markdown(f"### Sample {sentiment} tweets:")
            samples = df_filtered[df_filtered[col] == sentiment].head(3)
            if samples.empty:
                st.write(f"No {sentiment} tweets found for {model_name}.")
            else:
                for _, row in samples.iterrows():
                    st.markdown(f"- {row['full_text']}")


st.title("Sentimen Tweet Indonesia: Menggali Opini Pendidikan!")
st.write("""
Selamat datang di dashboard interaktif kami, tempatnya buat jelajahi opini masyarakat seputar dunia pendidikan di Indonesia di Platform **Twitter/X**. 
Di sini, kamu bisa nyelamin sentimen tweet dengan bantuan tiga model : **VADER**, **Logistic Regression**, dan **Random Forest**.""")
keyword = st.text_input("Masukkan kata kunci : ")

if st.button("Analisa"):
    if keyword.strip():
        sentiment_summary(keyword, pre_df)
    else:
        st.warning("Tolong Masukkan Kata Kunci")
