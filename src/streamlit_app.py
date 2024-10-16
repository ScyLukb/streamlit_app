import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Function to load data
@st.cache_data
def load_data():
    # Replace this with your actual data loading logic
    df = pd.DataFrame({
        'input_text': ['hello', 'hi there', 'goodbye', 'see you', 'how are you'],
        'output_text': ['greeting', 'greeting', 'farewell', 'farewell', 'inquiry'],
        'intent': ['greet', 'greet', 'bye', 'bye', 'ask'],
        'probability': [0.9, 0.8, 0.95, 0.85, 0.7],
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D')
    })
    return df

# Function to create probability distribution histogram
def plot_probability_histogram(data):
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='probability', kde=True, ax=ax)
    ax.set_title('Probability Distribution Histogram')
    return fig

# Function to create average probability by intent bar chart
def plot_avg_probability_by_intent(data):
    avg_prob = data.groupby('intent')['probability'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=avg_prob.index, y=avg_prob.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Average Probability by Intent')
    plt.tight_layout()
    return fig

# Function to create word cloud
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig

# Load data
df = load_data()

# Set up the Streamlit app
st.title('Intent Analysis Dashboard')

# Create a 2x2 grid layout
col1, col2 = st.columns(2)

with col1:
    # Top left: Probability Distribution Histogram
    st.subheader('Probability Distribution Histogram')
    st.pyplot(plot_probability_histogram(df))

with col2:
    # Top right: Average Probability by Intent
    st.subheader('Average Probability by Intent')
    st.pyplot(plot_avg_probability_by_intent(df))

col3, col4 = st.columns(2)

with col3:
    # Bottom left: Word Cloud for high probability inputs
    st.subheader('Word Cloud: High Probability Inputs (> 0.5)')
    high_prob_text = ' '.join(df[df['probability'] > 0.5]['input_text'])
    st.pyplot(create_wordcloud(high_prob_text, 'High Probability Inputs'))

with col4:
    # Bottom right: Word Cloud for low probability inputs
    st.subheader('Word Cloud: Low Probability Inputs (<= 0.5)')
    low_prob_text = ' '.join(df[df['probability'] <= 0.5]['input_text'])
    st.pyplot(create_wordcloud(low_prob_text, 'Low Probability Inputs'))
