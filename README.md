# NLP-Sentiment Analysis 

## 📝Overview
This project's goal is to build an Natural Language Processing model to analyze tweets' sentiments about Apple and Google products from Twitter, now X, and rate them as either positive, negatve or neutral.
The data comes from CrowdFlower via [data.world](https://data.world/crowdflower/brands-and-product-emotions) and contains over 9000 tweets.

## 💡Business Understanding
In the recent years, the global smartphone market remained highly competitive, with major brands such as Apple and Google continuing to dominate innovation and customer engagement discussions online. However, third-party tech distributors, who rely on these brands' public image to drive sales, often lack accessible tools to automatically analyze global customer sentiment. This limitation makes it difficult for them to gauge market trends, anticipate product reception, or adjust inventory and marketing strategies in real time.
By performing sentiment analysis on tweets related to Apple and Google products, this project aims to provide real-time insights into how consumers perceive these brands, enabling distributors to make data-driven decisions that align with evolving market sentiments.

## 🔍Data Understanding
**Source of data:** The dataset is sourced from CrowdFlower Brands and Product Emotions via [data.world](https://data.world/crowdflower/brands-and-product-emotions) and contains over 9000 tweets.
**Structure of the dataset:**
1. Total Records: 9,093 tweets
2. Columns: 3
  - `tweet_text` : The tweet.
  - `emotion_in_tweet_is_directed_at` : The brand or product that the tweet refers to.
  - `is_there_an_emotion_directed_at_a_brand_or_product` : The sentiment assigned to the tweet. Either **Positive, Negative or No emotion**
3. Target Variable: is_there_an_emotion_directed_at_a_brand_or_product
4. Missing Values: Some Tweets do not specify a brand or product under emotion_in_tweet_is directed_at
5. Data Type: All features are stored as object type.

## 🏆Conclusions & Recommendations

## 💻How to run the project
To run this project, you'll need a Python environment with the required libraries.
### 1. clone the repository
```bash
git clone <repository_url>
cd <repository_name>
```
### 2\. set up the environment
It is recommended to use `conda` or `pip` to manage your environment.

**Using `pip`:**

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
scikit-learn
matplotlib
seaborn
numpy
nltk
wordcloud
streamlit
```
### 3\. Run the Notebook

Launch Jupyter Notebook or JupyterLab from the project directory.

```bash
jupyter notebook
```
Open the main notebook file and run the cells in order. The notebook contains all the code and analysis from the project.
