# Netflix Tweets and Stock Price Prediction Using Sentiment, Price and Volume Signals

## Overview

This project investigates whether social media sentiment, tweet volume, and traditional price/volume signals can predict Netflix (NFLX) stock price movements. By combining Twitter data with historical price data, we engineer features and evaluate classification and regression models to forecast next-day opening and closing price direction.

---

## 1. Data Collection

- **Tweets**: Collected from Twitter API between January 1, 2020 and July 11, 2022, filtered for cashtags like `$NFLX`.  
- **Stock Prices**: Retrieved via `yfinance` and pre-saved as Parquet files containing daily Open, Close, High, Low, and Volume.

## 2. Data Preprocessing

- **Tweet Cleaning**:  
  - Removed URLs, mentions, hashtags, non-ASCII characters.  
  - Replaced cashtags (`$NFLX`) with `[TICKER]` tokens.  
  - Tokenized and lowercased text.  
- **Weekend Adjustment**: Shifted tweets from Saturday/Sunday to the next Monday to align with trading days.  
- **Aggregation**: Computed daily tweet volume (`tweetVol`) and retained cleaned text for feature engineering.

## 3. Feature Engineering

- **Sentiment Scores**:  
  - BERTweet model embeddings.  
  - DistilBERT (SST-2) sentiment scores.  
  - Loughran–McDonald lexicon-based scores.  
- **FAISS Smoothing**: Averaged sentiment scores over 500 nearest neighbors in embedding space to reduce noise.  
- **Daily Aggregates**:  
  - 7-day rolling z-scores of tweet volume.  
  - 1-day and 7-day lag features for both volume and sentiment.  
- **Ensemble & Interaction Terms**:  
  - Mean ensemble of the three sentiment models.  
  - Interaction term: `sentiment × z_vol_7d`.  
  - Earnings window dummy for quarterly release periods.

## 4. Modeling

- **Targets**: Binary labels indicating if next-day Open/Close > today’s Close.  
- **Data Splits**:  
  - **Train**: 2020–2021  
  - **Validation**: Jan–May 2022  
  - **Test**: Jun–Jul 2022  
  - TimeSeriesSplit strategy.  
- **Algorithms**:  
  - Logistic Regression  
  - MLP Classifier  
  - XGBoost  
  - Stacking Ensemble (with hyperparameter tuning and early stopping)  
- **Evaluation**:  
  - Classification reports, confusion matrices  
  - ROC-AUC  
  - Permutation feature importance  
  - SHAP analysis

## 5. Key Findings

- Tweet volume spikes often align with earnings or major news-driven price gaps.  
- Smoothed sentiment signals help reduce noise and improve model stability.  
- Earnings-week dummy and sentiment×volume interactions rank highly in feature importance.  
- The stacking ensemble achieved ~68% accuracy and ROC-AUC ~0.66 on the test set.

## Usage

1. **Install Dependencies**

   ```bash
   pip install pandas numpy yfinance sentence-transformers transformers faiss-cpu xgboost scikit-le
