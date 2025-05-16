# NetflixTweets_Stock_Sentiment-Analysis
OverviewThis project investigates whether social media sentiment, tweet volume, and traditional price/volume signals can predict Netflix (NFLX) stock price movements. By combining Twitter data with historical price data, we engineer features and evaluate classification and regression models to forecast next-day opening and closing price direction.

1. Data Collection

Tweets: Collected from Twitter API between January 1, 2020 and July 11, 2022, filtered for cashtags like $NFLX.

Stock Prices: Retrieved via yfinance and pre-saved as Parquet files containing daily Open, Close, High, Low, Volume.

2. Data Preprocessing

Tweet Cleaning: Removed URLs, mentions, hashtags, non-ASCII characters; replaced cashtags with [TICKER] tokens; tokenized and lowercased.

Weekend Adjustment: Tweets from Saturday/Sunday pushed to next Monday to align with trading days.

Aggregation: Daily tweet volume (tweetVol) and cleaned text stored for feature creation.

3. Feature Engineering

Sentiment Scores: Computed per-tweet scores using BERTweet, DistilBERT (SST-2), and Loughran–McDonald lexicon.

FAISS Smoothing: Smoothed BERTweet and DistilBERT scores by averaging over 500 nearest neighbors in embedding space.

Daily Aggregates: Rolled 7-day z-scores of tweet volume; computed 1-day/7-day lags for volume and sentiment.

Ensemble & Interaction Terms: Combined sentiment models into mean ensemble scores; added sentiment × z_vol_7d interactions; flagged earnings windows around quarterly releases.

4. Modeling

Targets: Binary labels for whether next-day Open/Close > today’s Close.

Splits: Train (2020–2021), Validation (Jan–May 2022), Test (Jun–Jul 2022), using time-series split.

Algorithms: Logistic Regression, MLP Classifier, XGBoost, and a Stacking Ensemble with tuned hyperparameters and early stopping.

Evaluation: Permutation importance, SHAP analysis, classification reports, ROC-AUC, and confusion matrices.

5. Key Findings

Tweet volume spikes often align with earnings or news-driven price gaps.

Smoothed sentiment signals reduce noise and can modestly enhance predictive performance.

Earnings-week dummy and sentiment×volume interactions are among top predictors.

Stacking ensemble achieved approximately 68% accuracy and ROC-AUC around 0.66 on the test set.

Usage

Install dependencies: pandas, numpy, yfinance, sentence-transformers, transformers, faiss-cpu, xgboost, scikit-learn, shap.

Place raw CSV/Parquet files under data/.

Run cells sequentially in the Jupyter notebook to reproduce data processing, feature engineering, model training, and evaluation.
