# 📈 Stock Price Prediction using LSTM

This project builds a deep learning model using **LSTM (Long Short-Term Memory)** networks to predict stock prices. We use historical stock data (from **2015 to 2024**) to train the model and evaluate its ability to forecast future prices.

---

## 📌 Project Highlights

* ✅ **Model Type**: Stacked LSTM (64 → 32 units)
* 📊 **Dataset**: TCS stock data from Yahoo Finance (`TCS.NS`)
* 📅 **Date Range**: 2015 to 2024
* 📉 **Performance Metrics**:

  * Final Validation MAE: \~2.09 ₹
  * Final Validation MAPE: \~3.42% (≈96.6% prediction accuracy)
  * Final RMSE (Test Set): ₹122.97
* ⚡ **Execution Time**: \~3s per epoch (10 epochs)

---

## 🧠 Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
```

* **Loss**: `mean_squared_error`
* **Metrics**: `mae`, `mape`

---

## 📦 Features Used

* Close price (only)
* Data normalized using MinMaxScaler
* 30-day lookback window for sequence generation

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/AnirudhPhophalia/Stock_price_rnn_lstm_model.git
   ```
2. Open the notebook and run cells step-by-step:

   * Download data
   * Normalize
   * Train LSTM
   * Evaluate performance

---

## 🔮 Next Steps

* Add more features (Moving Average, RSI, Volume)
* Predict next-day price
* Build a Streamlit UI for interactive prediction
* Try GRU or attention-based models

---

## 🧑‍💻 Author

**Anirudh Phophalia**
*This project is built for learning deep learning with text (NLP) using RNN architectures.*
