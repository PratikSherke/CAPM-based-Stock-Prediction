# 📈 Advanced Stock Market Analytics

![Stock Market Analytics](https://img.shields.io/badge/Project-Stock%20Market%20Analytics-blueviolet) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red)

A powerful, data-driven platform for stock market analysis, integrating **Capital Asset Pricing Model (CAPM)** for risk and return evaluation with **machine learning-based stock price predictions** using GRU, LSTM, SVM, and Random Forest models. Built with Python and Streamlit, this project delivers interactive visualizations and actionable insights for financial analysis.

## 🚀 Project Overview

This project provides a comprehensive suite for analyzing stock performance and forecasting prices. It combines traditional financial modeling with advanced machine learning techniques to empower users with robust stock market insights. Key features include:

- **CAPM Analysis**: Calculates beta and expected returns for single and multiple stocks, benchmarking against the S&P 500.
- **Machine Learning Predictions**: Forecasts stock prices using GRU, LSTM, SVM, and Random Forest models with customizable lookback and prediction periods.
- **Interactive Visualizations**: Dynamic Plotly charts for stock prices, normalized prices, beta scatter plots, and prediction trends.
- **User-Friendly Interface**: Built with Streamlit for seamless stock selection and parameter tuning.
- **Data Integration**: Leverages `yfinance` for real-time stock data and `pandas_datareader` for S&P 500 data.

This project is ideal for financial analysts, data scientists, and investors looking to evaluate stock risk and predict future performance.

## 🛠️ Features

- **Single Stock CAPM Analysis** (`Single_Stock_CAPM_Analysis.py`):
  - Computes beta and expected returns using CAPM.
  - Visualizes stock price trends and beta scatter plots against S&P 500.
  - Supports 50+ major stocks (e.g., TSLA, AAPL, MSFT).
  - Allows data export as CSV.

- **Multi-Stock CAPM Analysis** (`Stock_CAPM_Analysis.py`):
  - Analyzes up to 4 stocks simultaneously for comparative beta and return analysis.
  - Displays normalized stock prices and beta comparison bar charts.
  - Interactive Plotly visualizations for stock performance.

- **Stock Price Prediction** (`Stock_Price_Prediction.py`):
  - Predicts future stock prices using GRU, LSTM, SVM, and Random Forest models.
  - Evaluates model performance with MSE, RMSE, MAE, and R² metrics.
  - Visualizes predictions alongside historical prices.
  - Provides company information (e.g., market cap, price change) and recent stock data.

## 📊 Demo

![Stock Price Prediction](https://via.placeholder.com/600x300.png?text=Stock+Price+Prediction+Demo)  
*Interactive dashboard showing stock price predictions and CAPM analysis for selected stocks.*

## 📋 Prerequisites

- Python 3.8+
- Streamlit
- Required libraries (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/advanced-stock-market-analytics.git
   cd advanced-stock-market-analytics
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run Single_Stock_CAPM_Analysis.py
   ```
   or
   ```bash
   streamlit run Stock_Price_Prediction.py
   ```

## 📂 Repository Structure

```
advanced-stock-market-analytics/
├── Single_Stock_CAPM_Analysis.py  # Single stock CAPM analysis
├── Stock_CAPM_Analysis.py         # Multi-stock CAPM analysis
├── Stock_Price_Prediction.py      # Machine learning-based price predictions
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

## 💻 Usage

1. **Launch the Streamlit app**:
   - Run one of the Python scripts (e.g., `streamlit run Stock_Price_Prediction.py`).
   - Access the app in your browser (typically `http://localhost:8501`).

2. **Select Stocks**:
   - Choose from a list of 50+ major stocks (e.g., TSLA, AAPL, AMZN) via the dropdown menu.
   - For multi-stock analysis, select up to 4 stocks.

3. **Configure Parameters**:
   - Set the analysis period (1–10 years for CAPM).
   - Adjust lookback (30–120 days) and prediction (5–30 days) periods for price forecasting.

4. **Explore Results**:
   - View beta values, expected returns, and interactive charts for CAPM analysis.
   - Analyze predicted stock prices, model performance metrics, and company details for forecasting.

## 📈 Example Output

- **CAPM Analysis**:
  - Beta for TSLA: 1.23
  - Expected Return: -2.45%
  - Visualizations: Stock price trends, beta scatter plots.

- **Price Predictions**:
  - GRU Prediction for AMZN (15 days): $185.20–$190.50
  - LSTM RMSE: 3.45
  - Visualizations: Actual vs. predicted prices for multiple models.

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Interactive web interface.
- **yfinance & pandas_datareader**: Real-time stock and market data.
- **Plotly**: Dynamic visualizations.
- **TensorFlow & scikit-learn**: Machine learning models (GRU, LSTM, SVM, RF).
- **Pandas & NumPy**: Data manipulation and analysis.

## 📝 Future Enhancements

- Add support for additional financial models (e.g., Fama-French).
- Incorporate real-time news sentiment analysis for enhanced predictions.
- Expand stock selection to include international markets.
- Deploy the app to a cloud platform (e.g., Streamlit Cloud).

## 👨‍💻 Author

- **Your Name**  
  GitHub: [your-username](https://github.com/your-username)  
  LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)  
  Email: your.email@example.com

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data.
- [Streamlit](https://streamlit.io/) for the interactive interface.
- [TensorFlow](https://www.tensorflow.org/) and [scikit-learn](https://scikit-learn.org/) for machine learning models.

---

⭐ **Star this repository** if you find it useful!  
📧 Feel free to reach out for collaboration or feedback.