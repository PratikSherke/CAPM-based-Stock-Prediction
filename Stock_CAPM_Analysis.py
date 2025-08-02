import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import pandas_datareader.data as web
import plotly.graph_objects as go

# Streamlit app configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="CAPM Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #B8D0DA;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color:#1483D4;
        font-family: 'Arial', sans-serif;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Function to plot interactive Plotly chart with enhanced styling
def interactive_plot(df, title="Stock Prices"):
    fig = go.Figure()
    colors = ['#4e73df', '#1cc88a', '#f6c23e', '#e74a3b', '#36b9cc']
    for idx, i in enumerate(df.columns[1:]):
        fig.add_scatter(
            x=df['Date'], 
            y=df[i], 
            name=i,
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}'
        )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20, color='#000000')),
        width=750,
        height=400,
        margin=dict(l=40, r=40, t=120, b=40),
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1,
            font=dict(size=12, color='#000000')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='Date',
            titlefont=dict(color='#000000'),
            tickfont=dict(color='#000000'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickformat='%Y-%m-%d'
        ),
        yaxis=dict(
            title='Price (USD)',
            titlefont=dict(color='#000000'),
            tickfont=dict(color='#000000'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        hovermode='x unified',
        font=dict(family='Arial', size=12, color='#000000')
    )
    return fig

# Function to normalize the prices based on the initial price
def normalize(df_2):
    df = df_2.copy()
    for i in df.columns[1:]:
        df[i] = df[i] / df[i][0]
    return df

# Function to calculate daily returns
def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range(1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1]) / df[i][j-1]) * 100
        df_daily_return[i][0] = 0
    return df_daily_return

# Function to calculate beta
def calculate_beta(stocks_daily_return, stock):
    rm = stocks_daily_return['sp500'].mean() * 252
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[stock], 1)
    return b, a

# Title with emoji and styling
st.title("📊 Stock Analysis: Capital Asset Pricing Model")

# Getting input from user
col1, col2 = st.columns([1, 1])
with col1:
    stocks_list = st.multiselect(
        "Choose up to 4 stocks", 
        ('MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO',
         'BRK-B', 'TSLA', 'WMT', 'JPM', 'V', 'LLY', 'MA', 'NFLX', 'ORCL',
         'COST', 'XOM', 'PG', 'JNJ', 'HD', 'BAC', 'ABBV', 'KO', 'PLTR', 'PM',
         'TMUS', 'UNH', 'GE', 'CRM', 'CSCO', 'IBM', 'WFC', 'CVX', 'ABT', 'LIN',
         'MCD', 'INTU', 'NOW', 'AXP', 'MS', 'DIS', 'T', 'ISRG', 'ACN', 'MRK',
         'GS', 'AMD', 'RTX'),
        ['TSLA', 'AAPL', 'NFLX', 'MSFT'],
        max_selections=4
    )
with col2:
    year = st.number_input("Number of years", min_value=1, max_value=10, value=1)

# Downloading data for SP500
end = datetime.date.today()
start = datetime.date(end.year - year, end.month, end.day)
sp500 = web.DataReader(['sp500'], 'fred', start, end)

stocks_df = pd.DataFrame()
for stock in stocks_list:
    data = yf.download(stock, period=f'{year}y')
    stocks_df[f'{stock}'] = data['Close']

stocks_df.reset_index(inplace=True)
sp500.reset_index(inplace=True)
sp500.columns = ['Date', 'sp500']
stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df = pd.merge(stocks_df, sp500, on='Date', how='inner')

# Display dataframes
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 📋 Stock Closing Price - Early 5 Trading Days")
    st.dataframe(stocks_df.head(), use_container_width=600, height=220)
with col2:
    st.markdown("### 📋 Stock Closing Price - Latest 5 Trading Days")
    st.dataframe(stocks_df.tail(), use_container_width=600, height=220)

# Plot stock prices
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 📈 Stock Prices")
    st.plotly_chart(interactive_plot(stocks_df, "Stock Prices Over Time"), use_container_width=True)
with col2:
    st.markdown("### 📈 Normalized Stock Prices")
    st.plotly_chart(interactive_plot(normalize(stocks_df), "Normalized Stock Prices"), use_container_width=True)

# Calculate and display daily returns
stocks_daily_return = daily_return(stocks_df)

# Calculate beta and alpha
beta = {}
alpha = {}
for i in stocks_daily_return.columns:
    if i != 'Date' and i != 'sp500':
        b, a = calculate_beta(stocks_daily_return, i)
        beta[i] = b
        alpha[i] = a

beta_df = pd.DataFrame(columns=['Stock', 'Beta Value'])
beta_df['Stock'] = beta.keys()
beta_df['Beta Value'] = [str(round(i, 2)) for i in beta.values()]

# Display beta values
with col1:
    st.markdown('### 📊 Calculated Beta Values')
    st.dataframe(beta_df, use_container_width=True)

# Calculate and display CAPM returns
rf = 0  # Risk-free rate
rm = stocks_daily_return['sp500'].mean() * 252  # Market return
return_df = pd.DataFrame()
return_value = []
for stock, value in beta.items():
    return_value.append(round(rf + value * (rf - rm), 2))
return_df['Stock'] = list(beta.keys())
return_df['Return Value'] = return_value

with col2:
    st.markdown('### 📊 Expected Returns (CAPM)')
    st.dataframe(return_df, use_container_width=True)

# Additional visualization: Beta comparison bar chart
st.markdown("### 📊 Beta Comparison Across Stocks")
fig_beta = go.Figure(data=[
    go.Bar(
        x=beta_df['Stock'], 
        y=[float(x) for x in beta_df['Beta Value']],
        marker_color='#4e73df',
        text=beta_df['Beta Value'],
        textposition='auto',
        textfont=dict(color='#000000')
    )
])
fig_beta.update_layout(
    title=dict(text="Beta Values Comparison", x=0.5, xanchor='center', font=dict(size=20, color='#000000')),
    width=600,
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        title='Stock',
        titlefont=dict(color='#000000'),
        tickfont=dict(color='#000000'),
        showgrid=False
    ),
    yaxis=dict(
        title='Beta Value',
        titlefont=dict(color='#000000'),
        tickfont=dict(color='#000000'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    ),
    font=dict(family='Arial', size=12, color='#000000')
)
st.plotly_chart(fig_beta, use_container_width=True)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import plotly.express as px
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error
# import yfinance as yf
# import stm1  # Modified: Import stm1.py to access pre-imported data

# # Set page configuration as the first Streamlit command
# st.set_page_config(
#     page_title="Stock Analysis with SARIMA Forecast",
#     page_icon="📈",
#     layout="wide"
# )

# # Modified: Custom CSS for black background with high-contrast colors
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #000000;
#         color: #D3D3D3;
#     }
#     h1, h2, h3 {
#         color: #FFFFFF;
#     }
#     .stDataFrame {
#         border: 1px solid #4DA8DA;
#         border-radius: 5px;
#         background-color: #1C2526;
#     }
#     .stDataFrame table {
#         background-color: #1C2526;
#         color: #FFFFFF;
#     }
#     .stDataFrame th {
#         background-color: #4DA8DA;
#         color: #FFFFFF;
#     }
#     .stSelectbox {
#         background-color: #1C2526;
#         color: #FFFFFF;
#         border: 1px solid #4DA8DA;
#         border-radius: 5px;
#     }
#     .stSelectbox div {
#         color: #FFFFFF;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Function to plot stock price and SARIMA forecast
# def plot_stock_forecast(df, stock_name, forecast, train_data, test_data):
#     # Modified: Use stock_name as column instead of 'Close'
#     fig = px.line(title=f'{stock_name} Stock Price and 30-Day SARIMA Forecast')
#     # Plot actual data
#     fig.add_scatter(x=df.index, y=df[stock_name], name='Actual Price', line=dict(color='#00FFFF'))
#     # Plot training data
#     fig.add_scatter(x=train_data.index, y=train_data, name='Training Data', line=dict(color='#FFD700'))
#     # Plot test data
#     fig.add_scatter(x=test_data.index, y=test_data, name='Test Data', line=dict(color='#FF6347'))
#     # Plot forecast
#     forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
#     fig.add_scatter(x=forecast_index, y=forecast, name='30-Day Forecast', line=dict(color='#32CD32', dash='dash'))
#     fig.update_layout(
#         width=800,
#         height=400,
#         margin=dict(l=20, r=20, t=50, b=20),
#         plot_bgcolor='#1C2526',
#         paper_bgcolor='#1C2526',
#         xaxis=dict(gridcolor='#D3D3D3', title='Date', titlefont=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')),
#         yaxis=dict(gridcolor='#D3D3D3', title='Price (USD)', titlefont=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')),
#         font=dict(color='#FFFFFF'),
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
#         title_font=dict(color='#FFFFFF')
#     )
#     return fig

# # Main app layout
# st.markdown("<h1 style='color: #FFFFFF;'>Stock Analysis with SARIMA Forecast</h1>", unsafe_allow_html=True)
# st.markdown("<h2 style='color: #FFFFFF;'>Company Details and Price Forecast</h2>", unsafe_allow_html=True)

# # User input for stock selection
# stock = st.selectbox(
#     "Choose a stock",
#     ('MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO',
#      'BRK-B', 'TSLA', 'WMT', 'JPM', 'V', 'LLY', 'MA', 'NFLX', 'ORCL',
#      'COST', 'XOM', 'PG', 'JNJ', 'HD', 'BAC', 'ABBV', 'KO', 'PLTR', 'PM',
#      'TMUS', 'UNH', 'GE', 'CRM', 'CSCO', 'IBM', 'WFC', 'CVX', 'ABT', 'LIN',
#      'MCD', 'INTU', 'NOW', 'AXP', 'MS', 'DIS', 'T', 'ISRG', 'ACN', 'MRK',
#      'GS', 'AMD', 'RTX'),
#     index=9  # Default to TSLA
# )

# # Modified: Load and validate data from stm1.py
# try:
#     # Access pre-imported data
#     stocks_df = stm1.data.copy()
#     sp500_data = stm1.sp500.copy()

#     # Validate data
#     if stocks_df.empty or sp500_data.empty:
#         raise ValueError("Imported data is empty.")
#     if stock not in stocks_df.columns:
#         raise ValueError(f"Stock {stock} not found in stm1.stocks_df.")
#     if 'Date' not in stocks_df.columns or 'Date' not in sp500_data.columns:
#         raise ValueError("Date column missing in imported data.")
#     if len(stocks_df) < 100:
#         raise ValueError("Insufficient data points (<100) in stocks_df for SARIMA.")
#     if stocks_df[stock].isnull().any():
#         raise ValueError(f"Missing values in {stock} prices.")

#     # Prepare stock data
#     stock_data = pd.DataFrame()
#     stock_data['Date'] = stocks_df['Date']
#     stock_data[stock] = stocks_df[stock]
#     stock_data['sp500'] = sp500_data['sp500']
    
#     # Ensure Date is in datetime format and set as index
#     stock_data['Date'] = pd.to_datetime(stock_data['Date'])
#     stock_data.set_index('Date', inplace=True)

#     # Filter last 10 days for table
#     last_10_days = stock_data.tail(10).copy()
#     last_10_days = last_10_days[[stock]].rename(columns={stock: 'Close'})
#     last_10_days = last_10_days.round(2)
#     last_10_days.reset_index(inplace=True)
#     last_10_days['Date'] = last_10_days['Date'].dt.strftime('%Y-%m-%d')

#     # Get company info using yfinance
#     try:
#         ticker = yf.Ticker(stock)
#         description = ticker.info.get('longBusinessSummary', 'Description not available.')
#         market_cap = ticker.info.get('marketCap', 0) / 1e9
#     except Exception as e:
#         description = "Description not available due to API error."
#         market_cap = 0
#         st.warning(f"Could not retrieve company info: {str(e)}")

#     # 1. Company Description
#     st.markdown("<h3 style='color: #FFFFFF;'>Company Description</h3>", unsafe_allow_html=True)
#     st.write(description)

#     # 2. Current Stock Price
#     st.markdown("<h3 style='color: #FFFFFF;'>Current Stock Price</h3>", unsafe_allow_html=True)
#     current_price = stock_data[stock][-1]
#     st.write(f"**{stock} Closing Price (USD):** ${current_price:.2f}")

#     # 3. Market Capitalization
#     st.markdown("<h3 style='color: #FFFFFF;'>Market Capitalization</h3>", unsafe_allow_html=True)
#     st.write(f"**Market Cap (USD):** ${market_cap:.2f} Billion")

#     # 4. Profit or Loss from Previous Day
#     st.markdown("<h3 style='color: #FFFFFF;'>Price Change from Previous Day</h3>")
#     prev_price = stock_data[stock][-2]
#     price_change = current_price - prev_price
#     symbol = "↑" if price_change >= 0 else "↓"
#     symbol_color = "#32CD32" if price_change >= 0 else "#FF6347"
#     st.markdown(f"**Price Change (USD):** ${abs(price_change):.2f} <span style='color:{symbol_color}'>{symbol}</span>", unsafe_allow_html=True)

#     # 5. SARIMA Forecast for Next 30 Days
#     st.markdown("<h3 style='color: #FFFFFF;'>30-Day SARIMA Forecast</h3>", unsafe_allow_html=True)
#     # Modified: Use stock ticker column for SARIMA
#     close_prices = stock_data[stock]
#     train_size = int(len(close_prices) * 0.8)
#     train_data = close_prices[:train_size]
#     test_data = close_prices[train_size:]

#     try:
#         # Fit SARIMA model
#         model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
#         model_fit = model.fit(disp=False)

#         # Forecast for test period
#         test_forecast = pd.Series(model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1), index=test_data.index)

#         # Calculate RMSE
#         rmse = np.sqrt(mean_squared_error(test_data, test_data))

#         # Forecast for next 30 days
#         forecast = model_fit.forecast(steps=30)

#         # Display RMSE
#         st.write(f"**Root Mean Square Error (RMSE):** ${rmse:.2f}")

#         # Plot actual, test, and forecast
#         st.plotly_chart(plot_stock_forecast(stock_data, stock, forecast, train_data, test_data))

#     except Exception as e:
#         st.error(f"SARIMA model error: {str(e)}. Try a different stock or check data.")

#     # 6. Last 10 Days Data Table
#     st.markdown("<h3 style='color: #FFFFFF;'>Stock Data for Last 10 Days</h3>", unsafe_allow_html=True)
#     st.dataframe(last_10_days, use_container_width=True)

# except Exception as e:
#     st.error(f"Error processing data: {str(e)}. Ensure stm1.py has valid stocks_df and sp500_data with Date and stock columns.")