import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="📈 Single Stock CAPM Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for enhanced styling with improved color combination
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0C0C; /* Deep navy background */
        color: #E0E0E0; /* Light gray text */
    }
    .main { padding: 20px; }
    h1 {
        color: #FFD700; /* Golden yellow for main title */
        text-align: center;
        font-size: 2.5em;
       
        background: linear-gradient(45deg, #E4E5EA, #EBF0F3); /* Gradient background */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
        border-radius: 10px;
    }
    h2 { color: #FF69B4; } /* Vibrant pink for headers */
    h3 { color: #00BFFF; } /* Vibrant cyan for subheaders */
    .stDataFrame {
        border: 1px solid #00BFFF;
        border-radius: 5px;
        background-color: #1A2330; /* Darker navy for tables */
    }
    .stDataFrame table {
        background-color: #C2C8D6;
        color: #E0E0E0;
    }
    .stDataFrame th {
        background-color: #00BFFF;
        color: #0A0F1A;
    }
    .stSelectbox, .stNumberInput {
        background-color: #1A2330;
        color: #E0E0E0;
        border: 1px solid #00BFFF;
        border-radius: 5px;
    }
    .stSelectbox div, .stNumberInput div {
        color: #E0E0E0;
    }
    .stButton > button {
        background-color: #32CD32; /* Lime green buttons */
        color: #0A0F1A;
        border-radius: 5px;
    }
    .st-expander {
        border: 1px solid #00BFFF;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Function to plot stock price with enhanced styling
def plot_stock_price(df, stock_name):
    fig = px.line(
        df,
        x='Date',
        y=stock_name,
        title=f'{stock_name} Stock Price Over Time',
        line_shape='linear'
    )
    fig.update_traces(line=dict(color='#FFD700', width=2))  # Golden yellow line
    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor="#F4F6F9",
        xaxis=dict(gridcolor='#E0E0E0', title='Date', tickangle=45),
        yaxis=dict(gridcolor='#E0E0E0', title='Price (USD)'),
        font=dict(color='#E0E0E0'),
        showlegend=False,
        title_x=0.5,
        hovermode="x unified",
        template="plotly_dark"
    )
    return fig

# Function to plot beta scatter plot
def plot_beta_scatter(stocks_daily_return, stock, beta):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stocks_daily_return['sp500'],
        y=stocks_daily_return[stock],
        mode='markers',
        marker=dict(color='#FFD700', size=8),  # Golden yellow markers
        name='Daily Returns'
    ))
    # Add regression line
    x_range = np.array([stocks_daily_return['sp500'].min(), stocks_daily_return['sp500'].max()])
    y_range = beta * x_range
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        line=dict(color='#FF69B4', width=2),  # Pink regression line
        name=f'Beta = {round(beta, 2)}'
    ))
    fig.update_layout(
        title=f'Beta: {stock} vs. S&P 500',
        xaxis_title='S&P 500 Daily Returns (%)',
        yaxis_title=f'{stock} Daily Returns (%)',
        plot_bgcolor='#ffffff',
        paper_bgcolor="#F2F4F6",
        xaxis=dict(gridcolor='#E0E0E0'),
        yaxis=dict(gridcolor='#E0E0E0'),
        font=dict(color='#E0E0E0'),
        title_x=0.5,
        template="plotly_dark",
        showlegend=True
    )
    return fig

# Main app layout
st.markdown("<h1> 📈 Single Stock CAPM Analysis</h1>", unsafe_allow_html=True)

# 1. Input Selection
with st.expander("🔍 Input Parameters", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        stock = st.selectbox(
            "Choose a stock",
            ('MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO',
             'BRK-B', 'TSLA', 'WMT', 'JPM', 'V', 'LLY', 'MA', 'NFLX', 'ORCL',
             'COST', 'XOM', 'PG', 'JNJ', 'HD', 'BAC', 'ABBV', 'KO', 'PLTR', 'PM',
             'TMUS', 'UNH', 'GE', 'CRM', 'CSCO', 'IBM', 'WFC', 'CVX', 'ABT', 'LIN',
             'MCD', 'INTU', 'NOW', 'AXP', 'MS', 'DIS', 'T', 'ISRG', 'ACN', 'MRK',
             'GS', 'AMD', 'RTX'),
            index=9  # Default to TSLA
        )
    with col2:
        year = st.number_input("Number of years", min_value=1, max_value=10, value=1)

# Process data and display results
try:
    # Downloading data for the selected stock and SP500
    end = datetime.date.today()
    start = datetime.date(end.year - year, end.month, end.day)

    sp500 = web.DataReader(['sp500'], 'fred', start, end)
    data = yf.download(stock, period=f'{year}y')
    
    stocks_df = pd.DataFrame()
    stocks_df[stock] = data['Close']
    stocks_df.reset_index(inplace=True)
    
    sp500.reset_index(inplace=True)
    sp500.columns = ['Date', 'sp500']
    
    stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
    stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df = pd.merge(stocks_df, sp500, on='Date', how='inner')

    # Calculate daily returns
    stocks_daily_return = daily_return(stocks_df)

    # Calculate beta and alpha
    beta, alpha = calculate_beta(stocks_daily_return, stock)

    # Calculate CAPM expected return
    rf = 0  # Risk-free rate
    rm = stocks_daily_return['sp500'].mean() * 252  # Market return
    expected_return = round(rf + beta * (rf - rm), 2)

    # 2. Results Section
    with st.expander("📊 Results", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Beta Value")
            beta_df = pd.DataFrame({
                'Stock': [stock],
                'Beta Value': [round(beta, 2)]
            })
            st.dataframe(beta_df, use_container_width=True)
        with col2:
            st.markdown("### Expected Return (CAPM)")
            return_df = pd.DataFrame({
                'Stock': [stock],
                'Return Value (%)': [expected_return]
            })
            st.dataframe(return_df, use_container_width=True)

    # 3. Visualizations Section
    with st.expander("📈 Visualizations", expanded=True):
        st.markdown(f"### {stock} Stock Price Over {year} Year(s)")
        st.plotly_chart(plot_stock_price(stocks_df, stock), use_container_width=True)

        st.markdown("### Beta: Daily Returns vs. S&P 500")
        st.plotly_chart(plot_beta_scatter(stocks_daily_return, stock, beta), use_container_width=True)

    # 4. Download Data
    with st.expander("💾 Download Data", expanded=False):
        st.download_button(
            "Download Stock Data",
            stocks_df.to_csv(index=False),
            f"{stock}_stock_data.csv",
            help="Download the stock price data as a CSV file."
        )

except Exception as e:
    st.error(f"Error retrieving data: {str(e)}. Please try a different stock or time period.")