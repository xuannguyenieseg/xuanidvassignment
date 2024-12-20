# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import requests
import urllib
st.title("XUAN'S STOCK DASHBOARD ⭐")

# Create the sidebar
    # Get the list of stock tickers from S&P500
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
global ticker
ticker = st.sidebar.selectbox("**Ticker**", ticker_list)
global start_date, end_date
start_date = st.sidebar.date_input("**Start Date**", datetime.today().date() - timedelta(days=365))
end_date = st.sidebar.date_input("**End Date**", datetime.today().date())

# Using the professor's code to get financial infor from Yahoo Finance
class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

# Creat a function to get the stock information table
@st.cache_data
def GetStockData(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)  # Drop the indexes
    return stock_df
# Creat a function retrieves information from Yahoo Finance.
def GetCompanyData(ticker):
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj
# Create the main data_frames
global stock_price, company_data
stock_price = GetStockData(ticker, start_date, end_date)
company_data = GetCompanyData(ticker)

# Fetch data if Update button is pressed
button0 = st.sidebar.button("**Update**")
if button0:
    ticker = ticker
    stock_price = GetStockData(ticker, start_date, end_date)
    company_data = GetCompanyData(ticker)
else: # default is MMM
    ticker = 'MMM'
    stock_price = GetStockData(ticker, start_date, end_date)
    company_data = GetCompanyData(ticker)

# Create 5 tabs including Summary, Chart, Financials, Monte Carlo simulation, Top 10 News
summary, chart, financials, mc_simulation, top10 = st.tabs(["**Summary**", "**Chart**", "**Financials**", "**Monte Carlo Simulation**", "**Top 10 Companies**"])
# Coding for tab1 - Summary
def render_tab1():
    # Create 2 columes: one for some statistics as a DataFrame, the rest one for chart
    col1, col2 = st.columns([1,2])
    with col1: # show the DataFrame
        col1.write('**1. Key Statistics:**')
        info_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume',
                     'fiftyTwoWeekHigh': '52 Week High',
                     'fiftyTwoWeekLow': '52 Week Low',
                     'trailingPE'   : 'P/E Ratio (TTM)'}
        company_stats = {}  # Dictionary
        for key in info_keys:
            company_stats.update({info_keys[key]:company_data.info[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
        st.dataframe(company_stats) # show the DataFrame
    with col2: # show chart
        st.write('**2. Chart**')       
        fig = make_subplots(specs=[[{"secondary_y": True}]]) # initiate the plot figure
        # date_buttons
        date_buttons1 = [
    {'count': 1, 'label': '1M', 'step': "month", 'stepmode': "backward"},
    {'count': 3, 'label': '3M', 'step': "month", 'stepmode': "backward"},
    {'count': 1, 'label': 'YTD', 'step': "year", 'stepmode': "todate"},
    {'count': 1, 'label': '1Y', 'step': "year", 'stepmode': "backward"},
    {'count': 3, 'label': '3Y', 'step': "year", 'stepmode': "backward"},
    {'count': 5, 'label': '5Y', 'step': "year", 'stepmode': "backward"},
    {'step': "all", 'label': 'MAX'}]
        # stock_price price are plot
        are_plot = go.Scatter(x=stock_price['Date'], y=stock_price['Close'], fill = 'tozeroy', fillcolor= 'rgba(133, 133, 241, 0.15)', showlegend=False)
        fig.add_trace(are_plot, secondary_y=True)
        #Stock_price volume bar plot
        bar_plot = go.Bar(x=stock_price['Date'], y=stock_price['Volume'], marker_color = "grey", showlegend=False)
        fig.add_trace(bar_plot, secondary_y=False)
        #customize and show the figure
        fig.update_layout(template='plotly',xaxis_title='Date',
                    yaxis_title='Volume',
                    yaxis2_title='Stock Price (USD)', title = 'Price movement of ' + str(ticker),xaxis=dict(rangeselector=dict(buttons=date_buttons1 )))
        fig.update_yaxes(range=[0,250000000], secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)

    # Show the company profile and description using markdown + HTML
    # Profile
    st.write('**3. Company Profile and Description:**')
    company_profile = f""" 
        **Name:** {company_data.info.get('shortName', 'N/A')}  
        **Sector:** {company_data.info.get('sector', 'N/A')}  
        **Industry:** {company_data.info.get('industry', 'N/A')}  
        **Full Time Employees:** {company_data.info.get('fullTimeEmployees', 'N/A')}  
        **Phone:** {company_data.info.get('phone', 'N/A')}  
        **Website:** [{company_data.info.get('website', 'N/A')}]({company_data.info.get('website', '#')})  
        """
    st.markdown(company_profile)
    # Description
    st.markdown('<div style="text-align: justify;">' + \
                    company_data.info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
    # Holders

    major_holders = company_data.major_holders  # get infor about major holders
    institutional_holders = company_data.institutional_holders  # get infor about institutional holders
    # Convert the "Date Reported" Column to more readable format
    institutional_holders['Date Reported'] = institutional_holders['Date Reported'].dt.date
    # Show information about major holders
    st.write("**4. Shareholders**")
    st.write("Major Holders:")
    if isinstance(major_holders, pd.DataFrame):
        major_holders.index = [
    "Percentage of Shares Held by Insiders",
    "Percentage of Shares Held by Institutions",
    "Percentage of Float Held by Institutions",
    "Number of Institutions Holding Shares"] # rename of the index colume to make the infor clearer
        # reformat the number
        major_holders['Value'] = major_holders['Value'].apply(lambda x: f"{x * 100:.3f}%" if x <= 1 else f"{int(x):,}")
        st.dataframe(major_holders, use_container_width=True)
    else:
        st.write("No major holders data available.")
    # Show information about institution holders
    st.write("Institutional Holders:")
    if isinstance(institutional_holders, pd.DataFrame):
        st.dataframe(institutional_holders, hide_index=True, use_container_width=True)
    else:
        st.write("No institutional holders data available.")
# Adding the information of tab1 - summary                    
with summary:
    render_tab1()
# Coding for tab2 - chart
def render_tab2():
    # Create a button to switch between two types of charts
    type_chart = st.checkbox("Candle stick Chart")

    if type_chart:
    # Create date buttons
        date_buttons = [
    {'count': 1, 'label': '1M', 'step': "month", 'stepmode': "backward"},
    {'count': 3, 'label': '3M', 'step': "month", 'stepmode': "backward"},
    {'count': 1, 'label': 'YTD', 'step': "year", 'stepmode': "todate"},
    {'count': 1, 'label': '1Y', 'step': "year", 'stepmode': "backward"},
    {'count': 3, 'label': '3Y', 'step': "year", 'stepmode': "backward"},
    {'count': 5, 'label': '5Y', 'step': "year", 'stepmode': "backward"},
    {'step': "all", 'label': 'MAX'}] 
         # Create candle stick chart     
        fig = make_subplots(specs=[[{"secondary_y": True}]])              
        fig.add_trace(go.Candlestick(
            x=stock_price['Date'],
            open=stock_price['Open'],
            high=stock_price['High'],
            low=stock_price['Low'],
            close=stock_price['Close']),secondary_y=True)
        # Add bar chart of volume
        fig.add_trace(go.Bar(x=stock_price['Date'], y=stock_price['Volume'], marker_color=np.where(stock_price['Close'].pct_change() > 0, "green", "red"), showlegend=False),secondary_y=False)
        # Update layout
        fig.update_yaxes(range=[0,250000000], secondary_y=False)
        fig.update_layout(xaxis=dict(rangeselector=dict(buttons=date_buttons ),rangeslider=dict(visible=False)), # You can set this to True if needed),
                            title= f"Stock Price and Volume of {ticker}",
                             yaxis_title="Stock Price",
                            yaxis2_title="Volume")
        st.plotly_chart(fig, use_container_width=True)

    else: # Create the Line chart for the stock price + bar chart for the volume
        fig2 = make_subplots(specs=[[{"secondary_y": True}]]) # initiate the plot figure
        # Create date buttons
        date_buttons2 = [
    {'count': 1, 'label': '1M', 'step': "month", 'stepmode': "backward"},
    {'count': 3, 'label': '3M', 'step': "month", 'stepmode': "backward"},
    {'count': 1, 'label': 'YTD', 'step': "year", 'stepmode': "todate"},
    {'count': 1, 'label': '1Y', 'step': "year", 'stepmode': "backward"},
    {'count': 3, 'label': '3Y', 'step': "year", 'stepmode': "backward"},
    {'count': 5, 'label': '5Y', 'step': "year", 'stepmode': "backward"},
    {'step': "all", 'label': 'MAX'}]
        # stock_price price plot
        are_plot = go.Scatter(x=stock_price['Date'], y=stock_price['Close'], fill = 'tozeroy', fillcolor= 'rgba(133, 133, 241, 0.15)', showlegend=False)
        fig2.add_trace(are_plot, secondary_y=True)
        #Stock_price volume bar plot
        bar_plot = go.Bar(x=stock_price['Date'], y=stock_price['Volume'], marker_color=np.where(stock_price['Close'].pct_change() > 0, "green", "red"), showlegend=False)
        fig2.add_trace(bar_plot, secondary_y=False)
        #customize and show the figure
        fig2.update_layout(title = f"Stock Price and Volume of {ticker}",template='plotly',xaxis_title='Date',
                     yaxis_title='Volume',
                    yaxis2_title='Stock Price (USD)',xaxis=dict(rangeselector=dict(buttons=date_buttons2 )))
        fig2.update_yaxes(range=[0,250000000], secondary_y=False)
        st.plotly_chart(fig2, use_container_width=True)

# Adding the information of tab2 - chart
with chart:
    render_tab2()

# coding for tab3 - financials
def render_tab3():
    # Financial selection (Income Statement, Balance Sheet, Cash Flow)
    financial_option = st.selectbox(f"**Financials of {ticker}**", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period_option = st.radio("**Period**", ["Annual", "Quarterly"])

    # Display financials based on the selected option
    if financial_option == "Income Statement":
        fin_data = company_data.financials if period_option == "Annual" else company_data.quarterly_financials
    elif financial_option == "Balance Sheet":
        fin_data = company_data.balance_sheet if period_option == "Annual" else company_data.quarterly_balance_sheet
    else:
        fin_data = company_data.cashflow if period_option == "Annual" else company_data.quarterly_cashflow
    st.write(fin_data)
#Adding the information of tab3 - financials
with financials:
    render_tab3()
# coding for tab4 - Monte Carlo Simulation
def render_tab4():
    cola, colb = st.columns([1,1]) # Create 2 columns for nbr simulation and time horizone selection
    with cola:
        simulations = cola.selectbox("**Number of simulations**", (200, 500, 1000))
    with colb:
        time_horizone = colb.selectbox("**Time horizone (days)**", (30, 60, 90))
    simulation_df = pd.DataFrame()
    close_price = stock_price['Close']
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)
    for i in range(simulations):
        next_price = [] # the list to store the next stock price
        last_price = close_price.iloc[-1] # create the next stock price
        for j in range(time_horizone):
            future_return = np.random.normal(0, daily_volatility) # Generate the random percentage change around the mean 0 and std(daily_volatility)
            future_price = last_price * (1 + future_return) # Generate the random future price
            # Save the price and go next
            next_price.append(future_price)
            last_price = future_price
        # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis = 1)
    # Calculate Value at Risk (VaR) at 95% confidence level
    VaR_95 = np.percentile(simulation_df.iloc[-1, :], 5)  # 5th percentile of the last day's simulated prices
    current_price = close_price.iloc[-1]
    VaR_amount = current_price - VaR_95

    # plot the simulation stock price in the future
    fig, ax = plt.subplots()
    ax.plot(simulation_df)
    ax.axhline(y=current_price, color = 'red')
    # Customize the plot
    ax.set_title(f"Monte Carlo Simulation for {ticker} stock price in the next {time_horizone} days")
    ax.set_xlabel('Day')
    ax.set_ylabel('Price')
    ax.legend(['Current stock price is: ' + str(np.round(current_price, 2))])
    ax.get_legend().legend_handles[0].set_color('red')
    st.pyplot(fig)
    # st.plotly_chart(fig, use_container_width=True)
    st.write(f"Value at Risk (VaR) at 95% confidence level: ${np.round(VaR_amount, 2)}")
#Adding the information of tab4 - Monte Carlo Simulation
with mc_simulation:
    render_tab4()

# coding for tab5 - Top 10 companies:
# Create a function to fetch S&P 500 data with tickers, companies and industries
@st.cache_data
def fetch_sp500_data():
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0][['Symbol', 'Security', 'GICS Sector']] # Extract relevant columns
    sp500_table.columns = ['Ticker', 'Company', 'Industry']  # Rename columns
    return sp500_table
def render_tab5():
    # Fetch and prepare data
    sp500_data = fetch_sp500_data()

    # Create industry selection dropdown
    industry_list = sp500_data['Industry'].unique()
    selected_industry = st.selectbox("**Select an Industry**", industry_list)
   
    # Filter data by selected industry
    industry_data = sp500_data[sp500_data['Industry'] == selected_industry]
    # Initialize lists for additional columns
    market_caps = []
    previous_closes = []
    opens = []
    week_highs = []
    week_lows = []
    volumes = []
    pe = []
    for ticker in industry_data['Ticker']:
        company_data = yf.Ticker(ticker)
        info = company_data.info
        market_caps.append(info.get("marketCap", "N/A"))
        previous_closes.append(info.get("previousClose", "N/A"))
        opens.append(info.get("open", "N/A"))
        week_highs.append(info.get("fiftyTwoWeekHigh", "N/A"))
        week_lows.append(info.get("fiftyTwoWeekLow", "N/A"))
        volumes.append(info.get("volume", "N/A"))
        pe.append(info.get("trailingPE", "N/A"))
     # Add new columns to the industry data
    industry_data['Market Cap'] = market_caps
    industry_data['Previous Close'] = previous_closes
    industry_data['Open'] = opens
    industry_data['52 Week High'] = week_highs
    industry_data['52 Week Low'] = week_lows
    industry_data['Volume'] = volumes
    industry_data['P/E Ratio (TTM)'] = pe
    # Sort by Market Cap and select Top 10
    top_10_companies = industry_data.sort_values(by='Market Cap', ascending=False).head(10)
    top_10_companies.index = range(1, len(top_10_companies)+1)
    top_10_companies.index.name = 'Rank'

    # Display the results
    st.subheader(f"Top 10 Companies in {selected_industry} by Market Cap")
    st.dataframe(top_10_companies[['Ticker', 'Company', 'Market Cap', 'Previous Close', 'Open', 
                                   '52 Week High', '52 Week Low', 'Volume', 'P/E Ratio (TTM)']])

with top10:
    render_tab5()

# Change the background color with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# End