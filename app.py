
import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import ast
import os 
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
genai.GenerativeModel('gemini-1.5-flash')

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],suppress_callback_exceptions=True)
app.title = "TSLA Stock Analysis Dashboard"

def parse_price_list(price_str):
    """Parse price list from string format"""
    if pd.isna(price_str) or price_str == '':
        return []
    try:
        if isinstance(price_str, str):
            # Remove brackets and split by comma
            clean_str = price_str.strip('[]').replace(' ', '')
            if clean_str:
                return [float(x) for x in clean_str.split(',') if x]
        return []
    except:
        return []

def load_and_process_data():
    """Load and process TSLA data"""
    try:
        # Try to load the actual CSV file
        df = pd.read_csv('tlsa.csv')
        print(f"Loaded {len(df)} rows of TSLA data")
    except FileNotFoundError:
        print("CSV file not found, using enhanced sample data for demonstration")
        # Enhanced sample data with more realistic patterns
        date_range = pd.date_range('2022-08-25', '2022-12-30', freq='D')
        date_range = [d for d in date_range if d.weekday() < 5]  # Only weekdays
        
        np.random.seed(42)  # For reproducible data
        base_price = 300
        prices = []
        directions = []
        supports = []
        resistances = []
        volumes = []
        
        for i, date in enumerate(date_range):
            # Generate realistic price movement
            change = np.random.normal(0, 0.03) * base_price
            base_price = max(50, base_price + change)  # Don't go below $50
            
            # Generate OHLC data
            open_price = base_price + np.random.normal(0, 0.01) * base_price
            close_price = base_price + np.random.normal(0, 0.02) * base_price
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.015)) * base_price
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.015)) * base_price
            
            prices.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2)
            })
            
            # Generate trading signals with some logic
            if i > 5:
                recent_closes = [prices[j]['close'] for j in range(max(0, i-5), i)]
                trend = np.mean(np.diff(recent_closes))
                
                if trend > 2:
                    direction = 'LONG' if np.random.random() > 0.3 else None
                elif trend < -2:
                    direction = 'SHORT' if np.random.random() > 0.3 else None
                else:
                    direction = np.random.choice(['LONG', 'SHORT', None], p=[0.3, 0.3, 0.4])
            else:
                direction = np.random.choice(['LONG', 'SHORT', None], p=[0.3, 0.3, 0.4])
            
            directions.append(direction)
            
            # Generate support and resistance levels
            current_price = close_price
            support_levels = [
                round(current_price * (1 - np.random.uniform(0.05, 0.15)), 2) 
                for _ in range(np.random.randint(1, 4))
            ]
            resistance_levels = [
                round(current_price * (1 + np.random.uniform(0.02, 0.10)), 2) 
                for _ in range(np.random.randint(1, 4))
            ]
            
            supports.append(str(support_levels))
            resistances.append(str(resistance_levels))
            
            # Generate volume data
            base_volume = 50000000
            volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
            volumes.append(max(10000000, volume))
        
        sample_data = {
            'timestamp': [d.strftime('%Y-%m-%d') for d in date_range],
            'direction': directions,
            'Support': supports,
            'Resistance': resistances,
            'open': [p['open'] for p in prices],
            'high': [p['high'] for p in prices],
            'low': [p['low'] for p in prices],
            'close': [p['close'] for p in prices],
            'volume': volumes
        }
        df = pd.DataFrame(sample_data)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df['Date'] = pd.to_datetime(df.iloc[:, 0])
    
    # Standardize column names
    column_mapping = {
        'direction': 'Direction',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    # Parse support and resistance lists
    df['Support_List'] = df['Support'].apply(parse_price_list)
    df['Resistance_List'] = df['Resistance'].apply(parse_price_list)
    
    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    return df.sort_values('Date').reset_index(drop=True)

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def create_candlestick_chart(df):
    """Create enhanced candlestick chart with indicators"""
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('TSLA Stock Price', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='TSLA',
            increasing_line_color='#00C851',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='orange', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add trading signals
    long_signals = df[df['Direction'] == 'LONG']
    short_signals = df[df['Direction'] == 'SHORT']
    
    if not long_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=long_signals['Date'],
                y=long_signals['Low'] * 0.98,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#00C851'),
                name='LONG Signal',
                showlegend=True
            ),
            row=1, col=1
        )
    
    if not short_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=short_signals['Date'],
                y=short_signals['High'] * 1.02,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#ff4444'),
                name='SHORT Signal',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['#00C851' if close >= open else '#ff4444' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # RSI chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title="TSLA Stock Analysis with Technical Indicators",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def call_gemini_with_context(user_input, df,performance_metrics=None):
    try:
        from io import StringIO

        # Summarize the DataFrame
        recent_data = df.tail(20)
        summary = StringIO()
        recent_data.describe(include='all').to_string(buf=summary)
        summary_text = summary.getvalue()

        # Also send last few directional signals
        signals = recent_data[['Date', 'Direction', 'Close']].dropna().tail(5).to_string(index=False)

        # Prompt to Gemini
        model = genai.GenerativeModel('gemini-1.5-flash') 
        prompt = f"""
You are a financial assistant with expertise in stock market analysis.
You are analyzing Tesla (TSLA) stock data.

Here is a brief statistical summary of the recent data (last 20 trading days):

{summary_text}

Recent signal entries (Direction, Date, Close price):
{signals}

Now answer this user question using the context above, giving insights like a financial analyst:

Question: {user_input}
"""
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"❌ Gemini API failed: {str(e)}"


def create_performance_metrics(df):
    """Calculate performance metrics"""
    if len(df) == 0:
        return {}
    
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    volatility = df['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized
    max_drawdown = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
    
    long_signals = len(df[df['Direction'] == 'LONG'])
    short_signals = len(df[df['Direction'] == 'SHORT'])
    
    # Win rate calculation (simplified)
    correct_long = len(df[(df['Direction'] == 'LONG') & (df['Daily_Return'].shift(-1) > 0)])
    correct_short = len(df[(df['Direction'] == 'SHORT') & (df['Daily_Return'].shift(-1) < 0)])
    total_signals = long_signals + short_signals
    win_rate = ((correct_long + correct_short) / total_signals * 100) if total_signals > 0 else 0
    
    return {
        'Total Return': f"{total_return:.2f}%",
        'Volatility (Annualized)': f"{volatility:.2f}%",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Long Signals': long_signals,
        'Short Signals': short_signals,
        'Win Rate': f"{win_rate:.1f}%",
        'Current Price': f"${df['Close'].iloc[-1]:.2f}",
        'Price Range': f"${df['Low'].min():.2f} - ${df['High'].max():.2f}"
    }

# Load data
df = load_and_process_data()
performance_metrics = create_performance_metrics(df)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚗 TSLA Stock Analysis Dashboard", className="text-center mb-4"),
            html.P("Advanced technical analysis with AI-powered insights", 
                   className="text-center text-muted mb-4"),
        ], width=12)
    ]),
    
    dcc.Tabs(id="tabs", value="overview-tab", children=[
        dcc.Tab(label="📊 Overview", value="overview-tab"),
        dcc.Tab(label="📈 Technical Analysis", value="chart-tab"),
        dcc.Tab(label="📋 Data Table", value="table-tab"),
        dcc.Tab(label="🤖 AI Assistant", value="chatbot-tab"),
    ]),
    
    html.Div(id="tab-content")
], fluid=True)

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(active_tab):
    if active_tab == "overview-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Performance Metrics", className="card-title"),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.H5(performance_metrics['Current Price'], className="text-primary"),
                                    html.P("Current Price", className="text-muted small")
                                ], width=3),
                                dbc.Col([
                                    html.H5(performance_metrics['Total Return'], className="text-success"),
                                    html.P("Total Return", className="text-muted small")
                                ], width=3),
                                dbc.Col([
                                    html.H5(performance_metrics['Volatility (Annualized)'], className="text-warning"),
                                    html.P("Volatility", className="text-muted small")
                                ], width=3),
                                dbc.Col([
                                    html.H5(performance_metrics['Max Drawdown'], className="text-danger"),
                                    html.P("Max Drawdown", className="text-muted small")
                                ], width=3),
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯 Trading Signals", className="card-title"),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.H3(performance_metrics['Long Signals'], className="text-success"),
                                    html.P("LONG Signals", className="text-muted")
                                ], width=4),
                                dbc.Col([
                                    html.H3(performance_metrics['Short Signals'], className="text-danger"),
                                    html.P("SHORT Signals", className="text-muted")
                                ], width=4),
                                dbc.Col([
                                    html.H3(performance_metrics['Win Rate'], className="text-info"),
                                    html.P("Win Rate", className="text-muted")
                                ], width=4),
                            ])
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈 Quick Stats", className="card-title"),
                            html.Hr(),
                            html.P(f"📅 Trading Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"),
                            html.P(f"📊 Total Trading Days: {len(df)}"),
                            html.P(f"💰 Price Range: {performance_metrics['Price Range']}"),
                            html.P(f"📊 Average Volume: {df['Volume'].mean():,.0f}"),
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Price Distribution", className="card-title"),
                            dcc.Graph(
                                figure=px.histogram(
                                    df, x='Close', nbins=30,
                                    title="Distribution of Closing Prices",
                                    template="plotly_white"
                                ).update_layout(showlegend=False)
                            )
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈 Daily Returns", className="card-title"),
                            dcc.Graph(
                                figure=px.histogram(
                                    df, x='Daily_Return', nbins=30,
                                    title="Distribution of Daily Returns",
                                    template="plotly_white"
                                ).update_layout(showlegend=False)
                            )
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    elif active_tab == "chart-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id="main-chart",
                                figure=create_candlestick_chart(df),
                                style={'height': '800px'}
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 Technical Indicators Legend", className="card-title"),
                            html.Ul([
                                html.Li("🟢 Green Candlesticks: Bullish days (Close > Open)"),
                                html.Li("🔴 Red Candlesticks: Bearish days (Close < Open)"),
                                html.Li("🔵 Blue Line: 20-day Simple Moving Average"),
                                html.Li("🟠 Orange Line: 50-day Simple Moving Average"),
                                html.Li("🔺 Green Triangle Up: LONG Signal"),
                                html.Li("🔻 Red Triangle Down: SHORT Signal"),
                                html.Li("📊 Volume: Trading volume with color coding"),
                                html.Li("📈 RSI: Relative Strength Index (30-70 range highlighted)"),
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mt-4")
        ])
    
    elif active_tab == "table-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📋 TSLA Trading Data", className="card-title"),
                            html.P("Interactive data table with all trading information", className="card-text"),
                            
                            dash_table.DataTable(
                                id='data-table',
                                columns=[
                                    {'name': 'Date', 'id': 'Date', 'type': 'datetime'},
                                    {'name': 'Open', 'id': 'Open', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'High', 'id': 'High', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'Low', 'id': 'Low', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'Close', 'id': 'Close', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'Volume', 'id': 'Volume', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                    {'name': 'Direction', 'id': 'Direction', 'type': 'text'},
                                    {'name': 'Daily Return %', 'id': 'Daily_Return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                    {'name': 'RSI', 'id': 'RSI', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                ],
                                data=df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Direction', 'Daily_Return', 'RSI']].to_dict('records'),
                                filter_action="native",
                                sort_action="native",
                                page_action="native",
                                page_current=0,
                                page_size=20,
                                style_cell={'textAlign': 'center', 'padding': '10px'},
                                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{Direction} = LONG'},
                                        'backgroundColor': '#d4edda',
                                        'color': 'black',
                                    },
                                    {
                                        'if': {'filter_query': '{Direction} = SHORT'},
                                        'backgroundColor': '#f8d7da',
                                        'color': 'black',
                                    }
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    elif active_tab == "chatbot-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🤖 TSLA Data AI Assistant", className="card-title"),
                            html.P("Ask intelligent questions about the TSLA stock data!", className="card-text"),
                            
                            # Enhanced sample questions
                            html.H6("💡 Try these sample questions:"),
                            dbc.ButtonGroup([
                                dbc.Button("Trading performance analysis", id="sample-q1", size="sm", outline=True, color="primary"),
                                dbc.Button("Technical indicator insights", id="sample-q2", size="sm", outline=True, color="info"),
                                dbc.Button("Risk assessment summary", id="sample-q3", size="sm", outline=True, color="warning"),
                                dbc.Button("Best trading opportunities", id="sample-q4", size="sm", outline=True, color="success"),
                            ], className="mb-3", style={"flexWrap": "wrap"}),
                            
                            # Chat interface
                            dbc.InputGroup([
                                dbc.Input(id="chat-input", placeholder="Ask me anything about TSLA data...", type="text"),
                                dbc.Button("Send", id="send-btn", color="primary", n_clicks=0)
                            ], className="mb-3"),
                            
                            # Chat history
                            html.Div(id="chat-history", 
                                   style={"height": "500px", "overflow-y": "auto", 
                                         "border": "1px solid #dee2e6", "padding": "15px", 
                                         "border-radius": "0.375rem"})
                        ])
                    ])
                ], width=12)
            ])
        ])

# Enhanced chatbot callbacks
@app.callback(
    Output("chat-input", "value"),
    [Input("sample-q1", "n_clicks"),
     Input("sample-q2", "n_clicks"), 
     Input("sample-q3", "n_clicks"),
     Input("sample-q4", "n_clicks")],
    prevent_initial_call=True
)
def set_sample_question(q1, q2, q3, q4):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    questions = {
        "sample-q1": "Analyze the overall trading performance and success rate of the signals",
        "sample-q2": "What do the technical indicators tell us about TSLA's trend?",
        "sample-q3": "Provide a risk assessment including volatility and drawdown analysis",
        "sample-q4": "Identify the best trading opportunities based on the data"
    }
    
    return questions.get(button_id, "")

@app.callback(
    Output("chat-history", "children"),
    [Input("send-btn", "n_clicks")],
    [State("chat-input", "value"), State("chat-history", "children")],
    prevent_initial_call=True
)
def process_chat(n_clicks, user_input, chat_history):
    if not user_input or n_clicks == 0:
        return chat_history or []
    
    # Process the question with enhanced analysis
    response = call_gemini_with_context(user_input, df, performance_metrics)
    
    # Create chat messages with better styling
    user_message = dbc.Alert([
        html.Strong("👤 You: "),
        html.Span(user_input)
    ], color="light", className="mb-2")
    
    bot_message = dbc.Alert([
        html.Strong("🤖 AI Assistant: "),
        html.Div(response, style={"whiteSpace": "pre-line"})
    ], color="info", className="mb-2")
    
    if chat_history is None:
        chat_history = []
    
    chat_history.extend([user_message, bot_message])
    
    return chat_history

def process_enhanced_question(question, df, metrics):
    """Enhanced question processing with comprehensive analysis"""
    question_lower = question.lower()
    
    try:
        if "performance" in question_lower and "trading" in question_lower:
            return f"""📊 **COMPREHENSIVE TRADING PERFORMANCE ANALYSIS**

🎯 **Signal Performance:**
• Total Signals: {metrics['Long Signals'] + metrics['Short Signals']}
• LONG Signals: {metrics['Long Signals']} 
• SHORT Signals: {metrics['Short Signals']}
• Win Rate: {metrics['Win Rate']}

💰 **Financial Performance:**
• Total Return: {metrics['Total Return']}
• Current Price: {metrics['Current Price']}
• Price Range: {metrics['Price Range']}

📈 **Risk Metrics:**
• Volatility: {metrics['Volatility (Annualized)']}
• Max Drawdown: {metrics['Max Drawdown']}

🔍 **Key Insights:**
• Signal distribution shows {('balanced' if abs(metrics['Long Signals'] - metrics['Short Signals']) < 5 else 'LONG bias' if metrics['Long Signals'] > metrics['Short Signals'] else 'SHORT bias')} approach
• {('High' if float(metrics['Win Rate'].replace('%', '')) > 60 else 'Moderate' if float(metrics['Win Rate'].replace('%', '')) > 45 else 'Low')} win rate indicates {'strong' if float(metrics['Win Rate'].replace('%', '')) > 60 else 'acceptable' if float(metrics['Win Rate'].replace('%', '')) > 45 else 'needs improvement'} signal quality
• {'High' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 40 else 'Moderate' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 25 else 'Low'} volatility suggests {'aggressive' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 40 else 'balanced' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 25 else 'conservative'} risk profile"""

        elif "technical" in question_lower and "indicator" in question_lower:
            latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 'N/A'
            latest_close = df['Close'].iloc[-1]
            latest_sma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else None
            latest_sma50 = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else None
            
            trend_analysis = "NEUTRAL"
            if latest_sma20 and latest_sma50:
                if latest_close > latest_sma20 > latest_sma50:
                    trend_analysis = "STRONG BULLISH"
                elif latest_close > latest_sma20:
                    trend_analysis = "BULLISH"
                elif latest_close < latest_sma20 < latest_sma50:
                    trend_analysis = "STRONG BEARISH"
                elif latest_close < latest_sma20:
                    trend_analysis = "BEARISH"
            
            rsi_signal = "NEUTRAL"
            if isinstance(latest_rsi, (int, float)):
                if latest_rsi > 70:
                    rsi_signal = "OVERBOUGHT"
                elif latest_rsi < 30:
                    rsi_signal = "OVERSOLD"
                elif latest_rsi > 50:
                    rsi_signal = "BULLISH"
                else:
                    rsi_signal = "BEARISH"
            
            return f"""📈 **TECHNICAL INDICATOR ANALYSIS**

🎯 **Current Trend Analysis:**
• Overall Trend: {trend_analysis}
• Current Price: {metrics['Current Price']}
• SMA 20: ${latest_sma20:.2f if latest_sma20 else 'N/A'}
• SMA 50: ${latest_sma50:.2f if latest_sma50 else 'N/A'}

📊 **RSI Analysis:**
• Current RSI: {latest_rsi:.1f if isinstance(latest_rsi, (int, float)) else 'N/A'}
• RSI Signal: {rsi_signal}
• Market Condition: {'Potentially overvalued' if rsi_signal == 'OVERBOUGHT' else 'Potentially undervalued' if rsi_signal == 'OVERSOLD' else 'Normal trading range'}

🔍 **Key Technical Insights:**
• Moving Average Convergence: {'Bullish alignment' if trend_analysis in ['BULLISH', 'STRONG BULLISH'] else 'Bearish alignment' if trend_analysis in ['BEARISH', 'STRONG BEARISH'] else 'Mixed signals'}
• Momentum: {'Strong upward momentum' if rsi_signal == 'BULLISH' and trend_analysis in ['BULLISH', 'STRONG BULLISH'] else 'Strong downward momentum' if rsi_signal == 'BEARISH' and trend_analysis in ['BEARISH', 'STRONG BEARISH'] else 'Consolidation phase'}
• Trading Recommendation: {'Consider taking profits' if rsi_signal == 'OVERBOUGHT' else 'Look for buying opportunities' if rsi_signal == 'OVERSOLD' else 'Wait for clearer signals'}"""

        elif "risk" in question_lower and ("assessment" in question_lower or "analysis" in question_lower):
            daily_returns = df['Daily_Return'].dropna()
            var_95 = daily_returns.quantile(0.05) * 100 if len(daily_returns) > 0 else 0
            sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
            
            return f"""⚠️ **COMPREHENSIVE RISK ASSESSMENT**

📊 **Volatility Analysis:**
• Annualized Volatility: {metrics['Volatility (Annualized)']}
• Daily Volatility: {daily_returns.std()*100:.2f}%
• Risk Level: {'HIGH' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 40 else 'MODERATE' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 25 else 'LOW'}

📉 **Drawdown Analysis:**
• Maximum Drawdown: {metrics['Max Drawdown']}
• Risk Rating: {'HIGH RISK' if float(metrics['Max Drawdown'].replace('%', '').replace('-', '')) > 20 else 'MODERATE RISK' if float(metrics['Max Drawdown'].replace('%', '').replace('-', '')) > 10 else 'LOW RISK'}

📈 **Performance Metrics:**
• Value at Risk (95%): {var_95:.2f}% (daily)
• Sharpe Ratio: {sharpe_ratio:.2f}
• Return vs Risk: {'Favorable' if sharpe_ratio > 1 else 'Acceptable' if sharpe_ratio > 0.5 else 'Poor'} risk-adjusted returns

🎯 **Risk Management Recommendations:**
• Position Sizing: {'Conservative (1-2% per trade)' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 40 else 'Moderate (2-3% per trade)' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 25 else 'Standard (3-5% per trade)'}
• Stop Loss: {abs(var_95)*1.5:.1f}% recommended stop loss
• Portfolio Allocation: {'Limit to 5-10% of portfolio' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 40 else 'Can allocate 10-20% of portfolio' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 25 else 'Can be core holding (20%+ allocation)'}"""

        elif "opportunities" in question_lower and "trading" in question_lower:
            # Identify best trading days
            df_signals = df[df['Direction'].notna()].copy()
            if len(df_signals) > 0:
                df_signals['Next_Return'] = df_signals['Daily_Return'].shift(-1)
                successful_longs = len(df_signals[(df_signals['Direction'] == 'LONG') & (df_signals['Next_Return'] > 0)])
                successful_shorts = len(df_signals[(df_signals['Direction'] == 'SHORT') & (df_signals['Next_Return'] < 0)])
                
                best_opportunities = df_signals.nlargest(3, 'Next_Return')[['Date', 'Direction', 'Close', 'Next_Return']]
                
                return f"""🎯 **BEST TRADING OPPORTUNITIES ANALYSIS**

📈 **Signal Success Analysis:**
• Successful LONG signals: {successful_longs}/{metrics['Long Signals']} ({(successful_longs/max(1,metrics['Long Signals'])*100):.1f}%)
• Successful SHORT signals: {successful_shorts}/{metrics['Short Signals']} ({(successful_shorts/max(1,metrics['Short Signals'])*100):.1f}%)

🏆 **Top Performing Opportunities:**
{chr(10).join([f"• {row['Date'].strftime('%Y-%m-%d')}: {row['Direction']} signal at ${row['Close']:.2f} → {row['Next_Return']*100:.2f}% return" for _, row in best_opportunities.iterrows()])}

🔍 **Pattern Analysis:**
• {'LONG signals show better performance' if successful_longs/max(1,metrics['Long Signals']) > successful_shorts/max(1,metrics['Short Signals']) else 'SHORT signals show better performance' if successful_shorts/max(1,metrics['Short Signals']) > successful_longs/max(1,metrics['Long Signals']) else 'Both signal types show similar performance'}
• Optimal trading window: {'Early morning' if df['Volume'].idxmax() < len(df)/2 else 'Throughout the day'}
• Market condition preference: {'Trending markets' if float(metrics['Win Rate'].replace('%', '')) > 55 else 'Range-bound markets'}

💡 **Strategic Recommendations:**
• Focus on: {'LONG signals' if successful_longs/max(1,metrics['Long Signals']) > 0.6 else 'SHORT signals' if successful_shorts/max(1,metrics['Short Signals']) > 0.6 else 'Both signal types with proper risk management'}
• Best entry timing: Look for signals during {'high volatility periods' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 35 else 'normal market conditions'}
• Exit strategy: {'Quick scalping (1-3 days)' if abs(df['Daily_Return'].mean()) > 0.01 else 'Medium-term holds (3-7 days)'}"""
            else:
                return "No trading signals found in the dataset to analyze opportunities."

        elif "bullish" in question_lower or "bull" in question_lower:
            bullish_days = len(df[df['Close'] > df['Open']])
            percentage = (bullish_days / len(df)) * 100
            recent_trend = "BULLISH" if df['Close'].iloc[-5:].mean() > df['Close'].iloc[-10:-5].mean() else "BEARISH"
            
            return f"""🐂 **BULLISH TREND ANALYSIS**

📊 **Bullish Performance:**
• Bullish Days: {bullish_days}/{len(df)} ({percentage:.1f}%)
• Recent 5-day trend: {recent_trend}
• Bullish momentum: {'Strong' if percentage > 60 else 'Moderate' if percentage > 45 else 'Weak'}

📈 **LONG Signal Analysis:**
• LONG signals issued: {metrics['Long Signals']}
• Success rate: {metrics['Win Rate']}
• Market bias: {'LONG-favored market' if metrics['Long Signals'] > metrics['Short Signals'] else 'Balanced market' if abs(metrics['Long Signals'] - metrics['Short Signals']) < 3 else 'SHORT-favored market'}

🎯 **Bullish Opportunities:**
• Current trend supports: {'LONG positions' if recent_trend == 'BULLISH' else 'Caution on LONG positions'}
• Best bullish setup: Look for LONG signals when RSI < 50 and price above SMA 20
• Risk level for bulls: {('LOW' if percentage > 60 else 'MODERATE' if percentage > 45 else 'HIGH')} - market shows {'consistent' if percentage > 60 else 'mixed' if percentage > 45 else 'challenging'} bullish behavior"""

        elif "bearish" in question_lower or "bear" in question_lower:
            bearish_days = len(df[df['Close'] < df['Open']])
            percentage = (bearish_days / len(df)) * 100
            recent_decline = df['Close'].iloc[-1] < df['Close'].iloc[-5]
            
            return f"""🐻 **BEARISH TREND ANALYSIS**

📉 **Bearish Performance:**
• Bearish Days: {bearish_days}/{len(df)} ({percentage:.1f}%)
• Recent decline: {'Yes' if recent_decline else 'No'} - price {'falling' if recent_decline else 'stable/rising'}
• Bearish pressure: {'High' if percentage > 60 else 'Moderate' if percentage > 45 else 'Low'}

📊 **SHORT Signal Analysis:**
• SHORT signals issued: {metrics['Short Signals']}
• Market structure: {'Bears in control' if percentage > 55 else 'Bulls vs Bears balanced' if 45 <= percentage <= 55 else 'Bulls in control'}

⚠️ **Bearish Risk Assessment:**
• Downside risk: {'HIGH' if percentage > 60 and recent_decline else 'MODERATE' if percentage > 45 else 'LOW'}
• SHORT opportunity: {'Favorable' if metrics['Short Signals'] > metrics['Long Signals'] and percentage > 50 else 'Limited'}
• Support levels: Critical to watch for breakdown below recent lows"""

        elif any(word in question_lower for word in ["volume", "liquidity"]):
            avg_volume = df['Volume'].mean()
            volume_trend = "INCREASING" if df['Volume'].iloc[-5:].mean() > df['Volume'].iloc[-10:-5].mean() else "DECREASING"
            high_volume_days = len(df[df['Volume'] > avg_volume * 1.5])
            
            return f"""📊 **VOLUME & LIQUIDITY ANALYSIS**

💹 **Volume Statistics:**
• Average Volume: {avg_volume:,.0f} shares
• Volume Trend: {volume_trend}
• High Volume Days: {high_volume_days} ({(high_volume_days/len(df)*100):.1f}%)
• Max Volume: {df['Volume'].max():,.0f} shares

🔍 **Liquidity Insights:**
• Market liquidity: {'Excellent' if avg_volume > 70000000 else 'Good' if avg_volume > 50000000 else 'Moderate'}
• Trading difficulty: {'Easy execution' if avg_volume > 70000000 else 'Normal execution' if avg_volume > 50000000 else 'May face slippage'}
• Volume-Price relationship: {'Strong correlation' if df['Volume'].corr(abs(df['Daily_Return'])) > 0.3 else 'Weak correlation'} between volume and price movement

📈 **Volume-Based Signals:**
• High volume breakouts: Look for price moves on volume > {avg_volume*1.5:,.0f}
• Volume confirmation: {volume_trend} volume trend {'supports' if volume_trend == 'INCREASING' else 'challenges'} current price action"""

        else:
            # Comprehensive summary with all key insights
            total_days = len(df)
            long_signals = metrics['Long Signals']
            short_signals = metrics['Short Signals']
            
            return f"""🚗 **COMPREHENSIVE TSLA ANALYSIS SUMMARY**

📊 **Dataset Overview:**
• Trading Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
• Total Days: {total_days}
• Current Price: {metrics['Current Price']}
• Price Range: {metrics['Price Range']}

🎯 **Trading Performance:**
• Total Return: {metrics['Total Return']}
• Win Rate: {metrics['Win Rate']}
• LONG Signals: {long_signals} ({(long_signals/total_days*100):.1f}%)
• SHORT Signals: {short_signals} ({(short_signals/total_days*100):.1f}%)

⚠️ **Risk Assessment:**
• Volatility: {metrics['Volatility (Annualized)']}
• Max Drawdown: {metrics['Max Drawdown']}
• Risk Level: {'HIGH' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 40 else 'MODERATE' if float(metrics['Volatility (Annualized)'].replace('%', '')) > 25 else 'LOW'}

💡 **Key Insights:**
• Market behavior: {'Trending' if float(metrics['Win Rate'].replace('%', '')) > 55 else 'Range-bound' if float(metrics['Win Rate'].replace('%', '')) > 45 else 'Choppy'}
• Signal quality: {'Excellent' if float(metrics['Win Rate'].replace('%', '')) > 65 else 'Good' if float(metrics['Win Rate'].replace('%', '')) > 55 else 'Average' if float(metrics['Win Rate'].replace('%', '')) > 45 else 'Poor'}
• Trading difficulty: {'Beginner-friendly' if float(metrics['Volatility (Annualized)'].replace('%', '')) < 25 else 'Intermediate' if float(metrics['Volatility (Annualized)'].replace('%', '')) < 40 else 'Advanced traders only'}

🔍 **Ask me specific questions about:**
• "Technical indicator analysis"
• "Risk assessment and management"
• "Best trading opportunities"
• "Volume and liquidity analysis"
• "Bullish or bearish trend analysis"
• "Performance metrics explanation" """

    except Exception as e:
        return f"I encountered an error analyzing your question: {str(e)}. Please try asking in a different way or use one of the suggested sample questions!"

if __name__ == '__main__':
    app.run_server(debug=True)