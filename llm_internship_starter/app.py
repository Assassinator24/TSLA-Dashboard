import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chatbot import gemini_bot
import os
import json
from typing import Dict, Optional, List
import ta  # Technical Analysis library

# Configuration
COLUMN_MAPPINGS = {
    'date': ['Date', 'date', 'TIME', 'time', 'timestamp', 'Timestamp'],
    'open': ['Open', 'open', 'OPEN'],
    'high': ['High', 'high', 'HIGH'],
    'low': ['Low', 'low', 'LOW'],
    'close': ['Close', 'close', 'CLOSE'],
    'volume': ['Volume', 'volume', 'VOLUME', 'Vol', 'vol']
}

@st.cache_data
def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the CSV data with caching."""
    df = pd.read_csv(file_path)
    
    # Find date column
    date_column = find_column(df.columns, COLUMN_MAPPINGS['date'])
    if not date_column:
        raise ValueError("No date column found")
    
    # Convert to datetime and create time column
    df['time'] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['time'])
    
    # Sort by time for better performance
    df = df.sort_values('time').reset_index(drop=True)
    
    return df

def find_column(columns: List[str], possible_names: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of possibilities."""
    return next((name for name in possible_names if name in columns), None)

def find_ohlc_columns(columns: List[str]) -> Dict[str, str]:
    """Find OHLC column mappings."""
    ohlc_columns = {}
    for col_type in ['open', 'high', 'low', 'close', 'volume']:
        col_name = find_column(columns, COLUMN_MAPPINGS[col_type])
        if col_name:
            ohlc_columns[col_type] = col_name
    return ohlc_columns

def calculate_technical_indicators(df: pd.DataFrame, ohlc_columns: Dict[str, str]) -> pd.DataFrame:
    """Calculate technical indicators for the dataset."""
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df[ohlc_columns['close']].rolling(window=20).mean()
    df['SMA_50'] = df[ohlc_columns['close']].rolling(window=50).mean()
    df['SMA_200'] = df[ohlc_columns['close']].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df[ohlc_columns['close']].ewm(span=12).mean()
    df['EMA_26'] = df[ohlc_columns['close']].ewm(span=26).mean()
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.bollinger_hband(df[ohlc_columns['close']]), \
                                                      ta.volatility.bollinger_mavg(df[ohlc_columns['close']]), \
                                                      ta.volatility.bollinger_lband(df[ohlc_columns['close']])
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df[ohlc_columns['close']], window=14)
    
    # MACD
    df['MACD'] = ta.trend.macd(df[ohlc_columns['close']])
    df['MACD_signal'] = ta.trend.macd_signal(df[ohlc_columns['close']])
    df['MACD_histogram'] = ta.trend.macd_diff(df[ohlc_columns['close']])
    
    # Volume indicators
    if 'volume' in ohlc_columns:
        df['Volume_SMA'] = df[ohlc_columns['volume']].rolling(window=20).mean()
        df['OBV'] = ta.volume.on_balance_volume(df[ohlc_columns['close']], df[ohlc_columns['volume']])
    
    # Support and Resistance levels
    df['Support'] = df[ohlc_columns['low']].rolling(window=20).min()
    df['Resistance'] = df[ohlc_columns['high']].rolling(window=20).max()
    
    return df

def parse_support_resistance(value) -> List[float]:
    """Safely parse support/resistance values."""
    if pd.isna(value) or not value:
        return []
    
    try:
        if isinstance(value, str):
            try:
                parsed = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = eval(value)
        else:
            parsed = value
        
        return parsed if isinstance(parsed, list) else [parsed]
    except:
        return []

def create_advanced_chart(df: pd.DataFrame, ohlc_columns: Dict[str, str], indicators: Dict[str, bool]) -> go.Figure:
    """Create advanced candlestick chart with technical indicators."""
    
    # Create subplots with secondary y-axis for volume
    has_volume = 'volume' in ohlc_columns and indicators.get('volume', False)
    has_rsi = indicators.get('rsi', False)
    has_macd = indicators.get('macd', False)
    
    # Determine subplot configuration
    subplot_titles = ['TSLA Stock Price']
    rows = 1
    
    if has_volume:
        subplot_titles.append('Volume')
        rows += 1
    if has_rsi:
        subplot_titles.append('RSI')
        rows += 1
    if has_macd:
        subplot_titles.append('MACD')
        rows += 1
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        x_title='Date',
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        row_heights=[0.6] + [0.4/(rows-1)]*(rows-1) if rows > 1 else [1.0]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df[ohlc_columns['open']],
            high=df[ohlc_columns['high']],
            low=df[ohlc_columns['low']],
            close=df[ohlc_columns['close']],
            name='TSLA',
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF6B6B',
            increasing_fillcolor='#00D4AA',
            decreasing_fillcolor='#FF6B6B'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if indicators.get('sma', False):
        colors = ['#FFD93D', '#FF6B6B', '#00D4AA']
        periods = [20, 50, 200]
        for i, period in enumerate(periods):
            col_name = f'SMA_{period}'
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['time'],
                        y=df[col_name],
                        mode='lines',
                        name=f'SMA {period}',
                        line=dict(color=colors[i], width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
    
    # Bollinger Bands
    if indicators.get('bollinger', False):
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['BB_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(255, 107, 107, 0.3)', dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['BB_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(0, 212, 170, 0.3)', dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Support and Resistance
    if indicators.get('support_resistance', False):
        # Add dynamic support and resistance lines
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['Support'],
                mode='lines',
                name='Support',
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.6
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['Resistance'],
                mode='lines',
                name='Resistance',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # Direction markers
    if indicators.get('markers', False) and 'direction' in df.columns:
        add_direction_markers_advanced(fig, df, ohlc_columns)
    
    current_row = 2
    
    # Volume chart
    if has_volume:
        volume_colors = ['red' if close < open else 'green' 
                        for close, open in zip(df[ohlc_columns['close']], df[ohlc_columns['open']])]
        
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df[ohlc_columns['volume']],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ),
            row=current_row, col=1
        )
        
        if 'Volume_SMA' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['Volume_SMA'],
                    mode='lines',
                    name='Volume SMA',
                    line=dict(color='orange', width=2),
                    opacity=0.8
                ),
                row=current_row, col=1
            )
        
        current_row += 1
    
    # RSI
    if has_rsi:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=current_row, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=current_row, col=1)
        
        current_row += 1
    
    # MACD
    if has_macd:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['MACD_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df['MACD_histogram'],
                name='Histogram',
                marker_color=['green' if val >= 0 else 'red' for val in df['MACD_histogram']],
                opacity=0.6
            ),
            row=current_row, col=1
        )
    
    return fig

def add_direction_markers_advanced(fig: go.Figure, df: pd.DataFrame, ohlc_columns: Dict[str, str]) -> None:
    """Add enhanced trade direction markers to the chart."""
    if 'direction' not in df.columns:
        return
    
    direction_config = {
        'LONG': {
            'data': df[df['direction'] == 'LONG'],
            'y_pos': lambda row: row[ohlc_columns['low']] * 0.995,
            'marker': dict(symbol='triangle-up', size=12, color='#00D4AA', line=dict(width=2, color='white'))
        },
        'SHORT': {
            'data': df[df['direction'] == 'SHORT'],
            'y_pos': lambda row: row[ohlc_columns['high']] * 1.005,
            'marker': dict(symbol='triangle-down', size=12, color='#FF6B6B', line=dict(width=2, color='white'))
        },
        'NEUTRAL': {
            'data': df[df['direction'] == 'NEUTRAL'],
            'y_pos': lambda row: row[ohlc_columns['high']] * 1.005,
            'marker': dict(symbol='circle', size=10, color='#FFD93D', line=dict(width=2, color='white'))
        }
    }
    
    for direction, config in direction_config.items():
        if not config['data'].empty:
            fig.add_trace(go.Scatter(
                x=config['data']['time'],
                y=config['data'].apply(config['y_pos'], axis=1),
                mode='markers',
                marker=config['marker'],
                name=f'{direction} Signal',
                hovertemplate=f"<b>{direction}</b><br>%{{x}}<br>Price: %{{y:.2f}}<extra></extra>",
                showlegend=True
            ), row=1, col=1)

def configure_advanced_layout(fig: go.Figure, dark_theme: bool = True) -> None:
    """Configure advanced chart layout and styling."""
    
    theme_colors = {
        'bg_color': '#0E1117' if dark_theme else 'white',
        'grid_color': 'rgba(128,128,128,0.2)' if dark_theme else 'rgba(128,128,128,0.3)',
        'text_color': 'white' if dark_theme else 'black',
        'paper_color': '#0E1117' if dark_theme else 'white'
    }
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=850,
        template='plotly_dark' if dark_theme else 'plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.1)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        margin=dict(t=100, b=100, l=50, r=50),
        plot_bgcolor=theme_colors['bg_color'],
        paper_bgcolor=theme_colors['paper_color']
    )
    
    # Update all axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor=theme_colors['grid_color'],
        color=theme_colors['text_color']
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor=theme_colors['grid_color'],
        color=theme_colors['text_color']
    )
    
    # ADD THIS: Format specific subplot y-axes
    # Main price chart
    fig.update_yaxes(title_text="Price ($)", tickformat="$.2f", row=1, col=1)
    
    # Volume chart (if exists)
    fig.update_yaxes(title_text="Volume", tickformat="0", row=2, col=1)
    
    # RSI chart (if exists) 
    fig.update_yaxes(title_text="RSI", tickformat=".0f", range=[0, 100], row=3, col=1)
    
    # MACD chart (if exists)
    fig.update_yaxes(title_text="MACD", tickformat=".3f", row=4, col=1)

def create_market_overview(df: pd.DataFrame, ohlc_columns: Dict[str, str]) -> None:
    """Create market overview section with key metrics."""
    
    latest_data = df.iloc[-1]
    prev_data = df.iloc[-2] if len(df) > 1 else latest_data
    
    # Calculate metrics
    current_price = latest_data[ohlc_columns['close']]
    prev_close = prev_data[ohlc_columns['close']]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    
    # Volume metrics
    volume_available = 'volume' in ohlc_columns
    avg_volume = df[ohlc_columns['volume']].tail(20).mean() if volume_available else 0
    current_volume = latest_data[ohlc_columns['volume']] if volume_available else 0
    
    # Technical levels
    high_52w = df[ohlc_columns['high']].tail(252).max() if len(df) >= 252 else df[ohlc_columns['high']].max()
    low_52w = df[ohlc_columns['low']].tail(252).min() if len(df) >= 252 else df[ohlc_columns['low']].min()
    
    st.markdown("### 📊 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price", 
            f"${current_price:.2f}", 
            f"{change:+.2f} ({change_pct:+.1f}%)",
            delta_color="inverse"
        )
    
    with col2:
        st.metric("Day High", f"${latest_data[ohlc_columns['high']]:.2f}")
    
    with col3:
        st.metric("Day Low", f"${latest_data[ohlc_columns['low']]:.2f}")
    
    with col4:
        if volume_available:
            volume_change = ((current_volume / avg_volume) - 1) * 100 if avg_volume > 0 else 0
            st.metric(
                "Volume", 
                f"{current_volume/1e6:.1f}M", 
                f"{volume_change:+.1f}% vs avg"
            )
        else:
            st.metric("Records", f"{len(df):,}")
    
    with col5: 
        st.metric("52W Range", f"${low_52w:.2f} - ${high_52w:.2f}")

# Main application
def main():
    st.set_page_config(
        page_title="TSLA Advanced Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Chart Controls")
        
        # Theme selection
        dark_theme = st.checkbox("🌙 Dark Theme", value=True)
        
        st.subheader("Technical Indicators")
        indicators = {
            'sma': st.checkbox("📈 Moving Averages", value=True),
            'bollinger': st.checkbox("📊 Bollinger Bands", value=True),
            'volume': st.checkbox("📦 Volume", value=True),
            'rsi': st.checkbox("⚡ RSI", value=True),
            'macd': st.checkbox("🌊 MACD", value=True),
            'markers': st.checkbox("🎯 Trade Signals", value=True),
            'support_resistance': st.checkbox("🏔️ Support/Resistance", value=True)
        }
        
        st.subheader("Display Options")
        timeframe = st.selectbox("⏰ Timeframe", ["1D", "5D", "1M", "3M", "6M", "1Y"], index=2)
        
        # Real-time simulation
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Debug information
    if st.checkbox("🔍 Show debug info", value=False):
        st.info(f"Current working directory: {os.getcwd()}")
    
    # Load data with error handling
    try:
        df = load_and_process_data("E:/LLM intern/llm_internship_starter/data/tsla.csv")
        
        # Calculate technical indicators
        with st.spinner("🧮 Calculating technical indicators..."):
            df = calculate_technical_indicators(df, find_ohlc_columns(df.columns))
        
        if st.checkbox("📋 Show data info", value=False):
            st.success("✅ Data loaded successfully")
            st.info(f"Dataset shape: {df.shape}")
            st.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
            with st.expander("📊 Column Information"):
                st.write("Available columns:", df.columns.tolist())
                
    except FileNotFoundError as e:
        st.error("❌ CSV file not found. Please ensure 'data/tsla.csv' exists.")
        st.write(f"Error details: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Data processing error: {e}")
        st.write("Available columns:", df.columns.tolist() if 'df' in locals() else "Could not load data")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {e}")
        st.stop()
    
    # Find OHLC columns
    ohlc_columns = find_ohlc_columns(df.columns)
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in ohlc_columns]
    
    if missing_cols:
        st.error(f"❌ Missing required OHLC columns: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()
    
    # Main dashboard
    st.title("📈 TSLA Advanced Trading Dashboard")
    # st.markdown("*Professional-grade technical analysis with real-time insights*")
    
    # Market overview
    create_market_overview(df, ohlc_columns)
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Advanced Charts", "🤖 AI Analysis", "📈 Technical Analysis"])
    
    with tab1:
        # Create and display advanced chart
        with st.spinner("🚀 Generating advanced chart..."):
            fig = create_advanced_chart(df, ohlc_columns, indicators)
            configure_advanced_layout(fig, dark_theme)
            
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'tsla_chart',
                    'height': 800,
                    'width': 1200,
                    'scale': 1
                }
            })
        
        # Alert system
        with st.expander("🚨 Price Alerts & Signals", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔔 Active Alerts")
                current_price = df[ohlc_columns['close']].iloc[-1]
                rsi_current = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                
                if rsi_current > 70:
                    st.warning(f"⚠️ RSI Overbought: {rsi_current:.1f}")
                elif rsi_current < 30:
                    st.warning(f"⚠️ RSI Oversold: {rsi_current:.1f}")
                else:
                    st.success("✅ RSI in normal range")
            
            with col2:
                st.subheader("📊 Signal Summary")
                signals = {
                    "Bullish": len(df[df.get('direction', pd.Series()) == 'LONG'].tail(20)),
                    "Bearish": len(df[df.get('direction', pd.Series()) == 'SHORT'].tail(20)),
                    "Neutral": len(df[df.get('direction', pd.Series()) == 'NEUTRAL'].tail(20))
                }
                
                for signal, count in signals.items():
                    st.metric(f"{signal} Signals (20d)", count)
    
    with tab2:
        st.subheader("🤖 AI-Powered Market Analysis")
        st.markdown("Ask sophisticated questions about TSLA's technical patterns, market trends, and trading opportunities.")
        
        # Enhanced chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_area(
                "💬 Enter your analysis question:",
                placeholder="e.g., What do the current technical indicators suggest for TSLA's next move? Are there any bullish or bearish patterns forming?",
                height=100,
                help="Ask about technical analysis, price predictions, volume analysis, or market sentiment"
            )
        
        with col2:
            st.markdown("#### 💡 Sample Questions")
            sample_questions = [
                "Analyze current RSI levels",
                "MACD signal interpretation",
                "Bollinger Bands squeeze analysis",
                "Volume trend analysis",
                "Support/resistance levels"
            ]
            
            for i, sample in enumerate(sample_questions):
                if st.button(sample, key=f"sample_{i}"):
                    question = sample
        
        if question and question.strip():
            with st.spinner("🧠 Analyzing market data with AI..."):
                try:
                    def safe_format(df, column, format_spec, default='N/A'):
                        """Safely format a column value with fallback."""
                        if column in df.columns:
                            value = df[column].iloc[-1]
                            return f"{value:{format_spec}}"
                        return default

                    # Enhanced context for AI
                    market_context = f"""
                    Current TSLA Data Summary:
                    - Latest Price: ${df[ohlc_columns['close']].iloc[-1]:.2f}
                    - RSI: {safe_format(df, 'RSI', '.1f')}
                    - MACD: {safe_format(df, 'MACD', '.3f')}
                    - Volume: {safe_format(df, ohlc_columns.get('volume', ''), ',.0f') if 'volume' in ohlc_columns else 'N/A'}
                    """
                    
                    enhanced_question = f"{market_context}\n\nQuestion: {question}"
                    answer = gemini_bot(enhanced_question, df)
                    
                    st.success("🎯 **AI Market Analysis:**")
                    st.markdown(answer)
                    
                except Exception as e:
                    st.error(f"❌ Error getting AI response: {e}")
                    st.info("💡 Try rephrasing your question or check if the chatbot service is available.")
    
    with tab3:
        st.subheader("📈 Technical Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Current Indicators")
            if 'RSI' in df.columns:
                current_rsi = df['RSI'].iloc[-1]
                rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Normal"
                st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_status)
            
            if 'MACD' in df.columns:
                current_macd = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                macd_trend = "Bullish" if current_macd > macd_signal else "Bearish"
                st.metric("MACD", f"{current_macd:.3f}", macd_trend)
        
        with col2:
            st.markdown("#### 📋 Trading Summary")
            
            # Price action summary
            latest_price = df[ohlc_columns['close']].iloc[-1]
            sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else latest_price
            
            trend = "Above" if latest_price > sma_20 else "Below"
            st.metric("Price vs SMA(20)", trend, f"${abs(latest_price - sma_20):.2f}")
            
            # Volatility
            volatility = df[ohlc_columns['close']].pct_change().std() * 100
            st.metric("Daily Volatility", f"{volatility:.2f}%")

if __name__ == "__main__":
    main()