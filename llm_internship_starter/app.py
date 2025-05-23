import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from chatbot import gemini_bot
import os
import json
from typing import Dict, Optional, List

# Configuration
COLUMN_MAPPINGS = {
    'date': ['Date', 'date', 'TIME', 'time', 'timestamp', 'Timestamp'],
    'open': ['Open', 'open', 'OPEN'],
    'high': ['High', 'high', 'HIGH'],
    'low': ['Low', 'low', 'LOW'],
    'close': ['Close', 'close', 'CLOSE']
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
    for col_type in ['open', 'high', 'low', 'close']:
        col_name = find_column(columns, COLUMN_MAPPINGS[col_type])
        if col_name:
            ohlc_columns[col_type] = col_name
    return ohlc_columns

def parse_support_resistance(value) -> List[float]:
    """Safely parse support/resistance values."""
    if pd.isna(value) or not value:
        return []
    
    try:
        if isinstance(value, str):
            # Try to parse as JSON first, then eval as fallback
            try:
                parsed = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = eval(value)
        else:
            parsed = value
        
        return parsed if isinstance(parsed, list) else [parsed]
    except:
        return []

def create_candlestick_chart(df: pd.DataFrame, ohlc_columns: Dict[str, str]) -> go.Figure:
    """Create the main candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df[ohlc_columns['open']],
        high=df[ohlc_columns['high']],
        low=df[ohlc_columns['low']],
        close=df[ohlc_columns['close']],
        name='TSLA',
        increasing_line_color='#00D4AA',
        decreasing_line_color='#FF6B6B'
    )])
    
    return fig

def add_direction_markers(fig: go.Figure, df: pd.DataFrame, ohlc_columns: Dict[str, str]) -> None:
    """Add trade direction markers to the chart."""
    if 'direction' not in df.columns:
        return
    
    direction_config = {
        'LONG': {
            'data': df[df['direction'] == 'LONG'],
            'y_pos': lambda row: row[ohlc_columns['low']] * 0.99,
            'marker': dict(symbol='triangle-up', size=10, color='green')
        },
        'SHORT': {
            'data': df[df['direction'] == 'SHORT'],
            'y_pos': lambda row: row[ohlc_columns['high']] * 1.01,
            'marker': dict(symbol='triangle-down', size=10, color='red')
        },
        'NEUTRAL': {
            'data': df[df['direction'] == 'NEUTRAL'],
            'y_pos': lambda row: row[ohlc_columns['high']] * 1.01,
            'marker': dict(symbol='circle', size=8, color='#FFD93D')
        }
    }
    
    for direction, config in direction_config.items():
        if not config['data'].empty:
            fig.add_trace(go.Scatter(
                x=config['data']['time'],
                y=config['data'].apply(config['y_pos'], axis=1),
                mode='markers',
                marker=config['marker'],
                name=direction,
                hovertemplate=f"{direction}<br>%{{x}}<br>Price: %{{y:.2f}}<extra></extra>"
            ))

def add_support_resistance_lines(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add support and resistance lines to the chart."""
    for col_name, color, line_type in [('Support', 'green', 'min'), ('Resistance', 'red', 'max')]:
        if col_name not in df.columns:
            continue
        
        line_data = []
        for _, row in df.iterrows():
            values = parse_support_resistance(row[col_name])
            if values:
                value = min(values) if line_type == 'min' else max(values)
                line_data.append((row['time'], value))
        
        if line_data:
            x_data, y_data = zip(*line_data)
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=col_name,
                hovertemplate=f"{col_name}: %{{y:.2f}}<br>%{{x}}<extra></extra>"
            ))

def configure_chart_layout(fig: go.Figure) -> None:
    """Configure the chart layout and styling."""
    fig.update_layout(
        title={
            'text': 'TSLA Candlestick Chart',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=650,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    # Improve grid and styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

# Main application
def main():
    st.set_page_config(
        page_title="TSLA Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Debug information
    if st.checkbox("Show debug info", value=False):
        st.write("Current working directory:", os.getcwd())
    
    # Load data with error handling
    try:
        df = load_and_process_data("llm_internship_starter/data/tsla.csv")
        
        if st.checkbox("Show data info", value=False):
            st.success("Data loaded successfully")
            st.write(f"Dataset shape: {df.shape}")
            st.write("Columns available:", df.columns.tolist())
            st.write("Date range:", f"{df['time'].min()} to {df['time'].max()}")
            
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
    missing_cols = [col for col in ['open', 'high', 'low', 'close'] if col not in ohlc_columns]
    
    if missing_cols:
        st.error(f"❌ Missing required OHLC columns: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()
    
    # Main dashboard
    st.title("📈 TSLA Candlestick Dashboard + AI Chatbot")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Candlestick Chart", "🤖 Chat with Gemini"])
    
    with tab1:
        # Chart controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            show_markers = st.checkbox("Show trade markers", value=True)
        with col3:
            show_sr_lines = st.checkbox("Show S/R lines", value=True)
        
        # Create and display chart
        with st.spinner("📊 Generating chart..."):
            fig = create_candlestick_chart(df, ohlc_columns)
            
            if show_markers:
                add_direction_markers(fig, df, ohlc_columns)
            
            if show_sr_lines:
                add_support_resistance_lines(fig, df)
            
            configure_chart_layout(fig)
            
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            })
        
        # Display summary statistics
        if st.expander("📊 Market Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            latest_close = df[ohlc_columns['close']].iloc[-1]
            prev_close = df[ohlc_columns['close']].iloc[-2] if len(df) > 1 else latest_close
            change = latest_close - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            
            with col1:
                st.metric("Current Price", f"${latest_close:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
            with col2:
                st.metric("Day High", f"${df[ohlc_columns['high']].iloc[-1]:.2f}")
            with col3:
                st.metric("Day Low", f"${df[ohlc_columns['low']].iloc[-1]:.2f}")
            with col4:
                st.metric("Total Records", f"{len(df):,}")
    
    with tab2:
        st.subheader("🤖 Ask about TSLA data")
        st.markdown("Ask questions about the Tesla stock data, trends, or analysis.")
        
        # Chat interface
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What's the trend in TSLA price over the last week?",
            help="Ask about price movements, patterns, or any analysis of the TSLA data"
        )
        
        if question and question.strip():
            with st.spinner("🧠 Analyzing data and generating response..."):
                try:
                    answer = gemini_bot(question, df)
                    st.success("🎯 **AI Response:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error getting AI response: {e}")
                    st.info("💡 Try rephrasing your question or check if the chatbot service is available.")

if __name__ == "__main__":
    main()