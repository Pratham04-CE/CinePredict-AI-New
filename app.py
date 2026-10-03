import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json

# Safe Optional Imports with Fallback Guards
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False


# ==========================================
# PAGE CONFIGURATION & CINEMATIC DARK THEME
# ==========================================
st.set_page_config(
    page_title="CinePredict AI - Movie Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Cinematic Dark-Mode CSS with Hidden Sidebar
st.markdown("""
<style>
    /* Completely Hide Sidebar Navigation & Controls */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        width: 0px !important;
    }

    /* Theme-Aware Glassmorphism Card Styling */
    .glass-card {
        background: var(--background-color); 
        border: 1px solid rgba(150, 150, 150, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .glow-gold {
        border-color: rgba(255, 215, 0, 0.4) !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.1) !important;
    }

    .glow-red {
        border-color: rgba(229, 9, 20, 0.4) !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.1) !important;
    }

    /* Custom Metric Badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 10px;
    }
    
    .badge-blockbuster {
        background: linear-gradient(45deg, #FFD700, #FF8C00);
        color: #000000;
    }
    
    .badge-hit {
        background: linear-gradient(45deg, #00E676, #00B0FF);
        color: #000000;
    }
    
    .badge-average {
        background: linear-gradient(45deg, #FF9100, #FFEA00);
        color: #000000;
    }
    
    .badge-flop {
        background: linear-gradient(45deg, #FF1744, #D50000);
        color: #FFFFFF;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & SCHEMA DEFINITIONS
# ==========================================
CSV_FILE_PATH = "movie_data.csv"
DATASET_COLUMNS = [
    "Movie Name", "Industry", "Genre", "Lead Actors", "Co-Actors", 
    "Budget", "Description", "Festival Release", "Hype Score", "Predicted Box Office Collection"
]

INDUSTRIES = ["Bollywood", "South Indian", "Hollywood", "Dhollywood", "Pan-India"]
GENRES = ["Action", "Drama", "Comedy", "Sci-Fi", "Thriller", "Horror", "Romance", "Fantasy", "Animation"]

# ==========================================
# DATA LOADING & PERSISTENCE ENGINE
# ==========================================
import time

@st.cache_data(ttl=2)
def load_data():
    """
    Loads movie dataset from movie_data.csv safely.
    Ensures missing columns are filled, types are normalized, and returns a clean DataFrame.
    Does NOT overwrite movie_data.csv on read errors to prevent data loss.
    """
    if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(CSV_FILE_PATH, encoding=enc)
                for col in DATASET_COLUMNS:
                    if col not in df.columns:
                        df[col] = None
                df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce').fillna(0.0)
                df['Hype Score'] = pd.to_numeric(df['Hype Score'], errors='coerce').fillna(5.0)
                df['Predicted Box Office Collection'] = pd.to_numeric(df['Predicted Box Office Collection'], errors='coerce').fillna(0.0)
                return df[DATASET_COLUMNS]
            except Exception:
                continue

    return pd.DataFrame(columns=DATASET_COLUMNS)

def save_data(df):
    """
    Saves DataFrame to movie_data.csv atomically using a temporary file 
    and retry logic to prevent file corruption or truncation on Windows.
    """
    try:
        st.cache_data.clear()
    except Exception:
        pass

    tmp_path = CSV_FILE_PATH + ".tmp"
    
    for attempt in range(10):
        try:
            df.to_csv(tmp_path, index=False, encoding='utf-8')
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, CSV_FILE_PATH)
                return True
        except (PermissionError, OSError):
            time.sleep(0.15)
            try:
                df.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')
                return True
            except Exception:
                time.sleep(0.15)
        except Exception:
            time.sleep(0.15)
            
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
            
    return False

# ==========================================
# GEMINI GENERATIVE AI STORYLINE HYPE SCORER
# ==========================================
def analyze_storyline_with_gemini(movie_name, industry, genre, budget, lead_actors, description, api_key):
    """
    Sends storyline & movie parameters to Google Gemini API to produce an AI Hype Score (1-10)
    and short narrative reasoning. Fallback algorithm executes if key is missing or SDK not installed.
    """
    fallback_used = False
    ai_reasoning = ""
    hype_score = 7.0

    if HAS_GEMINI_SDK and api_key and len(api_key.strip()) > 10:
        try:
            genai.configure(api_key=api_key.strip())
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            You are an expert film industry box office analyst and trade expert.
            Analyze the following movie concept and provide a commercial Hype Score from 1.0 to 10.0 (where 10.0 is massive global blockbuster potential).
            
            Movie Name: {movie_name}
            Industry: {industry}
            Genre: {genre}
            Budget: ₹{budget} Crores
            Lead Actors/Star Power: {lead_actors}
            Storyline/Description: {description}
            
            Respond strictly in valid JSON format with two keys:
            "hype_score": (float between 1.0 and 10.0)
            "reasoning": (a concise 2-sentence commercial potential analysis)
            """
            
            response = model.generate_content(prompt)
            text_resp = response.text.strip()
            
            if "```json" in text_resp:
                text_resp = text_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in text_resp:
                text_resp = text_resp.split("```")[1].split("```")[0].strip()
                
            data = json.loads(text_resp)
            hype_score = float(data.get("hype_score", 7.5))
            hype_score = max(1.0, min(10.0, hype_score))
            ai_reasoning = data.get("reasoning", "Strong storyline narrative with solid commercial hook.")
            return hype_score, ai_reasoning, False

        except Exception as e:
            fallback_used = True
            ai_reasoning = f"AI API unavailable ({str(e)[:30]}...). Used smart algorithmic hype evaluation."
    else:
        fallback_used = True

    # Algorithmic Fallback Calculation
    if fallback_used:
        base_score = 6.0
        if genre in ["Action", "Sci-Fi", "Fantasy"]:
            base_score += 1.2
        elif genre in ["Comedy", "Thriller", "Horror"]:
            base_score += 0.8
        
        if "+" in lead_actors or len(lead_actors.split(",")) > 1:
            base_score += 1.0
            
        if budget > 200:
            base_score += 1.0
        elif budget > 50:
            base_score += 0.5
            
        if len(description) > 100:
            base_score += 0.5

        high_hype_keywords = ["hero", "villain", "epic", "universe", "war", "battle", "revenge", "empire", "spectacle", "multiverse"]
        if any(kw in description.lower() for kw in high_hype_keywords):
            base_score += 0.5
            
        hype_score = round(max(1.0, min(10.0, base_score)), 1)
        if not ai_reasoning:
            ai_reasoning = f"Evaluated based on {industry} market trends, {genre} genre appeal, and star cast potential."

    return hype_score, ai_reasoning, fallback_used

# ==========================================
# MACHINE LEARNING ENGINE (DYNAMICS & FALLBACK)
# ==========================================
def train_and_predict(df, new_movie_dict):
    """
    Trains ML model on existing entries stored in movie_data.csv.
    When dataset has < 5 entries or scikit-learn is absent, uses dynamic mathematical regression formula.
    """
    budget = new_movie_dict['Budget']
    hype = new_movie_dict['Hype Score']

    if HAS_SKLEARN and not df.empty and len(df) >= 5:
        try:
            df_train = df.dropna(subset=['Budget', 'Hype Score', 'Predicted Box Office Collection']).copy()
            if len(df_train) >= 5:
                le_industry = LabelEncoder()
                le_genre = LabelEncoder()
                le_festival = LabelEncoder()

                all_industries = list(set(df_train['Industry'].astype(str).unique()).union({new_movie_dict['Industry']}))
                all_genres = list(set(df_train['Genre'].astype(str).unique()).union({new_movie_dict['Genre']}))
                all_festivals = list(set(df_train['Festival Release'].astype(str).unique()).union({new_movie_dict['Festival Release']}))

                le_industry.fit(all_industries)
                le_genre.fit(all_genres)
                le_festival.fit(all_festivals)

                df_train['Industry_Enc'] = le_industry.transform(df_train['Industry'].astype(str))
                df_train['Genre_Enc'] = le_genre.transform(df_train['Genre'].astype(str))
                df_train['Festival_Enc'] = le_festival.transform(df_train['Festival Release'].astype(str))
                df_train['MultiStarer'] = df_train['Lead Actors'].astype(str).apply(lambda x: 1 if '+' in x else 0)
                
                X = df_train[['Industry_Enc', 'Genre_Enc', 'Budget', 'Festival_Enc', 'Hype Score', 'MultiStarer']]
                y = df_train['Predicted Box Office Collection']

                model = LinearRegression()
                model.fit(X, y)

                new_ind_enc = le_industry.transform([new_movie_dict['Industry']])[0]
                new_gen_enc = le_genre.transform([new_movie_dict['Genre']])[0]
                new_fest_enc = le_festival.transform([new_movie_dict['Festival Release']])[0]
                new_multistar = 1 if '+' in new_movie_dict['Lead Actors'] else 0

                X_new = np.array([[new_ind_enc, new_gen_enc, budget, new_fest_enc, hype, new_multistar]])
                prediction = float(model.predict(X_new)[0])
                min_reasonable = budget * 0.1
                return round(max(min_reasonable, prediction), 2)
        except Exception:
            pass

    multiplier = 1.1 + (hype - 5.0) * 0.35
    if new_movie_dict.get('Festival Release') == 'Yes':
        multiplier += 0.45
    if '+' in new_movie_dict.get('Lead Actors', ''):
        multiplier += 0.35
    
    ind = new_movie_dict.get('Industry', '')
    if ind == 'Pan-India':
        multiplier += 0.3
    elif ind == 'Hollywood':
        multiplier += 0.4
    
    prediction = budget * multiplier
    return round(max(budget * 0.2, prediction), 2)

# ==========================================
# CURRENCY FORMATTER & CLASSIFICATION LOGIC
# ==========================================
def format_currency(val_in_cr):
    """Formats numeric amount in Crores to Lakhs (₹ Lakh) or Crores (₹ Cr)."""
    if val_in_cr < 1.0:
        lakhs = val_in_cr * 100.0
        return f"₹{lakhs:,.1f} Lakh"
    else:
        return f"₹{val_in_cr:,.2f} Cr"

def classify_movie_performance(predicted_collection, budget):
    roi = predicted_collection / budget if budget > 0 else 1.0
    
    if roi >= 2.5:
        return "🔥 All-Time Blockbuster", "badge-blockbuster", roi
    elif roi >= 1.3:
        return "✅ Hit Movie (Profitable)", "badge-hit", roi
    elif roi >= 1.0:
        return "⚠️ Average / Moderate Return", "badge-average", roi
    else:
        return "❌ Flop / Loss Risk", "badge-flop", roi

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
def main():
    if 'movie_df' not in st.session_state:
        st.session_state['movie_df'] = load_data()
    else:
        disk_df = load_data()
        if not disk_df.equals(st.session_state['movie_df']):
            st.session_state['movie_df'] = disk_df
            
    df = st.session_state['movie_df']
    # Check environment variable first, then Streamlit secrets
    user_api_key = os.getenv("GEMINI_API_KEY")
    if not user_api_key:
        try:
            user_api_key = st.secrets.get("GEMINI_API_KEY", "")
        except FileNotFoundError:
            pass


    # --- MAIN HEADER ---
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 25px 0;">
            <h1 style="font-size: 2.8rem; font-weight: 900; letter-spacing: -1px; margin-bottom: 5px;">
                🎬 CinePredict AI
            </h1>
            <p style="font-size: 1.2rem; color: #A0AEC0;">
                Pan-India Box Office Collection Predictor & Dynamic Commercial Analytics Studio
            </p>
        </div>
    """, unsafe_allow_html=True)


    # Navigation Tabs
    tab1, tab2 = st.tabs(["🔮 Prediction Studio", "📊 Analytics & Insights Dashboard"])

    # ==========================================
    # TAB 1: PREDICTION STUDIO
    # ==========================================
    with tab1:
        st.markdown('<div class="glass-card glow-gold">', unsafe_allow_html=True)
        st.subheader("🎬 Enter Movie Concept Details")
        st.write("Input film parameters to calculate hype score, predict collection, and store entry into `movie_data.csv`.")

        col1, col2 = st.columns(2)

        with col1:
            movie_name = st.text_input("Movie Title", value="Stree 2", placeholder="e.g. War 2, Pushpa 2")
            industry = st.selectbox("Film Industry / Market", INDUSTRIES, index=0)
            genre = st.selectbox("Movie Genre", GENRES, index=0)
            
            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                budget_val = st.number_input("Production Budget Amount", min_value=0.01, max_value=50000.0, value=50.0, step=1.0)
            with b_col2:
                budget_unit = st.selectbox("Unit", ["Crores (Cr)", "Lakhs (Lakh)"], index=0)

            # Convert budget to Crores for standardized ML calculation
            budget = budget_val / 100.0 if "Lakh" in budget_unit else float(budget_val)

        with col2:
            lead_actors = st.text_input("Lead Star / Cast", value="Rajkummar Rao + Shraddha Kapoor", help="Use '+' for multi-starer casts (e.g., Yash + Sanjay Dutt)")
            co_actors = st.text_input("Supporting Cast / Director", value="Pankaj Tripathi, directed by Amar Kaushik")
            festival_release = st.radio("Festival / Holiday Release Window?", ["Yes", "No"], index=0, horizontal=True)

        description = st.text_area(
            "Storyline & Plot Summary (Passes to Gemini Generative AI)",
            value="Chanderi is haunted by a new headless demon named Sarkata, and Vicky along with his friends must unite to save their town.",
            height=110,
            help="Detailed plot descriptions yield higher accuracy from the Gemini AI Hype Scorer."
        )

        predict_btn = st.button("🚀 Predict Box Office & Save to CSV", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            if not movie_name.strip() or not description.strip():
                st.warning("Please provide a valid Movie Title and Storyline Description.")
            else:
                with st.spinner("🧠 Analyzing storyline & calculating box office prediction..."):
                    hype_score, ai_reasoning, fallback_used = analyze_storyline_with_gemini(
                        movie_name, industry, genre, budget, lead_actors, description, user_api_key
                    )

                    new_movie_dict = {
                        "Movie Name": movie_name.strip(),
                        "Industry": industry,
                        "Genre": genre,
                        "Lead Actors": lead_actors.strip(),
                        "Co-Actors": co_actors.strip(),
                        "Budget": float(budget),
                        "Description": description.strip(),
                        "Festival Release": festival_release,
                        "Hype Score": float(hype_score),
                    }

                    predicted_box_office = train_and_predict(df, new_movie_dict)
                    new_movie_dict["Predicted Box Office Collection"] = predicted_box_office

                    status_text, status_class, roi = classify_movie_performance(predicted_box_office, budget)

                    new_df = pd.concat([st.session_state['movie_df'], pd.DataFrame([new_movie_dict])], ignore_index=True)
                    saved_ok = save_data(new_df)
                    if saved_ok:
                        st.session_state['movie_df'] = new_df
                        df = new_df
                    else:
                        df = st.session_state['movie_df']

                # --- DISPLAY RESULTS CARD ---
                st.markdown("---")
                if not saved_ok:
                    st.warning("⚠️ `movie_data.csv` is currently open or locked by another editor program. Please close `movie_data.csv` and try again.")
                st.markdown('<div class="glass-card glow-red">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Prediction Results for **{movie_name}**")

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                with res_col1:
                    st.metric(label="Estimated Box Office", value=format_currency(predicted_box_office))
                
                with res_col2:
                    st.metric(label="Production Budget", value=format_currency(budget))

                with res_col3:
                    st.metric(label="Gemini Hype Score", value=f"⚡ {hype_score}/10")

                with res_col4:
                    st.metric(label="Projected ROI Multiplier", value=f"{roi:.2f}x")

                st.markdown(f"""
                    <div style="text-align: center; margin: 15px 0;">
                        <span class="status-badge {status_class}">{status_text}</span>
                    </div>
                """, unsafe_allow_html=True)

                # st.info(f"**🤖 Gemini AI Commercial Insight:** {ai_reasoning}")
                #
                # if fallback_used:
                #     st.caption("ℹ️ *Note: Computed using CinePredict fallback hype evaluator.*")
                # else:
                #     st.caption("✨ *Storyline hype score generated live via Google Gemini Generative AI API.*")

                st.success(f"✅ Entry saved to `movie_data.csv`! Total records in dataset: **{len(st.session_state['movie_df'])}**.")
                st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 2: ANALYTICS & INSIGHTS DASHBOARD
    # ==========================================
    with tab2:
        st.subheader("📊 Interactive Box Office & Market Analytics")
        st.write("Explore industry trends, top grossers, budget vs collection patterns, and hype impact.")
        df = st.session_state['movie_df']

        if df.empty or len(df) == 0:
            st.info("📥 **`movie_data.csv` is currently empty.** Go to the **Prediction Studio** tab and submit a movie to start generating interactive analytics!")
        else:
            # Overview Metrics Row
            st.markdown('<div class="glass-card glow-gold">', unsafe_allow_html=True)
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Total Movies Stored", len(df))
            with m_col2:
                st.metric("Industries Covered", df['Industry'].dropna().nunique())
            with m_col3:
                st.metric("Total Box Office Tracked", format_currency(df['Predicted Box Office Collection'].sum()))
            st.markdown('</div>', unsafe_allow_html=True)

            chart_option = st.selectbox(
                "Select Analytics Visualization Chart",
                [
                    "1. 🏢 Industry-wise Average Collection",
                    "2. 🏆 Top Highest Grossing Movies",
                    "3. 💰 Budget vs Collection Scatter Analysis",
                    "4. ⚡ Hype Score Impact on Collection",
                    "5. 🎭 Genre-wise Movie Distribution"
                ]
            )

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            # Chart 1: Industry-wise Average Collection
            if "1." in chart_option:
                ind_df = df.groupby('Industry')[['Predicted Box Office Collection', 'Budget']].mean().reset_index()
                if HAS_PLOTLY and len(ind_df) > 0:
                    fig = px.bar(
                        ind_df, 
                        x='Industry', 
                        y=['Predicted Box Office Collection', 'Budget'],
                        barmode='group',
                        title='Industry-wise Average Box Office Collection vs Average Budget (₹ Crores)',
                        labels={'value': '₹ Crores', 'variable': 'Metric'},
                        color_discrete_sequence=['#FFD700', '#E50914'],
                        template='plotly_dark'
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#F0F2F5"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(ind_df.set_index('Industry'))

            # Chart 2: Top Highest Grossing Movies
            elif "2." in chart_option:
                top_df = df.sort_values(by='Predicted Box Office Collection', ascending=False).head(10)
                if HAS_PLOTLY and len(top_df) > 0:
                    fig = px.bar(
                        top_df,
                        x='Predicted Box Office Collection',
                        y='Movie Name',
                        orientation='h',
                        color='Industry',
                        text='Predicted Box Office Collection',
                        title='Highest Grossing Movies Stored in Database (₹ Crores)',
                        template='plotly_dark',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#F0F2F5")
                    )
                    fig.update_traces(texttemplate='₹%{text:,.0f} Cr', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(top_df.set_index('Movie Name')['Predicted Box Office Collection'])

            # Chart 3: Budget vs Collection Scatter Analysis
            elif "3." in chart_option:
                if HAS_PLOTLY and len(df) > 0:
                    fig = px.scatter(
                        df,
                        x='Budget',
                        y='Predicted Box Office Collection',
                        color='Industry',
                        size='Hype Score',
                        hover_name='Movie Name',
                        hover_data=['Genre', 'Lead Actors', 'Festival Release'],
                        title='Budget vs Box Office Collection (Bubble size = Hype Score)',
                        template='plotly_dark'
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#F0F2F5"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.scatter_chart(df, x='Budget', y='Predicted Box Office Collection', color='Industry')

            # Chart 4: Hype Score Impact on Collection
            elif "4." in chart_option:
                if HAS_PLOTLY and len(df) > 0:
                    fig = px.scatter(
                        df,
                        x='Hype Score',
                        y='Predicted Box Office Collection',
                        color='Festival Release',
                        size='Budget',
                        hover_name='Movie Name',
                        title='AI Hype Score vs Box Office Collection (Colored by Festival Release)',
                        labels={'Hype Score': 'Gemini AI Hype Score (1-10)', 'Predicted Box Office Collection': 'Box Office (₹ Cr)'},
                        template='plotly_dark',
                        color_discrete_sequence=['#00E676', '#FF1744']
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#F0F2F5"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.scatter_chart(df, x='Hype Score', y='Predicted Box Office Collection', color='Festival Release')

            # Chart 5: Genre-wise Movie Distribution
            elif "5." in chart_option:
                genre_df = df['Genre'].value_counts().reset_index()
                genre_df.columns = ['Genre', 'Count']
                if HAS_PLOTLY and len(genre_df) > 0:
                    fig = px.pie(
                        genre_df,
                        values='Count',
                        names='Genre',
                        title='Genre Share & Distribution Across Stored Movies',
                        hole=0.4,
                        template='plotly_dark',
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#F0F2F5"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(genre_df.set_index('Genre'))

            st.markdown('</div>', unsafe_allow_html=True)

            # Interactive Data Explorer Table
            st.markdown("---")
            st.subheader("📋 Explore Stored Dataset (`movie_data.csv`)")
            search_query = st.text_input("🔍 Search Movie, Actor, or Industry", "")
            
            display_df = df.copy()
            if search_query:
                display_df = display_df[
                    display_df['Movie Name'].astype(str).str.contains(search_query, case=False, na=False) |
                    display_df['Lead Actors'].astype(str).str.contains(search_query, case=False, na=False) |
                    display_df['Industry'].astype(str).str.contains(search_query, case=False, na=False) |
                    display_df['Genre'].astype(str).str.contains(search_query, case=False, na=False)
                ]

            st.dataframe(
                display_df[['Movie Name', 'Industry', 'Genre', 'Budget', 'Hype Score', 'Predicted Box Office Collection', 'Festival Release', 'Lead Actors']],
                use_container_width=True,
                height=350
            )

            if st.button("🗑️ Reset Database to Blank", help="Clears all records in movie_data.csv"):
                empty_df = pd.DataFrame(columns=DATASET_COLUMNS)
                st.session_state['movie_df'] = empty_df
                save_data(empty_df)
                st.rerun()

if __name__ == "__main__":
    main()
