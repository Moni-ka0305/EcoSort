import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import random

# Page config
st.set_page_config(
    page_title="EcoSort - Smart Recycling Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Complete CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=DM+Sans:wght@400;500;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #f8faf9;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding: 1.5rem 2.5rem;
        max-width: 1400px;
    }
    
    /* Clean Navigation Bar */
    .nav-bar {
        background: white;
        padding: 1.5rem 2.5rem;
        margin: -1.5rem -2.5rem 2rem -2.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-bottom: 1px solid #e8ebe9;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: #16a34a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    
    /* Hero Section - Minimalist */
    .hero-minimal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 4rem 3rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-minimal::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        margin: 0 0 1rem 0;
        position: relative;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-weight: 500;
        position: relative;
        max-width: 600px;
    }
    
    /* Clean Card Design */
    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border-color: #10b981;
        transform: translateY(-2px);
    }
    
    /* Result Box - Clean Design */
    .result-box {
        background: white;
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .result-category {
        font-size: 3rem;
        font-weight: 900;
        color: #10b981;
        text-transform: capitalize;
        letter-spacing: -1px;
        margin: 0;
    }
    
    .confidence {
        color: #64748b;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Stats - Minimal Design */
    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: #10b981;
        margin: 0;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info Box - Clean Design */
    .info-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-heading {
        font-weight: 700;
        font-size: 1.1rem;
        color: #92400e;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-content {
        font-size: 1rem;
        line-height: 1.6;
        color: #78350f;
        margin: 0;
        font-weight: 500;
    }
    
    /* Impact Box - Clean Design */
    .impact-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .impact-fact {
        font-size: 1rem;
        line-height: 1.7;
        color: #1e3a8a;
        font-weight: 600;
        margin: 0;
    }
    
    /* Sidebar - Minimal Design */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #10b981;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Buttons - Clean Modern Style */
    .stButton>button {
        width: 100%;
        background: #10b981;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.875rem 1.5rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        background: #059669;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Upload - Clean Design */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #d1d5db;
        border-radius: 16px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #10b981;
        background: #f0fdf4;
    }
    
    /* Section Titles - Clean Design */
    .section-title {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
    }
    
    /* Feature Boxes - Clean Design */
    .feature {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature:hover {
        border-color: #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    .feature-title {
        color: #10b981;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0 0 0.5rem 0;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #64748b;
        margin: 0;
    }
    
    /* Metrics - Clean Design */
    .stMetric {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .stMetric label {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-weight: 800 !important;
    }
    
    /* Progress bar - Clean Design */
    .stProgress > div > div {
        background: #10b981;
        height: 8px;
        border-radius: 8px;
    }
    
    .stProgress > div {
        background: #e5e7eb;
        border-radius: 8px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #10b981 !important;
        border-right-color: #10b981 !important;
    }
    
    /* Image Display */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        font-weight: 600;
        color: #1f2937 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #10b981;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 0.25rem;
    }
    
    /* Alert Box */
    .alert-success {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

#New # WINNING BANNER - Add after CSS, before session state
st.markdown("""
<div style="background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
            padding: 1rem; border-radius: 10px; text-align: center; 
            margin-bottom: 2rem; color: white; border: 3px solid gold;">
    <h3>CODE4EARTH</h3>
    <p>AI-Powered Waste Classification + Behavioral Gamification</p>
</div>
""", unsafe_allow_html=True)

# New # Initialize session state
if 'eco_score' not in st.session_state:
    st.session_state.eco_score = 0
if 'total_items' not in st.session_state:
    st.session_state.total_items = 0
if 'energy_saved' not in st.session_state:
    st.session_state.energy_saved = 0
if 'co2_prevented' not in st.session_state:
    st.session_state.co2_prevented = 0
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    
# Eco-facts database
ECO_FACTS = {
    "cardboard": [
        {"fact": "📦 Recycling 1 ton saves 46 gallons of oil and prevents 3.3kg CO₂", "energy": 24, "co2": 3.3},
        {"fact": "🌳 Recycled cardboard uses 75% less energy than virgin materials", "energy": 18, "co2": 2.1}
    ],
    "glass": [
        {"fact": "✨ Recycling glass saves 30% energy vs making new glass", "energy": 30, "co2": 4.2},
        {"fact": "♾️ Glass can be recycled infinitely without quality loss", "energy": 28, "co2": 3.8}
    ],
    "metal": [
        {"fact": "📺 One aluminum can recycled runs a TV for 3 hours", "energy": 45, "co2": 6.5},
        {"fact": "⚡ Recycled aluminum uses 95% less energy than new production", "energy": 50, "co2": 7.2}
    ],
    "paper": [
        {"fact": "🌲 Recycling 1 ton saves 17 trees and 7,000 gallons water", "energy": 20, "co2": 2.8},
        {"fact": "🔋 Recycled paper needs 60% less energy than virgin paper", "energy": 15, "co2": 2.2}
    ],
    "plastic": [
        {"fact": "💻 One bottle recycled powers a computer for 25 minutes", "energy": 35, "co2": 4.8},
        {"fact": "⚡ Recycled plastic uses 88% less energy than new plastic", "energy": 38, "co2": 5.3}
    ],
    "trash": [
        {"fact": "🗑️ Proper sorting prevents contamination of recyclables", "energy": 5, "co2": 0.8},
        {"fact": "🌍 Mindful disposal protects soil and water", "energy": 4, "co2": 0.6}
    ]
}

RECYCLING_GUIDE = {
    "cardboard": "📦 BLUE BIN → Flatten boxes, remove tape, keep dry",
    "glass": "🗑️ GLASS BIN → Rinse bottles/jars, remove caps",
    "metal": "🗑️ METAL BIN → Rinse cans, crush aluminum",
    "paper": "📄 PAPER BIN → Keep dry, no grease stains",
    "plastic": "♻️ PLASTIC BIN → Check recycling number, rinse thoroughly",
    "trash": "🗑️ TRASH BIN → Non-recyclables, compost if organic"
}

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model_name = "yangy50/garbage-classification"
        model = AutoModelForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

# Smart classification function with wrapper detection
def classify_waste(image, model, processor):
    try:
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()
        
        labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        predictions = {labels[i]: float(probs[i]) for i in range(len(labels))}
        
        # SMART WRAPPER DETECTION LOGIC
        # If metal AND plastic/trash both high → it's a wrapper (metallized plastic)
        metal_score = predictions.get("metal", 0)
        plastic_score = predictions.get("plastic", 0)
        trash_score = predictions.get("trash", 0)
        
        # Rule 1: High metal + moderate plastic = wrapper → classify as plastic
        if metal_score > 0.35 and plastic_score > 0.20:
            top_class = "plastic"
            confidence = min(0.75, (metal_score + plastic_score) / 1.5)
        
        # Rule 2: High metal + high trash + low plastic = foil wrapper → plastic
        elif metal_score > 0.40 and trash_score > 0.25 and plastic_score > 0.05:
            top_class = "plastic"
            confidence = 0.70
        
        # Rule 3: Normal classification
        else:
            top_class = max(predictions, key=predictions.get)
            confidence = predictions[top_class]
            
            # If confidence too low, mark as trash
            if confidence < 0.45:
                top_class = "trash"
                confidence = max(0.50, trash_score)
        
        return top_class, confidence, predictions
        
    except Exception as e:
        st.error(f"Classification error: {e}")
        return None, None, None

# Navigation Bar
st.markdown("""
<div class="nav-bar">
    <div>
        <span class="logo">EcoSort</span>
        <span class="nav-subtitle">Smart Recycling Assistant • Enhanced for Mixed Materials</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-minimal">
    <h1 class="hero-title">Classify Your Waste<br>Instantly with AI</h1>
    <p class="hero-subtitle">Upload a photo and get instant recycling instructions powered by advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Stats Row
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.total_items}</div>
        <div class="stat-label">Items Classified</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.eco_score}</div>
        <div class="stat-label">Eco Points</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.energy_saved:.0f}</div>
        <div class="stat-label">kWh Saved</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.co2_prevented:.1f}</div>
        <div class="stat-label">kg CO₂ Reduced</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Quick Guide")
    
    st.markdown("""
    <div class="feature">
        <div class="feature-title">🎯 How It Works</div>
        <div class="feature-desc">
        1. Upload waste photo<br>
        2. AI analyzes instantly<br>
        3. Get recycling guide<br>
        4. Earn eco points
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ♻️ 6 Categories")
    categories = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
    
    for cat in categories:
        st.markdown(f"<span class='badge'>{cat}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Achievement badges
    # New Enhanced Leaderboard
st.markdown("### 🏆 Live Leaderboard")
leaderboard = [
    {"name": "Eco Warrior", "score": 250},
    {"name": "Green Champion", "score": 180},
    {"name": "⭐ YOU", "score": st.session_state.eco_score},
    {"name": "Recycle Master", "score": 120},
    {"name": "Planet Saver", "score": 90}
]
for i, user in enumerate(leaderboard, 1):
    emoji = "👑" if i == 1 else "⭐" if i <= 3 else "🔸"
    st.markdown(f"{emoji} **{user['name']}:** {user['score']} pts")

# Achievements
if st.session_state.achievements:
    st.markdown("### 🏅 Your Achievements")
    for achievement in st.session_state.achievements:
        st.markdown(f"<div class='badge'>{achievement}</div>", unsafe_allow_html=True)
    
    if st.button("🔄 Reset Stats", use_container_width=True):
        st.session_state.eco_score = 0
        st.session_state.total_items = 0
        st.session_state.energy_saved = 0
        st.session_state.co2_prevented = 0
        st.rerun()

# Main content
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-title">📸 Upload Waste Image</div>', unsafe_allow_html=True)
    
    # Simple file uploader without key parameter
    uploaded = st.file_uploader("Drop your image here or click to browse", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        # Store the uploaded file in session state
        st.session_state.uploaded_file = uploaded
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_column_width=True)

with col_right:
    st.markdown('<div class="section-title">🔍 Classification Results</div>', unsafe_allow_html=True)
    
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        with st.spinner("🔍 Analyzing with smart AI..."):
            model, processor = load_model()
            
            if model and processor:
                category, conf, all_preds = classify_waste(img, model, processor)
                
                if category:
                    # Result
                    st.markdown(f"""
                    <div class="result-box">
                        <p class="result-category">{category.title()}</p>
                        <p class="confidence">Confidence: {conf*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    #New impact metrics
                    impact_data = {
                        "plastic": {"energy": 35, "co2": 4.8, "water": 5, "trees": 0.1},
                        "metal": {"energy": 45, "co2": 6.5, "water": 8, "trees": 0.2},
                        "paper": {"energy": 20, "co2": 2.8, "water": 12, "trees": 0.3},
                        "glass": {"energy": 30, "co2": 4.2, "water": 3, "trees": 0.05},
                        "cardboard": {"energy": 24, "co2": 3.3, "water": 10, "trees": 0.25},
                        "trash": {"energy": 2, "co2": 0.3, "water": 1, "trees": 0.01}
                        }
                    
                    data = impact_data.get(category, impact_data["trash"])
                    st.markdown("#### 🌟 Environmental Impact")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⚡ Energy", f"{data['energy']} kWh")
                    with col2:
                        st.metric("🌫️ CO₂", f"{data['co2']} kg")
                    with col3:
                        st.metric("💧 Water", f"{data['water']}L")
                    with col4:
                        st.metric("🌳 Trees", f"{data['trees']}")
                        
                    st.progress(conf)
                    
                    # Guide
                    guide = RECYCLING_GUIDE.get(category, RECYCLING_GUIDE["trash"])
                    st.markdown(f"""
                    <div class="info-box">
                        <p class="info-heading">How to Recycle</p>
                        <p class="info-content">{guide}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Eco fact
                    eco_data = random.choice(ECO_FACTS.get(category, ECO_FACTS["trash"]))
                    st.markdown(f"""
                    <div class="impact-box">
                        <p class="impact-fact">{eco_data['fact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # New Action button
                    if st.button("🎯 RECYCLE & EARN 15 POINTS", use_container_width=True, type="primary"):
                        st.session_state.eco_score += 15
                        st.session_state.total_items += 1
                        st.session_state.energy_saved += eco_data['energy']
                        st.session_state.co2_prevented += eco_data['co2']
                        
                        # Achievements
                        if st.session_state.total_items == 1:
                            st.session_state.achievements.append("🌱 First Step")
                        if st.session_state.total_items == 5:
                            st.session_state.achievements.append("🔥 Recycling Streak")
                        if st.session_state.energy_saved > 100:
                            st.session_state.achievements.append("⚡ Energy Hero")
                        
                        st.balloons()
                        st.snow()
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #00ff88, #00ccff); 
                                    padding: 2rem; border-radius: 20px; text-align: center; 
                                    margin: 1rem 0; color: white; font-weight: bold;">
                            <h2>🎉 ECO HERO!</h2>
                            <p>+15 Points | Total: {st.session_state.eco_score}</p>
                            <p>You saved {eco_data['energy']} kWh of energy!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.uploaded_file = None
                        st.rerun()
                    
                    # All predictions
                    with st.expander("📊 All Predictions"):
                        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
                        for i, (name, score) in enumerate(sorted_preds, 1):
                            st.write(f"**{i}. {name.title()}**: {score*100:.2f}%")
    else:
        st.markdown("""
        <div class="feature">
            <p class="feature-title">👆 Get Started in 3 Easy Steps</p>
            <p class="feature-desc">
                <strong>1. 📸 Upload</strong> - Take or upload a waste item photo<br><br>
                <strong>2. 🤖 Analyze</strong> - AI classifies in seconds with smart wrapper detection<br><br>
                <strong>3. ♻️ Recycle</strong> - Follow disposal instructions<br><br>
                <br>
                <strong>What You'll Get:</strong><br>
                ✓ Smart AI <br>
                ✓ Detailed recycling instructions<br>
                ✓ Environmental impact metrics<br>
                ✓ Eco points & achievement badges<br>
                ✓ Track your contribution to saving the planet
            </p>
        </div>
        """, unsafe_allow_html=True)

# New #COLLECTIVE IMPACT DASHBOARD
st.markdown("---")
st.markdown("### 🌍 Your Collective Impact")

total_impact = {
    "trees_saved": st.session_state.total_items * 0.15,
    "water_saved": st.session_state.total_items * 8,
    "marine_life": st.session_state.total_items * 0.3
}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌳 Trees Saved", f"{total_impact['trees_saved']:.1f}")
with col2:
    st.metric("💧 Water Saved", f"{total_impact['water_saved']:.0f}L")  
with col3:
    st.metric("🐠 Marine Life", f"{total_impact['marine_life']:.1f} saved")

# Footer with Enhanced Design
st.markdown("---")
st.markdown('<p class="section-heading">🌍 Why Recycling Matters</p>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="card">
        <p class="feature-title">🌍 Environmental Impact</p>
        <p class="feature-desc">
            • Reduces landfill waste by 50-70%<br>
            • Prevents toxic soil & water contamination<br>
            • Protects wildlife & ecosystems<br>
            • Combats climate change
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="card">
        <p class="feature-title">⚡ Energy Conservation</p>
        <p class="feature-desc">
            • Saves up to 95% energy vs new materials<br>
            • Reduces greenhouse gas emissions<br>
            • Conserves natural resources<br>
            • Powers millions of homes
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="card">
        <p class="feature-title">💰 Economic Benefits</p>
        <p class="feature-desc">
            • Creates 6x more jobs than landfills<br>
            • Generates material revenue<br>
            • Reduces production costs<br>
            • Builds circular economy
        </p>
    </div>
    """, unsafe_allow_html=True)

# Add additional info section
st.markdown("---")
st.markdown('<p class="section-heading">📚 Supported Categories</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature">
        <p class="feature-title">♻️ Recyclable Materials</p>
        <p class="feature-desc">
            <strong>Paper & Cardboard:</strong> Newspapers, magazines, boxes<br>
            <strong>Plastics:</strong> Bottles, containers, wrappers <br>
            <strong>Glass:</strong> Bottles, jars (all colors)<br>
            <strong>Metals:</strong> Aluminum cans, steel cans, foil<br>
            <strong>Mixed Materials:</strong> Smart detection for wrappers
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature">
        <p class="feature-title">⚠️ Special Handling</p>
        <p class="feature-desc">
            <strong>Food Wrappers:</strong> Metallized plastic → plastic bin<br>
            <strong>Candy Wrappers:</strong> Check local rules, usually plastic<br>
            <strong>Organics:</strong> Compost or green waste bins<br>
            <strong>Hazardous:</strong> Paint, chemicals - special disposal<br>
            <strong>Mixed Materials:</strong> AI detects and guides properly
        </p>
    </div>
    """, unsafe_allow_html=True)

# Final call to action
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2.5rem; border-radius: 24px; text-align: center; margin: 2rem 0;
            box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);">
    <h2 style="color: white; margin: 0 0 1rem 0; font-size: 2rem;">
        🌟 Start Making a Difference Today!
    </h2>
    <p style="color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0;">
        Every item you recycle correctly helps build a sustainable future. 
        Together, we can make our planet cleaner and greener! 🌱
    </p>
</div>
""", unsafe_allow_html=True)
