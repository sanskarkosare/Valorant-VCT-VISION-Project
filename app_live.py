import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import xgboost as xgb
import catboost
import sklearn
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ==========================================
# 1. UI SETUP, CSS, & CACHING
# ==========================================
st.set_page_config(page_title="VCT Betting", page_icon="🔴", layout="wide")

st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #ff4655; color: white; border: none; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; padding: 0.75rem 1.5rem; transition: all 0.3s ease; }
    div.stButton > button:first-child:hover { background-color: #ff5865; box-shadow: 0px 4px 15px rgba(255, 70, 85, 0.4); transform: translateY(-2px); }
    div[data-testid="metric-container"] { background-color: #1a1a1a; border: 1px solid #333; padding: 5% 5% 5% 10%; border-radius: 8px; border-left: 5px solid #ff4655; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }
    div[data-testid="metric-container"] label { color: #8b978f !important; font-weight: 600; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ece8e1 !important; }
    .val-title { font-family: 'Arial Black', sans-serif; color: #ece8e1; text-transform: uppercase; letter-spacing: -1px; margin-top: 10px; }
    .val-red { color: #ff4655; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('valorant_v2_model.pkl')
        cols = joblib.load('training_columns_v2.pkl')
        return model, cols
    except Exception as e:
        return None, None

model, feature_columns = load_model()

if 'player_cache' not in st.session_state:
    st.session_state.player_cache = {}
if 'url_cache' not in st.session_state:
    st.session_state.url_cache = ''

# ==========================================
# 2. HELPER FUNCTIONS & SAFE PARSING
# ==========================================
def clean_agent(name): return name.lower().replace("/", "").replace(" ", "").strip() if name else "unknown"

def parse_stat(text, default):
    try:
        t = str(text).replace('%', '').strip()
        if not t or t == '-': return float(default)
        return float(t)
    except: return float(default)

def check_team_comp(agent_list):
    agents = set([clean_agent(a) for a in agent_list])
    duelists, controllers = {"jett", "raze", "reyna", "phoenix", "yoru", "neon", "iso"}, {"omen", "brimstone", "viper", "astra", "harbor", "clove"}
    initiators, sentinels = {"sova", "breach", "skye", "kayo", "fade", "gekko", "tejo"}, {"killjoy", "cypher", "sage", "chamber", "deadlock", "vyse", "waylay"}
    d, c, i, s = len(agents & duelists), len(agents & controllers), len(agents & initiators), len(agents & sentinels)
    if (i == 2 and d == 1 and s == 1 and c == 1) or (c == 2 and d == 1 and i == 1 and s == 1): return 2
    if d == 2 and c == 1 and i == 1 and s == 1: return 1
    return 0

# ==========================================
# 3. SCRAPER (Single Map Explicit Parsing)
# ==========================================
def get_visible_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") # CRITICAL FOR CLOUD DEPLOYMENT
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled") 
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Streamlit Cloud usually maps the chromium binary here automatically,
    # but sometimes specifying it helps prevent crashes.
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"})
    driver.set_page_load_timeout(30)
    return driver

def scrape_match_data(url, status_text):
    driver = get_visible_driver()
    try:
        status_text.text("📡 Uplink established. Bypassing Cloudflare...")
        
        driver.get(url) 
        wait = WebDriverWait(driver, 5)
        
        # 1. Team Names
        try:
            t1_name = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.match-header-link-name.mod-1"))).text.strip()
            t2_name = driver.find_element(By.CSS_SELECTOR, "div.match-header-link-name.mod-2").text.strip()
        except TimeoutException: 
            return {"error": "Cloudflare blocked the connection or match page doesn't exist."}

        status_text.text(f"⚔️ Match Acquired: {t1_name} vs {t2_name}. Processing player data...")

        # 2. Extract Map Name & Game ID Safely
        try:
            active_map_tab = driver.find_element(By.CSS_SELECTOR, ".vm-stats-gamesnav-item.mod-active")
            map_name = active_map_tab.text.replace('\n', ' ').split()[-1].strip()
            game_id = active_map_tab.get_attribute("data-game-id")
            if map_name.lower() == "all": 
                map_name = "All Maps"
        except:
            map_name = "Unknown Map"
            game_id = "all"

        # 3. Extract Players and Agents (Targeting the SPECIFIC map container)
        t1_players, t2_players, player_hrefs = [], [], {}
        t1_agents, t2_agents = [], []
        
        try:
            # Find the specific container for the active map
            game_container = driver.find_element(By.CSS_SELECTOR, f"div.vm-stats-game[data-game-id='{game_id}']")
            tables = game_container.find_elements(By.CSS_SELECTOR, "table.wf-table-inset")
        except:
            # Fallback if container isn't found
            tables = driver.find_elements(By.CSS_SELECTOR, "table.wf-table-inset")
        
        if len(tables) >= 2:
            for t_idx, tbl in enumerate(tables[:2]):
                for row in tbl.find_elements(By.CSS_SELECTOR, "tbody tr"):
                    try:
                        a_tag = row.find_element(By.CSS_SELECTOR, "td.mod-player a")
                        href = a_tag.get_attribute("href")
                        
                        try:
                            p_name = href.split('/player/')[1].split('/')[1].replace('-', ' ').title()
                        except:
                            p_name = row.find_element(By.CSS_SELECTOR, "td.mod-player").text.split('\n')[0].strip()
                            if not p_name: p_name = "Unknown"
                        
                        try: ag = row.find_element(By.CSS_SELECTOR, "td.mod-agents img").get_attribute("title")
                        except: ag = "unknown"
                        
                        if p_name not in player_hrefs:
                            player_hrefs[p_name] = href
                            
                        if t_idx == 0: 
                            t1_players.append(p_name)
                            t1_agents.append(clean_agent(ag))
                        else: 
                            t2_players.append(p_name)
                            t2_agents.append(clean_agent(ag))
                    except: pass

        if len(t1_players) < 5 or len(t2_players) < 5:
            return {"error": "VLR.gg doesn't have all 10 players listed yet for this specific map URL!"}
            
        t1_players, t2_players = t1_players[:5], t2_players[:5]
        t1_agents, t2_agents = t1_agents[:5], t2_agents[:5]

        # 4. Fetch Stats for specific agents played
        player_stats = {} 
        main_window = driver.current_window_handle
        
        for t_idx, p_list in enumerate([t1_players, t2_players]):
            for p_idx, p_name in enumerate(p_list):
                player_stats[p_name] = {}
                p_id = player_hrefs[p_name].split("player/")[1].split("/")[0]
                
                ag = t1_agents[p_idx] if t_idx == 0 else t2_agents[p_idx]
                
                status_text.text(f"📊 Scanning records for {p_name} ({ag})...")
                
                cache_key = f"{p_id}_{ag}"
                if cache_key in st.session_state.player_cache and ag != "unknown":
                    player_stats[p_name] = st.session_state.player_cache[cache_key]
                    continue
                
                # Default fallback
                p_stats = {'acs': 200.0, 'kd': 1.0, 'fkpr': 0.10, 'fdpr': 0.10, 'clutch_pct': 0.15, 'actual_agent': 'unknown'}
                
                driver.execute_script("window.open('');")
                driver.switch_to.window(driver.window_handles[-1])
                
                try:
                    data_found = False
                    for timespan in ["60d", "all"]:
                        if data_found: break
                        driver.get(f"https://www.vlr.gg/player/{p_id}/?timespan={timespan}")
                        time.sleep(0.5)
                        player_tables = driver.find_elements(By.CSS_SELECTOR, "table.wf-table")
                        
                        for tbl in player_tables:
                            if "ACS" not in tbl.text: continue
                            
                            for ar in tbl.find_elements(By.CSS_SELECTOR, "tbody tr"):
                                acols = ar.find_elements(By.TAG_NAME, "td")
                                if len(acols) >= 12:
                                    try: img_tag = acols[0].find_element(By.TAG_NAME, "img")
                                    except: continue
                                    
                                    raw_a = img_tag.get_attribute("title") or img_tag.get_attribute("alt")
                                    if not raw_a: 
                                        src = img_tag.get_attribute("src")
                                        if src: raw_a = src.split("/")[-1].split(".")[0]
                                    
                                    ar_agent = clean_agent(raw_a)
                                    
                                    if ag == "unknown" or ar_agent == ag:
                                        p_stats = {
                                            'acs': parse_stat(acols[4].text, 200.0),
                                            'kd': parse_stat(acols[5].text, 1.0),
                                            'fkpr': parse_stat(acols[10].text, 0.1),
                                            'fdpr': parse_stat(acols[11].text, 0.1),
                                            'clutch_pct': parse_stat(acols[-1].text, 15.0) / 100.0,
                                            'actual_agent': ar_agent
                                        }
                                        if ag != "unknown":
                                            st.session_state.player_cache[cache_key] = p_stats
                                        data_found = True
                                        break
                            if data_found: break
                except: pass
                finally:
                    driver.close()
                    driver.switch_to.window(main_window)
                
                player_stats[p_name] = p_stats
                
                # Backfill unknown agents
                if ag == "unknown":
                    resolved_agent = p_stats.get('actual_agent', 'unknown')
                    if t_idx == 0:
                        t1_agents[p_idx] = resolved_agent
                    else:
                        t2_agents[p_idx] = resolved_agent

        return {
            "t1_name": t1_name, "t2_name": t2_name,
            "t1_players": t1_players, "t2_players": t2_players,
            "t1_agents": t1_agents, "t2_agents": t2_agents,
            "map_name": map_name,
            "player_stats": player_stats
        }
    except Exception as e:
        return {"error": f"Critical Scraping Error: {str(e)}"}
    finally:
        driver.quit()

# ==========================================
# 4. FRONT-END INTERFACE
# ==========================================
col_logo, col_title = st.columns([1, 15])
with col_logo: 
    st.image("logo.png", width=65) 
with col_title: 
    st.markdown("<h1 class='val-title'><span class='val-red'>VCT</span> BETTING</h1>", unsafe_allow_html=True)

st.markdown("Paste a specific VLR.gg single map link below. The AI parses the combat records of all 10 players, calculates Synergy Differentials, and executes a Neural Network prediction.")
st.divider()

if model is None:
    st.error("❌ Neural Network Offline: 'valorant_v2_model.pkl' not found.")
    st.stop()

# Auto-clear memory if URL changes
match_url = st.text_input("🔗 VLR Single Map URL:", placeholder="https://www.vlr.gg/.../?game=12345...")
if match_url != st.session_state.url_cache:
    st.session_state.url_cache = match_url
    if 'match_data' in st.session_state: 
        del st.session_state['match_data']

if st.button("🔴 RUN PREDICTION ALGORITHM"):
    match_url = match_url.strip() 
    if not match_url or "vlr.gg" not in match_url:
        st.warning("⚠️ Invalid URL. Please provide a standard VLR.gg match link.")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)
        st.session_state.match_data = scrape_match_data(match_url, status_text)
        status_text.empty()
        progress_bar.empty()

# ==========================================
# 5. DYNAMIC STATEFUL DASHBOARD
# ==========================================
if 'match_data' in st.session_state:
    match_data = st.session_state.match_data
    
    if "error" in match_data:
        st.error(f"❌ {match_data['error']}")
    else:
        st.success("✅ Match Data Successfully Scraped & Cached!")
        st.divider()
        
        t1_agents = match_data['t1_agents']
        t2_agents = match_data['t2_agents']
        map_name = match_data['map_name']
        
        t1_stats = {'kd':[], 'acs':[], 'fkpr':[], 'fdpr':[], 'clutch':[]}
        t2_stats = {'kd':[], 'acs':[], 'fkpr':[], 'fdpr':[], 'clutch':[]}
        
        fallback_stats = {'kd':1.0, 'acs':200.0, 'fkpr':0.1, 'fdpr':0.1, 'clutch_pct':0.15}
        
        for p in match_data['t1_players']:
            stats = match_data['player_stats'].get(p, fallback_stats)
            for k in ['kd', 'acs', 'fkpr', 'fdpr']: t1_stats[k].append(stats[k])
            t1_stats['clutch'].append(stats['clutch_pct'])
            
        for p in match_data['t2_players']:
            stats = match_data['player_stats'].get(p, fallback_stats)
            for k in ['kd', 'acs', 'fkpr', 'fdpr']: t2_stats[k].append(stats[k])
            t2_stats['clutch'].append(stats['clutch_pct'])

        # --- MATH & PREDICTION ---
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in feature_columns:
            if col.lower() == map_name.lower(): input_df[col] = 1.0
            
        input_df['T1_Avg_KD'] = np.mean(t1_stats['kd']); input_df['T2_Avg_KD'] = np.mean(t2_stats['kd'])
        input_df['T1_Avg_ACS'] = np.mean(t1_stats['acs']); input_df['T2_Avg_ACS'] = np.mean(t2_stats['acs'])
        input_df['T1_Std_KD'] = np.std(t1_stats['kd']); input_df['T2_Std_KD'] = np.std(t2_stats['kd'])
        input_df['T1_Max_KD'] = np.max(t1_stats['kd']); input_df['T2_Max_KD'] = np.max(t2_stats['kd'])
        
        if 'T1_Avg_FKPR' in input_df.columns:
            input_df['T1_Avg_FKPR'] = np.mean(t1_stats['fkpr'])
            input_df['T2_Avg_FKPR'] = np.mean(t2_stats['fkpr'])
            input_df['T1_Avg_FDPR'] = np.mean(t1_stats['fdpr'])
            input_df['T2_Avg_FDPR'] = np.mean(t2_stats['fdpr'])
            input_df['T1_Avg_Clutch'] = np.mean(t1_stats['clutch'])
            input_df['T2_Avg_Clutch'] = np.mean(t2_stats['clutch'])
            
        input_df['Diff_Avg_KD'] = input_df['T1_Avg_KD'] - input_df['T2_Avg_KD']
        input_df['Diff_Avg_ACS'] = input_df['T1_Avg_ACS'] - input_df['T2_Avg_ACS']
        
        if 'Diff_Avg_FKPR' in input_df.columns:
            input_df['Diff_Avg_FKPR'] = input_df['T1_Avg_FKPR'] - input_df['T2_Avg_FKPR']
            input_df['Diff_Avg_Clutch'] = input_df['T1_Avg_Clutch'] - input_df['T2_Avg_Clutch']
            
        if 'Diff_Team_Combo' in input_df.columns:
            input_df['Diff_Team_Combo'] = check_team_comp(t1_agents) - check_team_comp(t2_agents)

        probs = model.predict_proba(input_df)[0]
        t1_conf = probs[0] * 100
        t2_conf = probs[1] * 100
        
        # --- UI TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(["🔮 AI Prediction", f"🛡️ {match_data['t1_name']} Roster", f"⚔️ {match_data['t2_name']} Roster", "🧠 Deep Backend Math"])
        
        with tab1:
            st.subheader(f"🗺️ Map Data: {map_name.upper()}")
            col1, col2 = st.columns(2)
            col1.metric(label=f"{match_data['t1_name']} Win Probability", value=f"{t1_conf:.2f}%")
            col2.metric(label=f"{match_data['t2_name']} Win Probability", value=f"{t2_conf:.2f}%")
                
            max_conf = max(t1_conf, t2_conf)
            winner = match_data['t1_name'] if t1_conf > t2_conf else match_data['t2_name']
            
            if max_conf >= 60: 
                st.success(f"📈 **APPROVED BET:** The Oracle strongly favors **{winner}**.")
            elif max_conf >= 55: 
                st.warning(f"⚠️ **MODERATE RISK:** The AI leans towards **{winner}**, but it's a volatile match.")
            else: 
                st.error(f"🛑 **COIN FLIP:** Do not bet. The statistics are perfectly deadlocked.")
        
        with tab2:
            st.write(f"**Specific Agent Data for {match_data['t1_name']} on {map_name}**")
            df_t1 = pd.DataFrame({
                "Player": match_data['t1_players'], 
                "Agent": [a.capitalize() for a in t1_agents], 
                "K/D": [f"{x:.2f}" for x in t1_stats['kd']], 
                "ACS": [f"{x:.1f}" for x in t1_stats['acs']], 
                "FKPR (First Kills)": [f"{x:.2f}" for x in t1_stats['fkpr']], 
                "Clutch %": [f"{x*100:.1f}%" for x in t1_stats['clutch']]
            })
            st.dataframe(df_t1, use_container_width=True, hide_index=True)

        with tab3:
            st.write(f"**Specific Agent Data for {match_data['t2_name']} on {map_name}**")
            df_t2 = pd.DataFrame({
                "Player": match_data['t2_players'], 
                "Agent": [a.capitalize() for a in t2_agents], 
                "K/D": [f"{x:.2f}" for x in t2_stats['kd']], 
                "ACS": [f"{x:.1f}" for x in t2_stats['acs']], 
                "FKPR (First Kills)": [f"{x:.2f}" for x in t2_stats['fkpr']], 
                "Clutch %": [f"{x*100:.1f}%" for x in t2_stats['clutch']]
            })
            st.dataframe(df_t2, use_container_width=True, hide_index=True)
            
        with tab4:
            st.markdown("### 🔢 Artificial Intelligence Differentials")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            col_m1.metric("Raw Aim Gap (K/D Diff)", f"{input_df['Diff_Avg_KD'].values[0]:.3f}")
            
            if 'Diff_Avg_FKPR' in input_df.columns: 
                col_m2.metric("First Blood Advantage (FKPR Diff)", f"{input_df['Diff_Avg_FKPR'].values[0]:.3f}")
                
            if 'Diff_Team_Combo' in input_df.columns:
                val = input_df['Diff_Team_Combo'].values[0]
                text_val = "Even" if val == 0 else f"+{val} Advantage"
                col_m3.metric("Draft Synergy Gap", text_val) 

                col_m2.metric("First Blood Advantage (FKPR Diff)", f"{input_df['Diff_Avg_FKPR'].values[0]:.3f}")

                

            if 'Diff_Team_Combo' in input_df.columns:

                val = input_df['Diff_Team_Combo'].values[0]

                text_val = "Even" if val == 0 else f"+{val} Advantage"

                col_m3.metric("Draft Synergy Gap", text_val)
