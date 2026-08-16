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
# 1. UI SETUP & CACHING
# ==========================================
st.set_page_config(page_title="VCT VISION", page_icon="🔴", layout="wide")

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
    except Exception:
        return None, None

model, feature_columns = load_model()

if 'player_cache' not in st.session_state:
    st.session_state.player_cache = {}
if 'url_cache' not in st.session_state:
    st.session_state.url_cache = ''

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_agent(name): 
    return name.lower().replace("/", "").replace(" ", "").strip() if name else "unknown"

def parse_stat(text, default):
    try:
        t = str(text).replace('%', '').strip()
        if not t or t == '-': return float(default)
        return float(t)
    except: 
        return float(default)

def check_team_comp(agent_list):
    agents = set([clean_agent(a) for a in agent_list])
    duelists = {"jett", "raze", "reyna", "phoenix", "yoru", "neon", "iso"}
    controllers = {"omen", "brimstone", "viper", "astra", "harbor", "clove"}
    initiators = {"sova", "breach", "skye", "kayo", "fade", "gekko", "tejo"}
    sentinels = {"killjoy", "cypher", "sage", "chamber", "deadlock", "vyse", "waylay"}
    d, c = len(agents & duelists), len(agents & controllers)
    i, s = len(agents & initiators), len(agents & sentinels)
    if (i == 2 and d == 1 and s == 1 and c == 1) or (c == 2 and d == 1 and i == 1 and s == 1): return 2
    if d == 2 and c == 1 and i == 1 and s == 1: return 1
    return 0

def check_agent_duo(agent_list, map_name):
    agents = set([clean_agent(a) for a in agent_list])
    map_name = map_name.lower()
    if "lotus" in map_name and {"fade", "raze"}.issubset(agents): return 1
    if "bind" in map_name and ({"raze", "brimstone"}.issubset(agents) or {"raze", "skye"}.issubset(agents)): return 1
    if "pearl" in map_name and {"harbor", "viper"}.issubset(agents): return 1
    if ("fracture" in map_name or "sunset" in map_name) and {"raze", "breach"}.issubset(agents): return 1
    if ("ascent" in map_name or "abyss" in map_name or "icebox" in map_name) and {"jett", "sova"}.issubset(agents): return 1
    if "split" in map_name and {"raze", "cypher"}.issubset(agents): return 1
    return 0

# ==========================================
# 3. ROBUST SCRAPER ENGINE
# ==========================================
def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"})
    driver.set_page_load_timeout(45)
    return driver

def scrape_match_data(url, status_text):
    driver = get_driver()
    try:
        status_text.markdown("📡 **Connecting to VLR match page...**")
        driver.get(url)
        wait = WebDriverWait(driver, 12)
        
        try:
            t1_name = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.match-header-link-name.mod-1"))).text.strip()
            t2_name = driver.find_element(By.CSS_SELECTOR, "div.match-header-link-name.mod-2").text.strip()
        except TimeoutException:
            return {"error": "Connection timed out or invalid match URL."}

        status_text.markdown(f"⚔️ **Match Loaded:** `{t1_name} vs {t2_name}` — Resolving rosters...")

        try:
            stats_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.vm-stats-container")))
        except TimeoutException:
            return {"error": "Stats container failed to load on VLR.gg."}

        active_game_id = None
        map_name = "Overall"
        
        try:
            active_tab = driver.find_element(By.CSS_SELECTOR, "div.vm-stats-gamesnav-item.mod-active")
            active_game_id = active_tab.get_attribute("data-game-id")
            raw_map = active_tab.text.replace('\n', ' ').split()[-1].strip()
            if raw_map.lower() not in ["all", "maps", ""]:
                map_name = raw_map.capitalize()
            else:
                map_name = "Overall"
        except:
            map_name = "Overall"

        target_container = stats_container
        if active_game_id and active_game_id not in ["all", "0"]:
            try:
                target_container = stats_container.find_element(By.CSS_SELECTOR, f"div.vm-stats-game[data-game-id='{active_game_id}']")
            except:
                pass

        wait.until(lambda d: len(target_container.find_elements(By.CSS_SELECTOR, "div.ovw-row, tbody tr")) > 0)
        all_rows = target_container.find_elements(By.CSS_SELECTOR, "div.ovw-row, tbody tr")

        t1_players, t2_players = [], []
        t1_agents, t2_agents = [], []
        player_hrefs = {}

        for row in all_rows:
            try:
                a_tags = row.find_elements(By.CSS_SELECTOR, "a[href*='/player/']")
                if not a_tags: 
                    continue
                
                href = a_tags[0].get_attribute("href")
                if not href or "/player/" not in href: 
                    continue

                p_id = href.split("/player/")[1].split("/")[0]
                p_name = a_tags[0].text.strip()
                if not p_name:
                    p_name = href.split("/player/")[1].split("/")[1].replace("-", " ").title()

                if p_name in player_hrefs:
                    continue

                player_hrefs[p_name] = href

                ag = "unknown"
                img_tags = row.find_elements(By.CSS_SELECTOR, "img[src*='/agents/']")
                if img_tags:
                    raw_ag = img_tags[0].get_attribute("title") or img_tags[0].get_attribute("alt") or ""
                    if not raw_ag:
                        src = img_tags[0].get_attribute("src") or ""
                        raw_ag = src.split("/")[-1].split(".")[0]
                    ag = clean_agent(raw_ag)

                if len(t1_players) < 5:
                    t1_players.append(p_name)
                    t1_agents.append(ag)
                elif len(t2_players) < 5:
                    t2_players.append(p_name)
                    t2_agents.append(ag)
            except:
                continue

        if len(t1_players) < 5 or len(t2_players) < 5:
            return {"error": f"Failed to resolve complete 10-player roster. Found: {len(t1_players)} T1, {len(t2_players)} T2."}

        player_stats = {}
        main_window = driver.current_window_handle
        total_players = len(t1_players) + len(t2_players)
        current_idx = 0

        for t_idx, p_list in enumerate([t1_players, t2_players]):
            for p_idx, p_name in enumerate(p_list):
                current_idx += 1
                p_id = player_hrefs[p_name].split("player/")[1].split("/")[0]
                ag = t1_agents[p_idx] if t_idx == 0 else t2_agents[p_idx]
                
                cache_key = f"{p_id}_{ag}"
                if cache_key in st.session_state.player_cache and ag != "unknown":
                    player_stats[p_name] = st.session_state.player_cache[cache_key]
                    status_text.markdown(f"📊 **[{current_idx}/{total_players}] Analyzed records for {p_name}** (Cached)")
                    continue

                status_text.markdown(f"📊 **[{current_idx}/{total_players}] Analyzing records for {p_name}**...")
                p_stats = {'acs': 200.0, 'kd': 1.0, 'fkpr': 0.10, 'fdpr': 0.10, 'resolved_agent': ag}

                driver.execute_script("window.open('');")
                driver.switch_to.window(driver.window_handles[-1])

                try:
                    found = False
                    for timespan in ["60d", "all"]:
                        if found: break
                        driver.get(f"https://www.vlr.gg/player/{p_id}/?timespan={timespan}")
                        time.sleep(0.3)

                        stat_tables = driver.find_elements(By.CSS_SELECTOR, "table.st-table")
                        for tbl in stat_tables:
                            if "ACS" not in tbl.text: continue
                            rows = tbl.find_elements(By.CSS_SELECTOR, "tbody tr")
                            if not rows: continue

                            target_rows = rows if ag != "unknown" else [rows[0]]

                            for r in target_rows:
                                cols = r.find_elements(By.TAG_NAME, "td")
                                if len(cols) < 16: continue

                                img = cols[0].find_elements(By.TAG_NAME, "img")
                                row_agent = "unknown"
                                if img:
                                    raw = img[0].get_attribute("alt") or img[0].get_attribute("title") or ""
                                    row_agent = clean_agent(raw)

                                if ag == "unknown" or row_agent == ag:
                                    rounds = parse_stat(cols[2].text, 1.0)
                                    fk = parse_stat(cols[14].text, 0.0)
                                    fd = parse_stat(cols[15].text, 0.0)

                                    p_stats = {
                                        'acs': parse_stat(cols[4].text, 200.0),
                                        'kd': parse_stat(cols[5].text, 1.0),
                                        'fkpr': round(fk / rounds, 2) if rounds > 0 else 0.10,
                                        'fdpr': round(fd / rounds, 2) if rounds > 0 else 0.10,
                                        'resolved_agent': row_agent if row_agent != "unknown" else "jett"
                                    }
                                    if ag != "unknown":
                                        st.session_state.player_cache[cache_key] = p_stats
                                    found = True
                                    break
                            if found: break
                except:
                    pass
                finally:
                    driver.close()
                    driver.switch_to.window(main_window)

                player_stats[p_name] = p_stats

                if ag == "unknown":
                    if t_idx == 0: t1_agents[p_idx] = p_stats['resolved_agent']
                    else: t2_agents[p_idx] = p_stats['resolved_agent']

        return {
            "t1_name": t1_name, "t2_name": t2_name,
            "t1_players": t1_players, "t2_players": t2_players,
            "t1_agents": t1_agents, "t2_agents": t2_agents,
            "map_name": map_name,
            "player_stats": player_stats
        }
    except Exception as e:
        return {"error": f"Scraping Failed: {str(e)}"}
    finally:
        driver.quit()

# ==========================================
# 4. FRONT-END INTERFACE
# ==========================================
col_logo, col_title = st.columns([1, 15])
with col_logo: 
    st.image("logo.png", width=65) 
with col_title: 
    st.markdown("<h1 class='val-title'><span class='val-red'>VCT</span> VISION</h1>", unsafe_allow_html=True)

st.markdown("Enter any VLR match link (live, upcoming, or completed). The AI extracts combat metrics, estimates drafts if unselected, and computes match probabilities.")
st.divider()

if model is None:
    st.error("❌ Model Offline: Missing model files.")
    st.stop()

match_url = st.text_input("🔗 VLR Match URL:", placeholder="https://www.vlr.gg/...")
if match_url != st.session_state.url_cache:
    st.session_state.url_cache = match_url
    if 'match_data' in st.session_state: 
        del st.session_state['match_data']

if st.button("🔴 RUN PREDICTION ALGORITHM"):
    cleaned_url = match_url.strip()
    if not cleaned_url or "vlr.gg" not in cleaned_url:
        st.warning("⚠️ Invalid URL. Enter a valid vlr.gg link.")
    else:
        status_text = st.empty()
        st.session_state.match_data = scrape_match_data(cleaned_url, status_text)
        status_text.empty()

# ==========================================
# 5. PREDICTION & METRICS DASHBOARD
# ==========================================
if 'match_data' in st.session_state:
    data = st.session_state.match_data
    
    if "error" in data:
        st.error(f"❌ {data['error']}")
    else:
        st.success("✅ Match Data Successfully Processed!")
        st.divider()
        
        t1_agents = data['t1_agents']
        t2_agents = data['t2_agents']
        map_name = data['map_name']
        
        t1_stats = {'kd':[], 'acs':[], 'fkpr':[], 'fdpr':[]}
        t2_stats = {'kd':[], 'acs':[], 'fkpr':[], 'fdpr':[]}
        fallback = {'kd':1.0, 'acs':200.0, 'fkpr':0.1, 'fdpr':0.1}
        
        for p in data['t1_players']:
            s_dict = data['player_stats'].get(p, fallback)
            for k in ['kd', 'acs', 'fkpr', 'fdpr']: t1_stats[k].append(s_dict[k])
            
        for p in data['t2_players']:
            s_dict = data['player_stats'].get(p, fallback)
            for k in ['kd', 'acs', 'fkpr', 'fdpr']: t2_stats[k].append(s_dict[k])

        input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in feature_columns:
            if map_name != "Overall" and col.lower() == map_name.lower(): 
                input_df[col] = 1.0
            
        input_df['T1_Avg_KD'] = np.mean(t1_stats['kd'])
        input_df['T2_Avg_KD'] = np.mean(t2_stats['kd'])
        input_df['T1_Avg_ACS'] = np.mean(t1_stats['acs'])
        input_df['T2_Avg_ACS'] = np.mean(t2_stats['acs'])
        input_df['T1_Std_KD'] = np.std(t1_stats['kd'])
        input_df['T2_Std_KD'] = np.std(t2_stats['kd'])
        input_df['T1_Max_KD'] = np.max(t1_stats['kd'])
        input_df['T2_Max_KD'] = np.max(t2_stats['kd'])
        
        if 'T1_Avg_FKPR' in input_df.columns:
            input_df['T1_Avg_FKPR'] = np.mean(t1_stats['fkpr'])
            input_df['T2_Avg_FKPR'] = np.mean(t2_stats['fkpr'])
            input_df['T1_Avg_FDPR'] = np.mean(t1_stats['fdpr'])
            input_df['T2_Avg_FDPR'] = np.mean(t2_stats['fdpr'])
            
        input_df['Diff_Avg_KD'] = input_df['T1_Avg_KD'] - input_df['T2_Avg_KD']
        input_df['Diff_Avg_ACS'] = input_df['T1_Avg_ACS'] - input_df['T2_Avg_ACS']
        input_df['Diff_Max_KD'] = input_df['T1_Max_KD'] - input_df['T2_Max_KD']
        
        if 'Diff_Avg_FKPR' in input_df.columns:
            input_df['Diff_Avg_FKPR'] = input_df['T1_Avg_FKPR'] - input_df['T2_Avg_FKPR']
            input_df['Diff_Avg_FDPR'] = input_df['T1_Avg_FDPR'] - input_df['T2_Avg_FDPR']
            
        if 'Diff_Team_Combo' in input_df.columns:
            input_df['Diff_Team_Combo'] = check_team_comp(t1_agents) - check_team_comp(t2_agents)
            input_df['Diff_Agent_Combo'] = check_agent_duo(t1_agents, map_name) - check_agent_duo(t2_agents, map_name)

        probs = model.predict_proba(input_df)[0]
        t1_conf, t2_conf = probs[0] * 100, probs[1] * 100
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔮 AI Prediction", f"🛡️ {data['t1_name']} Roster", f"⚔️ {data['t2_name']} Roster", "🧠 Differentials"])
        
        with tab1:
            display_map = map_name.upper() if map_name != "Overall" else "SERIES AGGREGATE (ALL MAPS)"
            st.subheader(f"🗺️ Map: {display_map}")
            col1, col2 = st.columns(2)
            col1.metric(label=f"{data['t1_name']} Win Probability", value=f"{t1_conf:.2f}%")
            col2.metric(label=f"{data['t2_name']} Win Probability", value=f"{t2_conf:.2f}%")
                
            max_conf = max(t1_conf, t2_conf)
            winner = data['t1_name'] if t1_conf > t2_conf else data['t2_name']
            
            if max_conf >= 60: 
                st.success(f"🟢 **HIGH CONFIDENCE:** Prediction favors **{winner}**.")
            elif max_conf >= 55: 
                st.warning(f"🟡 **MODERATE RISK:** Slight statistical edge for **{winner}**.")
            else: 
                st.error(f"🔴 **COIN TOSS:** Highly unpredictable.")
        
        with tab2:
            st.dataframe(pd.DataFrame({
                "Player": data['t1_players'], 
                "Agent": [a.capitalize() for a in t1_agents], 
                "K/D": [f"{x:.2f}" for x in t1_stats['kd']], 
                "ACS": [f"{x:.1f}" for x in t1_stats['acs']], 
                "FKPR": [f"{x:.2f}" for x in t1_stats['fkpr']]
            }), use_container_width=True, hide_index=True)

        with tab3:
            st.dataframe(pd.DataFrame({
                "Player": data['t2_players'], 
                "Agent": [a.capitalize() for a in t2_agents], 
                "K/D": [f"{x:.2f}" for x in t2_stats['kd']], 
                "ACS": [f"{x:.1f}" for x in t2_stats['acs']], 
                "FKPR": [f"{x:.2f}" for x in t2_stats['fkpr']]
            }), use_container_width=True, hide_index=True)
            
        with tab4:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Aim Differential (K/D Diff)", f"{input_df['Diff_Avg_KD'].values[0]:.3f}")
            if 'Diff_Avg_FKPR' in input_df.columns: 
                col_m2.metric("First Blood Advantage (FKPR Diff)", f"{input_df['Diff_Avg_FKPR'].values[0]:.3f}")
            if 'Diff_Team_Combo' in input_df.columns:
                val = input_df['Diff_Team_Combo'].values[0]
                text_val = "Even" if val == 0 else f"{val:+d} Advantage"
                col_m3.metric("Draft Synergy Gap", text_val)
