"""
VCT VISION v2 — Streamlit Match Prediction App
===============================================
Changes from v1:
  - Full Valorant dark gaming UI
  - Live scraping progress terminal
  - Agent icons from valorant-api.com
  - How-to-use steps
  - Improved team cards with glow effects
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, time, re, warnings, requests
warnings.filterwarnings('ignore')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VCT Vision", page_icon="🎯",
    layout="wide", initial_sidebar_state="collapsed"
)

# ─── FULL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0d0d14;
    color: #e0e0e0;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── HEADER ── */
.vct-header {
    background: linear-gradient(135deg, #0d0d14 0%, #1a0a10 50%, #0d0d14 100%);
    border-bottom: 2px solid #ff4655;
    padding: 20px 0 16px 0;
    margin-bottom: 24px;
    text-align: center;
}
.vct-title {
    font-size: 42px; font-weight: 700; letter-spacing: 6px;
    color: #fff; text-transform: uppercase; margin: 0;
}
.vct-title span { color: #ff4655; }
.vct-sub { font-size: 13px; letter-spacing: 3px; color: #888;
           text-transform: uppercase; margin-top: 4px; }

/* ── HOW TO USE ── */
.how-box {
    background: #12121f; border: 1px solid #1e1e30;
    border-left: 3px solid #ff4655;
    border-radius: 4px; padding: 16px 20px; margin-bottom: 20px;
}
.how-title { font-size: 12px; letter-spacing: 3px; color: #ff4655;
             text-transform: uppercase; margin-bottom: 10px; font-weight: 700; }
.how-steps { display: flex; gap: 24px; flex-wrap: wrap; }
.how-step  { display: flex; align-items: flex-start; gap: 10px; }
.step-num  { background: #ff4655; color: #fff; font-weight: 700; font-size: 11px;
             width: 20px; height: 20px; border-radius: 2px;
             display: flex; align-items: center; justify-content: center;
             flex-shrink: 0; margin-top: 2px; }
.step-txt  { font-size: 13px; color: #aaa; line-height: 1.4; }

/* ── TEAM CARDS ── */
.team-card-win {
    background: linear-gradient(160deg, #1a0810 0%, #12121f 60%);
    border: 2px solid #ff4655;
    box-shadow: 0 0 24px rgba(255,70,85,0.25);
    border-radius: 6px; padding: 28px 20px; text-align: center;
}
.team-card-lose {
    background: #12121f;
    border: 1px solid #1e1e30;
    border-radius: 6px; padding: 28px 20px; text-align: center;
}
.team-name { font-size: 20px; font-weight: 700; letter-spacing: 2px;
             text-transform: uppercase; color: #e0e0e0; margin-bottom: 12px; }
.win-pct-win  { font-size: 72px; font-weight: 700; color: #ff4655;
                font-family: 'Rajdhani', sans-serif; line-height: 1;
                text-shadow: 0 0 30px rgba(255,70,85,0.5); }
.win-pct-lose { font-size: 72px; font-weight: 700; color: #555;
                font-family: 'Rajdhani', sans-serif; line-height: 1; }
.win-label { font-size: 11px; letter-spacing: 3px; color: #666;
             text-transform: uppercase; margin-top: 6px; }

/* ── VS ── */
.vs-block { text-align: center; }
.vs-text  { font-size: 32px; font-weight: 700; color: #333;
            letter-spacing: 4px; text-transform: uppercase; }

/* ── CONFIDENCE BADGES ── */
.conf-high { background: rgba(74,222,128,0.15); color: #4ade80;
             border: 1px solid #4ade80; border-radius: 2px;
             padding: 4px 12px; font-size: 11px; font-weight: 700;
             letter-spacing: 2px; text-transform: uppercase; }
.conf-med  { background: rgba(251,191,36,0.12); color: #fbbf24;
             border: 1px solid #fbbf24; border-radius: 2px;
             padding: 4px 12px; font-size: 11px; font-weight: 700;
             letter-spacing: 2px; text-transform: uppercase; }
.conf-low  { background: rgba(248,113,113,0.12); color: #f87171;
             border: 1px solid #f87171; border-radius: 2px;
             padding: 4px 12px; font-size: 11px; font-weight: 700;
             letter-spacing: 2px; text-transform: uppercase; }

/* ── SCRAPE TERMINAL ── */
.scrape-terminal {
    background: #080810; border: 1px solid #1e1e30;
    border-top: 2px solid #ff4655;
    border-radius: 0 0 6px 6px; padding: 14px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px; color: #44ff88; min-height: 120px;
    max-height: 280px; overflow-y: auto;
    white-space: pre-wrap; line-height: 1.6;
}
.terminal-header {
    background: #12121f; border: 1px solid #1e1e30;
    border-bottom: none; border-radius: 6px 6px 0 0;
    padding: 8px 14px; font-size: 11px;
    letter-spacing: 2px; color: #666; text-transform: uppercase;
    display: flex; gap: 8px; align-items: center;
}
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot-r { background: #ff4655; } .dot-y { background: #fbbf24; } .dot-g { background: #4ade80; }

/* ── STAT SECTION ── */
.section-label {
    font-size: 11px; letter-spacing: 3px; color: #ff4655;
    text-transform: uppercase; font-weight: 700;
    border-bottom: 1px solid #1e1e30; padding-bottom: 8px;
    margin-bottom: 14px; margin-top: 8px;
}
.stat-grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 1px; background: #1e1e30; border-radius: 4px; overflow: hidden;
}
.stat-cell {
    background: #12121f; padding: 10px 14px; text-align: center;
}
.stat-cell-label { font-size: 10px; letter-spacing: 2px; color: #555;
                   text-transform: uppercase; margin-bottom: 4px; }
.stat-cell-val   { font-size: 18px; font-weight: 700; color: #e0e0e0;
                   font-family: 'Share Tech Mono', monospace; }
.stat-cell-val-win { color: #ff4655; }

/* ── PLAYER ROW ── */
.player-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 12px; background: #12121f;
    border-bottom: 1px solid #1a1a28;
}
.player-row:last-child { border-bottom: none; }
.agent-icon { width: 36px; height: 36px; border-radius: 4px;
              object-fit: cover; background: #1e1e30; }
.player-name-row { font-weight: 600; font-size: 14px;
                   letter-spacing: 1px; color: #e0e0e0; }
.player-agent { font-size: 11px; color: #666; text-transform: capitalize; }
.player-stat  { font-family: 'Share Tech Mono', monospace;
                font-size: 13px; color: #aaa; text-align: right; }
.player-stat-label { font-size: 10px; letter-spacing: 1px; color: #444;
                     text-transform: uppercase; }

/* ── WARNING ── */
.warn-box { background: rgba(251,191,36,0.08); border: 1px solid #f59e0b;
            border-left: 3px solid #f59e0b; border-radius: 4px;
            padding: 10px 14px; color: #fbbf24; font-size: 13px; margin-bottom: 8px; }

/* ── FOOTER ── */
.footer { text-align: center; color: #333; font-size: 11px;
          letter-spacing: 2px; text-transform: uppercase;
          padding: 20px 0 0 0; border-top: 1px solid #1e1e30; margin-top: 24px; }

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stDecoration"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ─── AGENT ICONS ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_agent_icons():
    """Fetch agent icon URLs from public Valorant API."""
    try:
        r = requests.get("https://valorant-api.com/v1/agents?isPlayableCharacter=true", timeout=5)
        if r.status_code == 200:
            return {a['displayName'].lower(): a['displayIcon'] for a in r.json()['data']}
    except: pass
    return {}

AGENT_ICONS = load_agent_icons()

def agent_icon_url(agent_name):
    name = (agent_name or "").lower().strip()
    if name in AGENT_ICONS: return AGENT_ICONS[name]
    for k,v in AGENT_ICONS.items():
        if name in k or k in name: return v
    return "https://www.vlr.gg/img/vlr/game/agents/sova.png"  # fallback

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing VCT Vision...")
def load_models():
    xgb_m = xgb.XGBClassifier(); xgb_m.load_model('match_xgb.json')
    cb_m  = CatBoostClassifier(); cb_m.load_model('match_cb.cbm')
    with open('match_iso.pkl','rb')     as f: iso     = pickle.load(f)
    with open('match_feats.pkl','rb')   as f: feats   = pickle.load(f)
    with open('match_weights.pkl','rb') as f: weights = pickle.load(f)
    with open('team_state.pkl','rb')    as f: state   = pickle.load(f)
    return xgb_m, cb_m, iso, feats, weights, state

xgb_m, cb_m, iso, FEATS, WEIGHTS, STATE = load_models()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def clean_agent(n): return n.lower().replace("/","").replace(" ","").strip() if n else "unknown"
def parse_stat(t, d):
    try: return float(str(t).replace('%','').strip()) if str(t).strip() not in ('','-') else float(d)
    except: return float(d)

def fuzzy_lookup(name, teams_dict):
    if name in teams_dict: return name
    low = {k.lower():k for k in teams_dict}
    if name.lower() in low: return low[name.lower()]
    for k in teams_dict:
        if name.lower() in k.lower() or k.lower() in name.lower(): return k
    return None

def get_team_state(name):
    key = fuzzy_lookup(name, STATE['teams'])
    if key: return STATE['teams'][key], key, True
    return {'elo':1000.0,'form':0.5,'recent_results':[],'matches_played':0}, name, False

def get_h2h(t1_key, t2_key):
    h2h = STATE['h2h']
    wins  = h2h.get((t1_key,t2_key), 0)
    total = wins + h2h.get((t2_key,t1_key), 0)
    return wins/total if total>0 else 0.5, wins, total

def confidence_label(p):
    m = abs(p-0.5)
    if m >= 0.15: return "HIGH CONFIDENCE",   "conf-high"
    if m >= 0.08: return "MEDIUM CONFIDENCE", "conf-med"
    return "LOW CONFIDENCE", "conf-low"

def form_icons(results):
    return " ".join(["🟢" if r==1 else "🔴" for r in results[-5:]]) or "—"

def make_driver():
    o = webdriver.ChromeOptions()
    o.add_argument("--headless=new"); o.add_argument("--disable-gpu")
    o.add_argument("--no-sandbox"); o.add_argument("--disable-dev-shm-usage")
    o.add_argument("--disable-blink-features=AutomationControlled")
    o.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36")
    import tempfile
    td = tempfile.mkdtemp(prefix="chrome_vct_app_")
    o.add_argument(f"--user-data-dir={td}")
    o.add_experimental_option("excludeSwitches",["enable-automation"])
    d = webdriver.Chrome(options=o)
    d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
        {"source":"Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})
    d.set_page_load_timeout(40)
    return d, td

def scrape_player_stats(driver, pid, pname, agent):
    dflt = {'acs':200.0,'kd':1.0,'fkpr':0.10,'fdpr':0.10}
    main = driver.current_window_handle
    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        for ts in ["90d","all"]:
            try:
                driver.get(f"https://www.vlr.gg/player/{pid}/{pname}/?timespan={ts}")
                time.sleep(0.6)
                tbls = driver.find_elements(By.CSS_SELECTOR,"table.st-table")
                if not tbls: continue
                IDX = {'rnd':2,'acs':4,'kd':5,'fk':14,'fd':15}
                best = None
                for row in tbls[0].find_elements(By.CSS_SELECTOR,"tbody tr"):
                    cols = row.find_elements(By.TAG_NAME,"td")
                    if len(cols) <= max(IDX.values()): continue
                    try:
                        img = cols[0].find_element(By.TAG_NAME,"img")
                        src = img.get_attribute("src") or ""
                        ag = clean_agent(src.split("/agents/")[-1].replace(".png",""))
                    except: ag = "unknown"
                    rnd = parse_stat(cols[IDX['rnd']].text,1.0) or 1.0
                    s = {'acs': parse_stat(cols[IDX['acs']].text,200.0),
                         'kd':  parse_stat(cols[IDX['kd']].text,1.0),
                         'fkpr':parse_stat(cols[IDX['fk']].text,0.0)/rnd,
                         'fdpr':parse_stat(cols[IDX['fd']].text,0.0)/rnd}
                    if ag == clean_agent(agent): return s
                    if best is None: best = s
                if best: return best
            except: continue
    except: pass
    finally:
        try: driver.close(); driver.switch_to.window(main)
        except: pass
    return dflt

def scrape_match(url, log):
    """Scrape match page. log() updates the terminal UI in real time."""
    driver, tmpdir = make_driver()
    result = {}
    try:
        log("◈  INITIALIZING CHROME DRIVER")
        log(f"◈  NAVIGATING TO MATCH PAGE")
        log(f"    {url}")
        driver.get(url); time.sleep(2)
        wait = WebDriverWait(driver,20)

        try:
            t1 = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR,"div.match-header-link-name.mod-1"))).text.strip()
            t2 = driver.find_element(By.CSS_SELECTOR,"div.match-header-link-name.mod-2").text.strip()
        except TimeoutException:
            return None, "Could not find team names. Is this a valid VLR match page?"

        log(f"◈  TEAMS DETECTED")
        log(f"    TEAM 1 → {t1}")
        log(f"    TEAM 2 → {t2}")
        result['team1'] = t1; result['team2'] = t2

        try:
            parts = driver.title.split("|")
            result['tournament'] = parts[1].strip() if len(parts)>=2 else "Unknown"
            log(f"◈  EVENT → {result['tournament']}")
        except: result['tournament'] = "Unknown"

        try:
            el = driver.find_element(By.CSS_SELECTOR,".moment-tz-convert")
            result['date'] = el.get_attribute("data-utc-ts") or el.text.strip()
        except: result['date'] = ""

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.vm-stats-container")))
        tabs = driver.find_elements(By.CSS_SELECTOR,".vm-stats-gamesnav-item")
        chosen = tabs[0]
        for tb in tabs:
            if tb.get_attribute("data-game-id") not in ("all",None,""): chosen=tb; break
        driver.execute_script("arguments[0].click();", chosen); time.sleep(0.8)
        gid = chosen.get_attribute("data-game-id")
        log(f"◈  READING MAP STATS  [game-id: {gid}]")

        try:
            cont = driver.find_element(By.CSS_SELECTOR,f"div.vm-stats-game[data-game-id='{gid}']")
        except:
            cont = driver.find_element(By.CSS_SELECTOR,"div.vm-stats-game")

        prows = cont.find_elements(By.CSS_SELECTOR,"div.ovw-row")
        t1p, t2p = [], []
        for prow in prows:
            cls = prow.get_attribute("class") or ""
            if "mod-head" in cls: continue
            a = prow.find_elements(By.CSS_SELECTOR,"a[href*='/player/']")
            if not a: continue
            href  = a[0].get_attribute("href")
            pid   = href.split("/player/")[1].split("/")[0]
            try: pname = prow.find_element(By.CSS_SELECTOR,"div.ovw-player-name").text.strip()
            except: pname = a[0].text.strip().split("\n")[0]
            pname = pname or pid
            ag = "unknown"
            for asel in ["div.ovw-cell.mod-agents img","img[src*='/agents/']"]:
                imgs = prow.find_elements(By.CSS_SELECTOR, asel)
                if imgs:
                    src = imgs[0].get_attribute("src") or ""
                    ag = clean_agent(src.split("/agents/")[-1].replace(".png","")) if "/agents/" in src \
                         else clean_agent(imgs[0].get_attribute("title") or "")
                    break
            if len(t1p)<5: t1p.append({'pid':pid,'name':pname,'agent':ag})
            elif len(t2p)<5: t2p.append({'pid':pid,'name':pname,'agent':ag})

        log(f"◈  PLAYERS FOUND → {len(t1p)+len(t2p)}/10")
        log(f"")
        log(f"◈  SCRAPING PLAYER PROFILES")
        log(f"{'─'*44}")

        for plist in [t1p, t2p]:
            for p in plist:
                log(f"  ↳ {p['name'].upper():<16} [{p['agent'].upper()}]")
                s = scrape_player_stats(driver, p['pid'], p['name'], p['agent'])
                p.update(s)
                log(f"      KD {s['kd']:.2f}  ACS {s['acs']:.0f}  "
                    f"FK {s['fkpr']:.3f}  FD {s['fdpr']:.3f}")

        log(f"{'─'*44}")
        log(f"◈  ALL PROFILES LOADED. RUNNING MODEL...")
        result['t1_players'] = t1p; result['t2_players'] = t2p

    except WebDriverException as e:
        return None, f"Browser error: {e}"
    finally:
        try: driver.quit()
        except: pass
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
    return result, None

def build_features(match_data):
    t1,t2 = match_data['team1'],match_data['team2']
    t1s,t1_key,t1_known = get_team_state(t1)
    t2s,t2_key,t2_known = get_team_state(t2)
    h2h_rate,h2h_wins,h2h_total = get_h2h(t1_key,t2_key)
    def avg(players,key): return np.mean([p.get(key,0) for p in players]) if players else 0.5
    t1p = match_data.get('t1_players',[]); t2p = match_data.get('t2_players',[])
    t1_kd=avg(t1p,'kd'); t2_kd=avg(t2p,'kd'); t1_acs=avg(t1p,'acs'); t2_acs=avg(t2p,'acs')
    t1_fk=avg(t1p,'fkpr'); t2_fk=avg(t2p,'fkpr'); t1_fd=avg(t1p,'fdpr'); t2_fd=avg(t2p,'fdpr')
    t1_std=np.std([p.get('kd',1.0) for p in t1p]) if t1p else 0.1
    t2_std=np.std([p.get('kd',1.0) for p in t2p]) if t2p else 0.1
    row = {f:0.0 for f in FEATS}
    row.update({'T1_elo':t1s['elo'],'T2_elo':t2s['elo'],'elo_diff':t1s['elo']-t2s['elo'],
                'T1_form':t1s['form'],'T2_form':t2s['form'],'form_diff':t1s['form']-t2s['form'],
                'T1_h2h':h2h_rate,'T1_Avg_KD':t1_kd,'T2_Avg_KD':t2_kd,'KD_diff':t1_kd-t2_kd,
                'T1_Avg_ACS':t1_acs,'T2_Avg_ACS':t2_acs,'ACS_diff':t1_acs-t2_acs,
                'T1_Avg_FKPR':t1_fk,'T2_Avg_FKPR':t2_fk,'FKPR_diff':t1_fk-t2_fk,
                'T1_Avg_FDPR':t1_fd,'T2_Avg_FDPR':t2_fd,'FDPR_diff':t1_fd-t2_fd,
                'T1_Std_KD':t1_std,'T2_Std_KD':t2_std,'T1_map_wr':0.5,'T2_map_wr':0.5,
                'map_wr_diff':0.0,'Team1_Team_Combo':1,'Team2_Team_Combo':1,
                'Team1_Agent_Combo':0,'Team2_Agent_Combo':0,'is_T1':1})
    X = pd.DataFrame([row])[FEATS].astype(np.float32)
    return X,t1s,t2s,h2h_rate,h2h_wins,h2h_total,t1_known,t2_known

def predict(X):
    xp = xgb_m.predict_proba(X)[:,1]; cp = cb_m.predict_proba(X)[:,1]
    wx,wc = WEIGHTS['xgb'],WEIGHTS['cb']
    ens = (wx*xp + wc*cp)/(wx+wc)
    return float(iso.transform(ens)[0])

# ─── UI ───────────────────────────────────────────────────────────────────────

# HEADER
st.markdown("""
<div class="vct-header">
  <div class="vct-title">VCT <span>VISION</span></div>
  <div class="vct-sub">AI-Powered Match Prediction Engine · Valorant Champions Tour</div>
</div>
""", unsafe_allow_html=True)

# HOW TO USE
st.markdown("""
<div class="how-box">
  <div class="how-title">◈ How To Use</div>
  <div class="how-steps">
    <div class="how-step">
      <div class="step-num">1</div>
      <div class="step-txt">Go to <b>vlr.gg</b> and find any VCT or Challengers match (live, upcoming, or completed)</div>
    </div>
    <div class="how-step">
      <div class="step-num">2</div>
      <div class="step-txt">Copy the full match URL from your browser address bar</div>
    </div>
    <div class="how-step">
      <div class="step-num">3</div>
      <div class="step-txt">Paste it below and click <b>ANALYZE MATCH</b></div>
    </div>
    <div class="how-step">
      <div class="step-num">4</div>
      <div class="step-txt">Wait 30–60 seconds while live player data is fetched, then view the prediction</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# URL INPUT
url_input = st.text_input(
    "VLR.GG MATCH URL",
    placeholder="https://www.vlr.gg/734312/loud-vs-mibr-vct-2026-americas-stage-2-lr2",
    label_visibility="visible"
)

col_btn, _ = st.columns([2, 8])
with col_btn:
    predict_btn = st.button("⚡  ANALYZE MATCH", type="primary", use_container_width=True)

st.divider()

if predict_btn and url_input.strip():
    url = url_input.strip()
    if "vlr.gg" not in url:
        st.error("Please enter a valid vlr.gg match URL.")
    else:
        # ── LIVE TERMINAL ──────────────────────────────────────────────────────
        st.markdown("""
        <div class="terminal-header">
          <span class="dot dot-r"></span>
          <span class="dot dot-y"></span>
          <span class="dot dot-g"></span>
          &nbsp; VCT VISION · DATA ACQUISITION TERMINAL
        </div>""", unsafe_allow_html=True)

        log_box   = st.empty()
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            log_box.markdown(
                f'<div class="scrape-terminal">' +
                '\n'.join(log_lines) +
                '</div>', unsafe_allow_html=True
            )

        log("◈  VCT VISION PREDICTION ENGINE ONLINE")
        log(f"◈  TARGET URL RECEIVED")
        match_data, err = scrape_match(url, log)

        if err or not match_data:
            st.error(f"Scraping failed: {err}")
        else:
            X,t1s,t2s,h2h_rate,h2h_wins,h2h_total,t1_known,t2_known = build_features(match_data)
            prob_t1 = predict(X)
            prob_t2 = 1 - prob_t1
            conf_txt, conf_cls = confidence_label(prob_t1)
            t1 = match_data['team1']; t2 = match_data['team2']
            t1p = match_data.get('t1_players',[]); t2p = match_data.get('t2_players',[])
            winner = t1 if prob_t1 > 0.5 else t2

            log(f"◈  PREDICTION COMPLETE → {winner.upper()} FAVORED ({max(prob_t1,prob_t2)*100:.1f}%)")

            # ── WARNINGS ──────────────────────────────────────────────────────
            st.write("")
            if not t1_known: st.markdown(f'<div class="warn-box">⚠ {t1} not in training database — Elo defaulted to 1000</div>', unsafe_allow_html=True)
            if not t2_known: st.markdown(f'<div class="warn-box">⚠ {t2} not in training database — Elo defaulted to 1000</div>', unsafe_allow_html=True)
            if h2h_total==0: st.markdown(f'<div class="warn-box">⚠ No head-to-head record found between these teams</div>', unsafe_allow_html=True)

            # ── MATCH META ────────────────────────────────────────────────────
            st.markdown(f"""
            <div style='font-size:12px;letter-spacing:2px;color:#666;
                        text-transform:uppercase;text-align:center;margin:12px 0'>
              {match_data.get('tournament','')} &nbsp;·&nbsp; {match_data.get('date','')}
            </div>""", unsafe_allow_html=True)

            # ── WIN PROBABILITY CARDS ─────────────────────────────────────────
            c1, cmid, c2 = st.columns([5,2,5])
            t1_wins = prob_t1 > 0.5
            t1_card = "team-card-win" if t1_wins else "team-card-lose"
            t2_card = "team-card-win" if not t1_wins else "team-card-lose"
            t1_pct_cls = "win-pct-win" if t1_wins else "win-pct-lose"
            t2_pct_cls = "win-pct-win" if not t1_wins else "win-pct-lose"

            with c1:
                st.markdown(f"""
                <div class="{t1_card}">
                  <div class="team-name">{t1}</div>
                  <div class="{t1_pct_cls}">{prob_t1*100:.1f}%</div>
                  <div class="win-label">Win Probability</div>
                  <div style="margin-top:12px;font-size:13px;color:#555">
                    Elo {t1s['elo']:.0f} &nbsp;·&nbsp; Form {t1s['form']:.0%}
                  </div>
                </div>""", unsafe_allow_html=True)

            with cmid:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="vs-block">
                  <div class="vs-text">VS</div>
                  <div style="margin-top:12px">
                    <span class="{conf_cls}">{conf_txt}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="{t2_card}">
                  <div class="team-name">{t2}</div>
                  <div class="{t2_pct_cls}">{prob_t2*100:.1f}%</div>
                  <div class="win-label">Win Probability</div>
                  <div style="margin-top:12px;font-size:13px;color:#555">
                    Elo {t2s['elo']:.0f} &nbsp;·&nbsp; Form {t2s['form']:.0%}
                  </div>
                </div>""", unsafe_allow_html=True)

            # PROBABILITY BAR
            st.write("")
            bar_w = int(prob_t1 * 100)
            st.markdown(f"""
            <div style="position:relative;height:6px;background:#1e1e30;border-radius:3px;margin:8px 0 4px 0">
              <div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#ff4655,#ff6b7a);
                          border-radius:3px;transition:width 0.5s"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#555;
                        letter-spacing:1px;text-transform:uppercase">
              <span>{t1}</span><span>{t2}</span>
            </div>""", unsafe_allow_html=True)

            # ── KEY STATS ─────────────────────────────────────────────────────
            st.write("")
            st.markdown('<div class="section-label">◈ Head-to-Head Statistics</div>', unsafe_allow_html=True)

            def avg_p(players, key): return np.mean([p.get(key,0) for p in players]) if players else 0

            def stat_cell(label, v1, v2, higher_is_better=True, fmt=".2f"):
                v1f,v2f = float(v1),float(v2)
                v1_str = f"{v1f:{fmt}}"
                v2_str = f"{v2f:{fmt}}"
                c1_cls = "stat-cell-val-win" if (higher_is_better and v1f>v2f) or (not higher_is_better and v1f<v2f) else "stat-cell-val"
                c2_cls = "stat-cell-val-win" if (higher_is_better and v2f>v1f) or (not higher_is_better and v2f<v1f) else "stat-cell-val"
                return f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:#1e1e30;
                            margin-bottom:2px;border-radius:2px;overflow:hidden">
                  <div style="background:#12121f;padding:10px 14px;text-align:right">
                    <div class="{c1_cls}" style="font-size:18px;font-weight:700;font-family:'Share Tech Mono',monospace">{v1_str}</div>
                  </div>
                  <div style="background:#0d0d14;padding:10px 14px;text-align:center">
                    <div style="font-size:10px;letter-spacing:2px;color:#444;text-transform:uppercase;
                                padding-top:4px">{label}</div>
                  </div>
                  <div style="background:#12121f;padding:10px 14px;text-align:left">
                    <div class="{c2_cls}" style="font-size:18px;font-weight:700;font-family:'Share Tech Mono',monospace">{v2_str}</div>
                  </div>
                </div>"""

            h2h_t1_str = f"{h2h_wins}W / {h2h_total-h2h_wins}L" if h2h_total>0 else "No record"
            h2h_t2_str = f"{h2h_total-h2h_wins}W / {h2h_wins}L" if h2h_total>0 else "No record"

            header_html = f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:#1e1e30;
                        margin-bottom:4px;border-radius:2px;overflow:hidden">
              <div style="background:#0a0a12;padding:8px 14px;text-align:right;
                          font-size:11px;letter-spacing:2px;color:#ff4655;text-transform:uppercase;font-weight:700">
                {t1}</div>
              <div style="background:#0a0a12;padding:8px 14px;text-align:center;
                          font-size:11px;color:#333">STAT</div>
              <div style="background:#0a0a12;padding:8px 14px;text-align:left;
                          font-size:11px;letter-spacing:2px;color:#ff4655;text-transform:uppercase;font-weight:700">
                {t2}</div>
            </div>"""

            stats_html = (header_html +
                stat_cell("ELO RATING",   t1s['elo'], t2s['elo'], fmt=".0f") +
                stat_cell("AVG K/D",      avg_p(t1p,'kd'), avg_p(t2p,'kd')) +
                stat_cell("AVG ACS",      avg_p(t1p,'acs'), avg_p(t2p,'acs'), fmt=".0f") +
                stat_cell("FIRST KILL/R", avg_p(t1p,'fkpr'), avg_p(t2p,'fkpr'), fmt=".3f") +
                stat_cell("FIRST DIE/R",  avg_p(t1p,'fdpr'), avg_p(t2p,'fdpr'), higher_is_better=False, fmt=".3f")
            )
            st.markdown(stats_html, unsafe_allow_html=True)

            # Form + H2H text row
            col_f1, col_h, col_f2 = st.columns(3)
            with col_f1:
                st.markdown(f"<div style='text-align:right;padding:4px 0'>"
                            f"<span style='font-size:12px;color:#555;letter-spacing:1px'>FORM &nbsp;</span>"
                            f"{form_icons(t1s['recent_results'])}</div>", unsafe_allow_html=True)
            with col_h:
                st.markdown(f"<div style='text-align:center;font-size:11px;color:#444;padding:6px 0'>"
                            f"HEAD TO HEAD<br>"
                            f"<span style='color:#aaa;font-weight:700'>{h2h_t1_str} &nbsp;·&nbsp; {h2h_t2_str}</span>"
                            f"</div>", unsafe_allow_html=True)
            with col_f2:
                st.markdown(f"<div style='text-align:left;padding:4px 0'>"
                            f"{form_icons(t2s['recent_results'])}"
                            f"<span style='font-size:12px;color:#555;letter-spacing:1px'> &nbsp;FORM</span>"
                            f"</div>", unsafe_allow_html=True)

            # ── PLAYER BREAKDOWN ──────────────────────────────────────────────
            st.write("")
            st.markdown('<div class="section-label">◈ Player Roster Analysis</div>', unsafe_allow_html=True)
            pc1, pc2 = st.columns(2)

            def render_players(players, team_name, col):
                with col:
                    st.markdown(f"<div style='font-size:11px;letter-spacing:2px;color:#666;"
                                f"text-transform:uppercase;margin-bottom:8px'>{team_name}</div>",
                                unsafe_allow_html=True)
                    if not players:
                        st.markdown("<div style='color:#444;font-size:13px'>No player data available</div>",
                                    unsafe_allow_html=True)
                        return
                    rows_html = '<div style="border:1px solid #1e1e30;border-radius:4px;overflow:hidden">'
                    for p in players:
                        icon = agent_icon_url(p.get('agent',''))
                        kd_col = "#ff4655" if p.get('kd',1.0) >= 1.1 else ("#aaa" if p.get('kd',1.0) >= 0.9 else "#666")
                        rows_html += f"""
                        <div class="player-row">
                          <img src="{icon}" class="agent-icon" onerror="this.style.display='none'" />
                          <div style="flex:1">
                            <div class="player-name-row">{p['name']}</div>
                            <div class="player-agent">{p.get('agent','?').capitalize()}</div>
                          </div>
                          <div style="text-align:right">
                            <div class="player-stat" style="color:{kd_col}">{p.get('kd',1.0):.2f}</div>
                            <div class="player-stat-label">K/D</div>
                          </div>
                          <div style="text-align:right;margin-left:16px">
                            <div class="player-stat">{p.get('acs',200):.0f}</div>
                            <div class="player-stat-label">ACS</div>
                          </div>
                          <div style="text-align:right;margin-left:16px">
                            <div class="player-stat">{p.get('fkpr',0.1):.3f}</div>
                            <div class="player-stat-label">FK/R</div>
                          </div>
                        </div>"""
                    rows_html += '</div>'
                    st.markdown(rows_html, unsafe_allow_html=True)

            render_players(t1p, t1, pc1)
            render_players(t2p, t2, pc2)

            # ── FOOTER ────────────────────────────────────────────────────────
            st.markdown(f"""
            <div class="footer">
              VCT Vision &nbsp;·&nbsp; XGBoost + CatBoost Ensemble
              &nbsp;·&nbsp; 61.3% Overall · 67.9% High-Confidence
              &nbsp;·&nbsp; 13,522 Matches Training Data
              &nbsp;·&nbsp; Temporal Evaluation
            </div>""", unsafe_allow_html=True)

elif predict_btn:
    st.warning("Please enter a VLR.gg match URL first.")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ About VCT Vision")
    st.markdown("""
VCT Vision is an AI-powered Valorant match prediction engine built for the Valorant Champions Tour.

**Data sources:** vlr.gg live match data, player career profiles

**Features used:**
- 🏆 Team Elo ratings (computed from 13,500+ matches)
- 📈 Rolling form (last 5 match outcomes)
- ⚔️ Head-to-head historical record
- 🎯 Player career ACS, K/D, FK, FD rates
- 🗺️ Historical map win rates

**Models:** XGBoost + CatBoost soft-voting ensemble with isotonic calibration

**Evaluation:** Temporal split — trained on past, tested on future matches only. Reported accuracy is never inflated by random splitting.
    """)
    st.divider()
    st.markdown("### ◈ Team Elo Lookup")
    search = st.text_input("Search team", placeholder="NRG, Paper Rex, Sentinels...")
    if search and len(search) >= 2:
        matches = [(k,v) for k,v in STATE['teams'].items()
                   if search.lower() in k.lower()][:8]
        if matches:
            for name, s in sorted(matches, key=lambda x:-x[1]['elo']):
                st.markdown(
                    f"**{name}**  \n"
                    f"Elo `{s['elo']:.0f}` · Form `{s['form']:.0%}` · "
                    f"`{s['matches_played']}` matches"
                )
        else:
            st.caption("No team found.")