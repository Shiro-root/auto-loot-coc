"""
main.py - Clash of Clans Auto-Attack Bot (lazy-import, menu-driven)

Run:
    python main.py

Modes:
    1) Full auto                     - attacks forever, auto-deploys
    2) Full auto + storage check     - stops at 24M gold/elixir, auto-deploys
    3) Auto attack (no deploy)       - runs the attack cycle but skips
                                       goblins/heroes/abilities so you can
                                       deploy manually. Waits MANUAL_DEPLOY_WAIT
                                       seconds then proceeds to end battle.
    0) Exit
"""

import os
import csv
import re
import time
import json
import random
import signal
import argparse
import subprocess
from datetime import datetime

# =========================================================================
# RUNTIME FLAGS (set from CLI args in main())
# =========================================================================
USE_GPU   = False   # --gpu       try cv2.cuda template matching
FAST_MODE = False   # --fast      stop scale search at first good hit
GPU_READY = False   # set True once a CUDA device is actually confirmed

# =========================================================================
# LAZY MODULE LOADING
# =========================================================================
pyautogui = None
cv2 = None
np = None
gw = None
RapidOCR = None
mss_mod = None


def _ensure_pyautogui():
    global pyautogui
    if pyautogui is None:
        print("  [deps] importing pyautogui...")
        import pyautogui as _pa
        pyautogui = _pa
        pyautogui.PAUSE = 0.02
        pyautogui.FAILSAFE = True


def _ensure_full_deps(need_ocr=False):
    global cv2, np, gw, RapidOCR
    _ensure_pyautogui()

    if cv2 is None:
        print("  [deps] importing opencv-python...")
        import cv2 as _cv
        cv2 = _cv

    if np is None:
        print("  [deps] importing numpy...")
        import numpy as _np
        np = _np

    if gw is None:
        print("  [deps] importing pygetwindow...")
        import pygetwindow as _gw
        gw = _gw

    if need_ocr and RapidOCR is None:
        try:
            print("  [deps] importing rapidocr-onnxruntime (may take ~2s)...")
            from rapidocr_onnxruntime import RapidOCR as _r
            RapidOCR = _r
        except ImportError:
            print("\n  [!] rapidocr-onnxruntime not installed.")
            print("      pip install rapidocr-onnxruntime")
            print("      Falling back to mode without storage check.\n")
            return False

    _ensure_mss()
    _check_gpu()
    return True


def _ensure_mss():
    """mss grabs frames straight from the OS compositor - typically 3-8x
    faster than pyautogui.screenshot()/PIL ImageGrab, which matters a lot
    when we're polling for an icon every SCAN_INTERVAL seconds."""
    global mss_mod
    if mss_mod is None:
        try:
            import mss as _mss
            mss_mod = _mss
            print("  [deps] mss available -> using fast screen capture")
        except ImportError:
            mss_mod = False
            print("  [deps] mss not installed (pip install mss) -> "
                  "falling back to pyautogui.screenshot (slower)")
    return mss_mod


def _check_gpu():
    """Confirms an OpenCV build with CUDA + an actual CUDA device are both
    present. --gpu only does anything useful if both are true."""
    global GPU_READY
    if not USE_GPU:
        return False
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            GPU_READY = True
            dev_name = ""
            try:
                cv2.cuda.printCudaDeviceInfo(0)
            except Exception:
                pass
            print(f"  [gpu] {count} CUDA device(s) detected -> GPU template matching ON")
        else:
            print("  [gpu] --gpu was passed but no CUDA device was found "
                  "(need an NVIDIA GPU + opencv-contrib-python built with CUDA). "
                  "Falling back to CPU.")
    except AttributeError:
        print("  [gpu] --gpu was passed but this OpenCV build has no cv2.cuda module "
              "(pip install opencv-contrib-python built with CUDA support, or use the "
              "CPU build). Falling back to CPU.")
    return GPU_READY


# =========================================================================
# PATHS + CONFIG
# =========================================================================
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE   = os.path.join(ASSET_DIR, "config.json")
LOG_FILE      = os.path.join(ASSET_DIR, "bot_log.csv")
LOOT_LOG_FILE = os.path.join(ASSET_DIR, "loot_log.csv")
REGIONS_FILE  = os.path.join(ASSET_DIR, "resource_regions.json")

ASSETS = {
    "attack_menu":      "attack_btn_menu_1.png",
    "find_match":       "find_match_btn_2.png",
    "attack_go":        "attack_btn_go_3.png",
    "dragon":           "dragon_click_4.png",
    "end_battle":       "end_battlel_5.png",
    "surrender_dialog": "surrender_okay_6.png",
    "return_home":      "return_home_7.png",
    # --- optional: auto wall upgrade (mode 5) ---
    "wall_tile":        "wall_tile_8.png",
    "wall_use_gold":    "wall_use_gold_10.png",
    "wall_use_elixir":  "wall_use_elixir_11.png",
    "wall_confirm":     "wall_confirm_12.png",
}
WALL_ASSET_KEYS = ["wall_tile", "wall_use_gold", "wall_use_elixir"]   # required
# wall_confirm_12.png is OPTIONAL: only used if the file exists (game may not ask)
ASSET_PATHS = {k: os.path.join(ASSET_DIR, v) for k, v in ASSETS.items()}

# Search area per asset, as fractions (x0, y0, x1, y1) of the game window.
# Needed because the wall icons (coin / elixir drop) also appear in the
# top-right resource bar and the Next button; without a ROI the bot would
# match those instead of the real Upgrade buttons.
ASSET_ROI = {
    "wall_tile":       (0.08, 0.12, 0.92, 0.78),   # village area only
    "wall_use_gold":   (0.10, 0.78, 0.90, 1.00),   # bottom action bar
    "wall_use_elixir": (0.10, 0.78, 0.90, 1.00),
    "wall_confirm":    (0.20, 0.20, 0.80, 0.85),   # centre dialog
}

DEFAULT_CONFIDENCE = {
    "attack_menu":      0.82,
    "find_match":       0.72,
    "attack_go":        0.82,
    "dragon":           0.85,
    "end_battle":       0.68,
    "surrender_dialog": 0.90,
    "return_home":      0.82,
    "wall_tile":        0.90,
    "wall_use_gold":    0.85,
    "wall_use_elixir":  0.85,
    "wall_confirm":     0.88,
}

DEFAULT_SCALES = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                  1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg.setdefault("confidence", dict(DEFAULT_CONFIDENCE))
    cfg.setdefault("scales", list(DEFAULT_SCALES))
    for k, v in DEFAULT_CONFIDENCE.items():
        cfg["confidence"].setdefault(k, v)
    save_config(cfg)
    return cfg


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# =========================================================================
# CONSTANTS
# =========================================================================
WINDOW_TITLE_KEYWORD = "Clash of Clans"
GAME_EXE_PATH = r"C:\Program Files\Google\Play Games\current\emulator\crosvm.exe"

MATCHMAKING_TIMEOUT = 60
STEP_TIMEOUT        = 12
SCAN_INTERVAL       = 0.05
CLICK_MOVE_DURATION = 0.02
POST_CLICK_PAUSE    = 0.25
BATTLE_END_WAIT     = 6.0
BASE_PREVIEW_WAIT   = 2.0
MANUAL_DEPLOY_WAIT  = 60.0   # option 3: how long to wait for your manual deploy

FALLBACK_MARGIN       = 0.05
SURRENDER_OKAY_OFFSET = (0.74, 0.83)

ZOOM_OUT_TIMES  = 0   # layar 1920x1080: base sudah terlihat penuh
ZOOM_STEP_DELAY = 0.3

MAX_GOLD   = 24_000_000
MAX_ELIXIR = 24_000_000
SANITY_MAX = 100_000_000

SAVE_DEBUG = False
DEBUG_DIR  = os.path.join(ASSET_DIR, "debug_crops")

DEFAULT_REGIONS = {
    "gold":        [1053, 242, 190, 42],
    "elixir":      [1054, 294, 193, 39],
    "dark_elixir": [1097, 350, 146, 32],
    "gems":        [1099, 395, 143, 40],
}

# --- deploy plan: DRAGON x8 -> siege (Z) -> heroes (Q W E R) -> spells (A, S) ---
DRAGON_COUNT        = 8        # exactly 8 dragons
DRAGON_TAP_DELAY    = 0.12
DRAGON_SPOT_JITTER  = 8        # px jitter around the chosen spot

DRAGON_BUTTON = (325, 995)     # card slot "1"
SIEGE_BUTTON  = (460, 995)     # card slot "Z"
SPELL_A_BUTTON = (1090, 995)   # card slot "A"  (x5)
SPELL_S_BUTTON = (1228, 995)   # card slot "S"  (x3)
SPELL_A_COUNT = 5
SPELL_S_COUNT = 0              # set to 3 if you also want the S spells used
SPELL_TAP_DELAY = 0.25
BASE_CENTER = (960, 470)       # spells land between the deploy spot and this point

# Surrender when the battle clock shows this many seconds remaining.
# The 3:00 clock starts at the first deploy tap. 120 = surrender at "2m 00s".
BATTLE_TOTAL_SECONDS     = 180
SURRENDER_AT_REMAINING   = 120
_battle_t0 = None

DEPLOY_MIN_X = 190
DEPLOY_MAX_X = 1850
DEPLOY_MIN_Y = 110
DEPLOY_MAX_Y = 800

GOBLIN_LOOP = [
    (1700, 512), (1632, 559), (1575, 599), (1525, 634),
    (1480, 665), (1439, 694), (1400, 722), (1362, 748),
    (1326, 774), (1289, 800), (614, 800), (578, 774),
    (541, 748), (503, 722), (464, 694), (423, 665),
    (378, 634), (328, 599), (271, 559), (204, 512),
    (271, 464), (328, 424), (378, 389), (423, 358),
    (464, 329), (503, 301), (541, 275), (578, 249),
    (614, 223), (652, 197), (691, 169), (732, 140),
    (1171, 140), (1212, 169), (1251, 197), (1289, 223),
    (1326, 249), (1362, 275), (1400, 301), (1439, 329),
    (1480, 358), (1525, 389), (1575, 424), (1632, 464),
]

HERO_BUTTONS = [
    (595, 995), (716, 995), (837, 995), (958, 995),
]
HERO_TAPS_PER_SPOT = 2
HERO_SELECT_DELAY  = 0.20
HERO_TAP_DELAY     = 0.12
HERO_BETWEEN_DELAY = 0.25
HERO_ABILITY_DELAY = 0.20
HERO_ABILITY_WAIT  = 6.0


# =========================================================================
# HUMANIZE (anti-pattern): random delays, click jitter, breaks
# =========================================================================
HUMANIZE = True                 # --no-human to disable
HUMAN_CLICK_SPREAD   = 0.28     # click lands inside +-28% of button size (gaussian)
HUMAN_REACTION       = (0.12, 0.55)   # delay between "icon found" and click
HUMAN_MOVE_DUR       = (0.08, 0.30)   # mouse travel time
HUMAN_HOLD           = (0.04, 0.13)   # mouse button held down time
HUMAN_CYCLE_GAP      = (1.5, 6.0)     # pause between attacks
HUMAN_BATTLE_JITTER  = (-1.0, 3.0)    # extra +- seconds on BATTLE_END_WAIT
HUMAN_BREAK_EVERY    = (6, 14)        # short break after N attacks
HUMAN_BREAK_LEN      = (45, 240)      # seconds
HUMAN_LONG_EVERY     = (35, 70)       # long break after N attacks
HUMAN_LONG_LEN       = (600, 1800)    # seconds

_cycles_since_break = 0
_cycles_since_long  = 0
_next_break_at = random.randint(*HUMAN_BREAK_EVERY)
_next_long_at  = random.randint(*HUMAN_LONG_EVERY)


def hsleep(base, spread=0.35):
    """sleep(base) with +-spread random variation (exact when HUMANIZE off)."""
    if not HUMANIZE:
        time.sleep(base)
        return
    time.sleep(max(0.01, base * random.uniform(1 - spread, 1 + spread)))


def human_pick_point(match):
    """Click somewhere inside the button (gaussian around center), not the exact pixel."""
    if "ax" in match:      # explicit aim point (e.g. the wall itself)
        ax, ay = match["ax"], match["ay"]
        if not HUMANIZE:
            return ax, ay
        return ax + int(max(-3, min(3, random.gauss(0, 1.5)))), \
               ay + int(max(-3, min(3, random.gauss(0, 1.5))))
    cx, cy = match["x"] + match["w"] // 2, match["y"] + match["h"] // 2
    if not HUMANIZE:
        return cx, cy
    lim_x, lim_y = match["w"] * 0.40, match["h"] * 0.40
    dx = max(-lim_x, min(lim_x, random.gauss(0, match["w"] * HUMAN_CLICK_SPREAD / 2)))
    dy = max(-lim_y, min(lim_y, random.gauss(0, match["h"] * HUMAN_CLICK_SPREAD / 2)))
    return int(cx + dx), int(cy + dy)


def human_click(x, y):
    if not HUMANIZE:
        pyautogui.moveTo(x, y, duration=CLICK_MOVE_DURATION)
        pyautogui.click()
        return
    tween = random.choice([pyautogui.easeOutQuad, pyautogui.easeInOutQuad,
                           pyautogui.easeOutCubic])
    pyautogui.moveTo(x, y, duration=random.uniform(*HUMAN_MOVE_DUR), tween=tween)
    time.sleep(random.uniform(0.02, 0.09))
    pyautogui.mouseDown()
    time.sleep(random.uniform(*HUMAN_HOLD))
    pyautogui.mouseUp()


def deploy_tap(x, y):
    """Fast tap used while deploying (still with random hold/travel time)."""
    if not HUMANIZE:
        pyautogui.moveTo(x, y)
        pyautogui.click()
        return
    pyautogui.moveTo(x, y, duration=random.uniform(0.01, 0.06))
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.02, 0.07))
    pyautogui.mouseUp()


def click_match(region, match):
    if HUMANIZE:
        time.sleep(random.uniform(*HUMAN_REACTION))
    cx, cy = human_pick_point(match)
    click_region_point(region, cx, cy)


def human_cycle_pause():
    """Called between attacks. Random gap + occasional short/long breaks."""
    global _cycles_since_break, _cycles_since_long, _next_break_at, _next_long_at
    if not HUMANIZE:
        time.sleep(1.5)
        return
    _cycles_since_break += 1
    _cycles_since_long += 1

    if _cycles_since_long >= _next_long_at:
        secs = random.uniform(*HUMAN_LONG_LEN)
        print(f"\n[human] long break {secs/60:.1f} min (Ctrl+C to cancel)...")
        time.sleep(secs)
        _cycles_since_long = _cycles_since_break = 0
        _next_long_at = random.randint(*HUMAN_LONG_EVERY)
        _next_break_at = random.randint(*HUMAN_BREAK_EVERY)
    elif _cycles_since_break >= _next_break_at:
        secs = random.uniform(*HUMAN_BREAK_LEN)
        print(f"\n[human] short break {secs:.0f}s...")
        time.sleep(secs)
        _cycles_since_break = 0
        _next_break_at = random.randint(*HUMAN_BREAK_EVERY)
    else:
        time.sleep(random.uniform(*HUMAN_CYCLE_GAP))


# =========================================================================
# AUTO UPGRADE WALL (mode 5) - runs when gold or elixir is full
# =========================================================================
WALL_MAX_ROUNDS = 6      # max "upgrade" repetitions per time storage is full


def wall_assets_missing():
    return [ASSETS[k] for k in WALL_ASSET_KEYS if not os.path.exists(ASSET_PATHS[k])]


def _close_dialogs(win=None):
    """Tap an empty grass spot at the left edge to dismiss menus/selection.
    (NOT Esc: on the village screen Esc/Back opens the 'quit game?' dialog.)"""
    try:
        win = win or find_game_window()
        region = get_region(win)
        click_region_point(region, int(region[2] * 0.03), int(region[3] * 0.45))
    except Exception:
        pass
    hsleep(0.6)


def wall_detect_only(win, cfg):
    """Prints wall asset scores for the CURRENT screen (no clicks)."""
    region = get_region(win)
    screen = capture(region)
    print("\n--- wall detection on current screen (no clicks) ---")
    for key in ("wall_tile", "wall_use_gold", "wall_use_elixir", "wall_confirm"):
        if not os.path.exists(ASSET_PATHS[key]):
            print(f"  {key:<16} (template file missing)")
            continue
        m = match_template(screen, key, cfg, thorough=True)
        if "error" in m:
            print(f"  {key:<16} ERROR {m['error']}")
            continue
        print(f"  {key:<16} {'FOUND' if m['found'] else 'miss '} "
              f"score={m['score']:.3f} (need {m['threshold']:.2f}) "
              f"scale={m['scale']:.2f} click rel=({m.get('ax', m['x'] + m['w'] // 2)},"
              f"{m.get('ay', m['y'] + m['h'] // 2)})")


def _pay_order(res):
    """Which upgrade button(s) to try, most-full resource first."""
    gold = (res or {}).get("gold") or 0
    elixir = (res or {}).get("elixir") or 0
    order = []
    if gold >= MAX_GOLD or gold >= elixir:
        order = ["wall_use_gold", "wall_use_elixir"]
    else:
        order = ["wall_use_elixir", "wall_use_gold"]
    if elixir >= MAX_ELIXIR and gold < MAX_GOLD:
        order = ["wall_use_elixir", "wall_use_gold"]
    return order


def upgrade_one_wall(win, cfg, res):
    """tap wall -> tap Upgrade (gold or elixir) -> optional Confirm.
       Fallback-click is disabled: these clicks spend resources."""
    if not wait_and_click(win, cfg, "wall_tile", "Wall on base", timeout=8,
                          allow_fallback=False):
        print("  ! no wall found on the village screen")
        return False
    hsleep(0.9)     # let the bottom action bar slide in

    paid = False
    for key in _pay_order(res):
        label = "Upgrade with GOLD" if key == "wall_use_gold" else "Upgrade with ELIXIR"
        if wait_and_click(win, cfg, key, label, timeout=4, allow_fallback=False):
            paid = True
            break
    if not paid:
        print("  ! upgrade button not found (wall maxed / already upgrading?)")
        _close_dialogs(win)
        return False
    hsleep(0.8)

    if os.path.exists(ASSET_PATHS["wall_confirm"]):
        # confirm is optional: the game only asks in some cases
        if wait_and_click(win, cfg, "wall_confirm", "Confirm upgrade", timeout=3,
                          allow_fallback=False):
            hsleep(0.8)
    return True


def upgrade_walls(win, cfg, regions, res):
    """Must be on the home village. Repeats until neither gold nor elixir
       is full anymore (max WALL_MAX_ROUNDS)."""
    print("\n=== Storage full -> upgrading walls ===")
    for rnd in range(1, WALL_MAX_ROUNDS + 1):
        print(f"--- wall round {rnd}/{WALL_MAX_ROUNDS} ---")
        if not upgrade_one_wall(win, cfg, res):
            _close_dialogs(win)
            return False
        hsleep(1.5)
        res = read_resources(regions, verbose=False)
        print_resources(f"after wall round {rnd}", res)
        if res and not is_storage_full(res):
            print("=== Storage no longer full, walls done ===\n")
            return True

    print("  ! still full after max wall rounds")
    return not is_storage_full(res)


def mode_wall_test(cfg):
    print("\nWALL UPGRADE TEST (be on the home village, storage can be anything)")
    print("  a) detect only (no clicks, shows scores on current screen)")
    print("  b) upgrade ONE wall for real (spends resources!)")
    try:
        c = input("Choose [a/b]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    missing = wall_assets_missing()
    if missing:
        print("[!] Missing templates:", ", ".join(missing))
        return
    win = ensure_game_running()
    if c == "a":
        wall_detect_only(win, cfg)
    elif c == "b":
        ok = upgrade_one_wall(win, cfg, None)
        print("RESULT:", "wall upgrade started" if ok else "FAILED (see messages above)")


# =========================================================================
# MENU
# =========================================================================
def print_menu():
    print()
    print("=" * 60)
    print("  Clash of Clans Auto-Attack Bot")
    print("=" * 60)
    print("  1) Full auto")
    print("     attacks forever, auto-deploys 8 dragons + siege + heroes + spells")
    print()
    print("  2) Full auto + storage check")
    print("     same as 1, but stops when gold or elixir reaches 24M")
    print()
    print("  3) Auto attack only (NO deploy)")
    print("     clicks Attack / Find Match / Attack!, then waits")
    print(f"     {MANUAL_DEPLOY_WAIT:.0f}s for YOU to deploy manually,")
    print("     then clicks End Battle / Surrender / Return Home")
    print()
    print("  4) Test / calibrate asset detection")
    print("     no clicking - repeatedly scans for every icon and prints")
    print("     live score + timing per asset, so you can tune confidence")
    print("     thresholds and check detection speed/accuracy")
    print()
    print("  5) Full auto + storage check + AUTO UPGRADE WALL")
    print("     same as 2, but when gold/elixir is full it upgrades walls")
    print("     (needs wall_*.png templates), then keeps attacking")
    print()
    print("  6) Wall upgrade test (detect only / upgrade 1 wall now)")
    print()
    print("  0) Exit")
    print("=" * 60)
    print(f"  gpu={'ON' if (USE_GPU and GPU_READY) else ('requested, unavailable' if USE_GPU else 'off')}"
          f"   fast={'ON' if FAST_MODE else 'off'}"
          f"   capture={'mss' if mss_mod else 'pyautogui'}")


# =========================================================================
# LOGGING
# =========================================================================
def ensure_log_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "step", "asset", "found", "score", "threshold",
                "scale", "rel_x", "rel_y", "abs_x", "abs_y",
                "window_left", "window_top", "window_w", "window_h", "note",
            ])


def log_attempt(step, asset_key, match, region, note=""):
    ts = datetime.now().isoformat()
    if "error" in match:
        row = [ts, step, asset_key, False, "", "", "", "", "", "", "",
               region[0], region[1], region[2], region[3], match["error"]]
        print(f"[log] {ts} | {step:<18} | {asset_key:<16} | ERROR: {match['error']}")
    else:
        cx, cy = match.get("ax", match["x"] + match["w"] // 2), \
                 match.get("ay", match["y"] + match["h"] // 2)
        abs_x, abs_y = region[0] + cx, region[1] + cy
        row = [ts, step, asset_key, match["found"], f"{match['score']:.4f}",
               f"{match['threshold']:.2f}", f"{match['scale']:.2f}",
               cx, cy, abs_x, abs_y,
               region[0], region[1], region[2], region[3], note]
        status = "FOUND" if match["found"] else "miss "
        print(f"[log] {ts} | {step:<18} | {asset_key:<16} | {status} | "
              f"score={match['score']:.3f} (need {match['threshold']:.2f}) | "
              f"scale={match['scale']:.2f} | rel=({cx},{cy}) abs=({abs_x},{abs_y})")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def log_loot_row(before, after, gained):
    exists = os.path.exists(LOOT_LOG_FILE)

    def v(d, k):
        val = d.get(k)
        return "" if val is None else val

    with open(LOOT_LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "timestamp",
                "gold_before", "elixir_before", "de_before", "gems_before",
                "gold_after",  "elixir_after",  "de_after",  "gems_after",
                "gold_gained", "elixir_gained", "de_gained", "gems_gained",
            ])
        w.writerow([
            datetime.now().isoformat(),
            v(before, "gold"),        v(before, "elixir"),
            v(before, "dark_elixir"), v(before, "gems"),
            v(after, "gold"),         v(after, "elixir"),
            v(after, "dark_elixir"),  v(after, "gems"),
            v(gained, "gold"),        v(gained, "elixir"),
            v(gained, "dark_elixir"), v(gained, "gems"),
        ])


def print_resources(tag, res):
    if not res:
        print(f"[loot] {tag}  (read failed)")
        return

    def fmt(v):
        return f"{v:>11,}" if v is not None else "        n/a"

    print(f"[loot] {tag}  gold={fmt(res.get('gold'))}  "
          f"elixir={fmt(res.get('elixir'))}  "
          f"de={fmt(res.get('dark_elixir'))}  "
          f"gems={fmt(res.get('gems'))}")


# =========================================================================
# OCR
# =========================================================================
def load_regions():
    if os.path.exists(REGIONS_FILE):
        try:
            with open(REGIONS_FILE) as f:
                data = json.load(f)
            if data:
                return data
        except Exception as e:
            print(f"  ! could not read {REGIONS_FILE}: {e}, using defaults")
    return dict(DEFAULT_REGIONS)


def _preprocess(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    h, w = thresh.shape
    thresh = cv2.resize(thresh, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    return thresh


def _parse_number(text):
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


_ocr_engine = None


def read_resources(regions, verbose=True):
    win = find_game_window()
    if win is None:
        return None

    rect = get_region(win)
    shot = pyautogui.screenshot(region=rect)
    img = np.array(shot)

    if SAVE_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, "_window.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    engine = _ocr_engine
    out = {}

    for name in ("gold", "elixir", "dark_elixir", "gems"):
        crop_rect = regions.get(name)
        if not crop_rect or crop_rect[2] < 5 or crop_rect[3] < 5:
            out[name] = None
            continue

        l, t, w, h = crop_rect
        crop = img[t:t + h, l:l + w]
        if crop.size == 0:
            out[name] = None
            continue

        if SAVE_DEBUG:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        processed = _preprocess(crop)
        try:
            result, _ = engine(processed)
        except Exception:
            out[name] = None
            continue

        if not result:
            out[name] = None
            if verbose:
                print(f"    [{name}] raw='' -> None")
            continue

        text = " ".join(item[1] for item in result)
        value = _parse_number(text)

        if value is not None and value > SANITY_MAX:
            if verbose:
                print(f"    [{name}] raw={text!r} -> {value}  ** ignored **")
            value = None
        elif verbose:
            print(f"    [{name}] raw={text!r} -> {value}")

        out[name] = value

    return out


# =========================================================================
# WINDOW + MATCHING
# =========================================================================
def find_game_window():
    for w in gw.getAllWindows():
        if WINDOW_TITLE_KEYWORD.lower() in w.title.lower() and w.width > 0 and w.height > 0:
            return w
    return None


def ensure_game_running():
    win = find_game_window()
    if win:
        return win
    print(f"No window matching '{WINDOW_TITLE_KEYWORD}' found. Launching...")
    if os.path.exists(GAME_EXE_PATH):
        subprocess.Popen([GAME_EXE_PATH])
    else:
        print(f"WARNING: {GAME_EXE_PATH} not found. Start the game manually.")
    for _ in range(60):
        time.sleep(1)
        win = find_game_window()
        if win:
            print("Game window detected.")
            return win
    raise RuntimeError("Timed out waiting for the game window.")


_ACTIVATE_EVERY = 2.0   # seconds between win.activate() calls
_last_activate_ts = 0.0


def get_region(win, force_activate=False):
    """Returns (left, top, width, height) for the game window.

    win.activate() is a real OS call (window-manager round trip) and the
    old code did it on *every* poll inside the scan loops (every
    SCAN_INTERVAL == 0.05s). That's ~20 activations/sec doing nothing
    useful once the window is already focused, and it's the single
    biggest reason icon lookups felt slow. Now we only re-activate every
    _ACTIVATE_EVERY seconds (or when explicitly asked / window looks
    invalid), and just re-read left/top/width/height (cheap, no IPC)
    every other call.
    """
    global _last_activate_ts
    win = find_game_window() or win
    now = time.time()

    try:
        if win.isMinimized:
            win.restore()
            time.sleep(0.2)
            force_activate = True
        if force_activate or (now - _last_activate_ts) >= _ACTIVATE_EVERY:
            win.activate()
            _last_activate_ts = now
    except Exception:
        pass

    left, top, width, height = win.left, win.top, win.width, win.height

    if left < -5000 or top < -5000 or width <= 0 or height <= 0:
        print("  ! window rect invalid, retrying...")
        time.sleep(0.6)
        win2 = find_game_window()
        if win2:
            try:
                if win2.isMinimized:
                    win2.restore()
                    time.sleep(0.2)
                win2.activate()
            except Exception:
                pass
            left, top, width, height = win2.left, win2.top, win2.width, win2.height

    return (left, top, width, height)


def click_region_point(region, rel_x, rel_y):
    abs_x = region[0] + rel_x
    abs_y = region[1] + rel_y

    screen_w, screen_h = pyautogui.size()
    safe_x = min(max(abs_x, 2), screen_w - 2)
    safe_y = min(max(abs_y, 2), screen_h - 2)

    try:
        human_click(safe_x, safe_y)
    except pyautogui.FailSafeException:
        print(f"  ! fail-safe on click ({safe_x},{safe_y})")
    except Exception as e:
        print(f"  ! click error ({safe_x},{safe_y}): {e}")


_mss_thread_local = {}


def capture(region):
    left, top, width, height = region
    if mss_mod:
        # mss instances aren't thread-safe to share, but we're single
        # threaded here; keep one instance alive instead of re-creating
        # it (which is the expensive part) on every single poll.
        sct = _mss_thread_local.get("sct")
        if sct is None:
            sct = mss_mod.mss()
            _mss_thread_local["sct"] = sct
        monitor = {"left": left, "top": top, "width": width, "height": height}
        raw = sct.grab(monitor)
        # mss gives BGRA, already in BGR-first order - just drop alpha.
        return np.array(raw)[:, :, :3]

    shot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


_TEMPLATE_CACHE = {}          # asset_key -> BGR ndarray (avoid re-reading PNGs from disk)
_GPU_TEMPLATE_CACHE = {}      # asset_key -> cv2.cuda_GpuMat of the template
_gpu_screen_mat = None        # reused GpuMat for the screenshot upload


def _load_template(asset_key):
    if asset_key not in _TEMPLATE_CACHE:
        path = ASSET_PATHS[asset_key]
        if not os.path.exists(path):
            _TEMPLATE_CACHE[asset_key] = None
            return None, f"missing asset: {path}"
        img = cv2.imread(path)
        if img is None:
            _TEMPLATE_CACHE[asset_key] = None
            return None, f"could not read: {path}"
        _TEMPLATE_CACHE[asset_key] = img
    img = _TEMPLATE_CACHE[asset_key]
    if img is None:
        return None, f"missing/unreadable asset: {asset_key}"
    return img, None


_ANCHOR_CACHE = {}


def _wall_anchor(asset_key):
    """Click point INSIDE the template as fractions (fx, fy).
    wall_tile is a crop of a diagonal wall piece with lots of grass around it,
    so its centre is grass. We aim at the light-teal wall pixels instead.
    Falls back to the centre (0.5, 0.5) if none are found."""
    if asset_key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[asset_key]
    anchor = (0.5, 0.5)
    tmpl, err = _load_template(asset_key)
    if tmpl is not None:
        hsv = cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        mask = (H >= 80) & (H <= 110) & (V > 130)
        ys, xs = np.nonzero(mask)
        if len(xs) >= 20:
            th, tw = tmpl.shape[:2]
            anchor = (float(xs.mean()) / tw, float(ys.mean()) / th)
    _ANCHOR_CACHE[asset_key] = anchor
    return anchor


def match_template(screen_bgr, asset_key, cfg, thorough=None):
    """Finds asset_key inside screen_bgr across cfg['scales'].

    thorough=None uses the global FAST_MODE flag (inverted): by default
    (FAST_MODE off) behaves like the original - scans every scale so the
    reported score is the true best score, which is what you want while
    tuning confidence thresholds. With FAST_MODE on, or thorough=False,
    it stops at the first scale that clears the threshold, which is a
    big speedup once thresholds are already dialed in (the common case
    once you're just running the bot).
    """
    template_orig, err = _load_template(asset_key)
    if err:
        return {"error": err}

    threshold = cfg["confidence"].get(asset_key, 0.82)
    full_h, full_w = screen_bgr.shape[:2]
    off_x = off_y = 0
    roi = ASSET_ROI.get(asset_key)
    if roi:
        off_x, off_y = int(full_w * roi[0]), int(full_h * roi[1])
        screen_bgr = screen_bgr[off_y:int(full_h * roi[3]), off_x:int(full_w * roi[2])]
    screen_h, screen_w = screen_bgr.shape[:2]

    scales = list(cfg["scales"])
    pref = cfg.get("preferred_scale", {}).get(asset_key)
    if pref is not None and pref in scales:
        scales.remove(pref)
        scales.insert(0, pref)

    stop_early = FAST_MODE if thorough is None else (not thorough)

    if GPU_READY and USE_GPU:
        best = _match_template_gpu(screen_bgr, template_orig, asset_key,
                                    scales, screen_w, screen_h,
                                    threshold, stop_early)
    else:
        best = _match_template_cpu(screen_bgr, template_orig, scales,
                                    screen_w, screen_h, threshold, stop_early)

    if best is None:
        return {"error": "template larger than screen at every scale"}

    best["x"] += off_x
    best["y"] += off_y
    if asset_key == "wall_tile":
        fx, fy = _wall_anchor(asset_key)
        best["ax"] = best["x"] + int(fx * best["w"])
        best["ay"] = best["y"] + int(fy * best["h"])
    best["threshold"] = threshold
    best["found"] = best["score"] >= threshold
    return best


def _match_template_cpu(screen_bgr, template_orig, scales, screen_w, screen_h,
                         threshold, stop_early):
    best = None
    h0, w0 = template_orig.shape[:2]
    for scale in scales:
        w, h = int(w0 * scale), int(h0 * scale)
        if w < 5 or h < 5 or w > screen_w or h > screen_h:
            continue
        template = cv2.resize(template_orig, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best["score"]:
            best = {"x": max_loc[0], "y": max_loc[1], "w": w, "h": h,
                    "score": max_val, "scale": scale}
        if stop_early and max_val >= threshold:
            break
    return best


def _match_template_gpu(screen_bgr, template_orig, asset_key, scales,
                         screen_w, screen_h, threshold, stop_early):
    """Same search, run on the GPU via cv2.cuda. Uploading the screenshot
    once per call and reusing GpuMats for the templates is what makes this
    actually faster than CPU - re-uploading per scale would eat the gain."""
    global _gpu_screen_mat
    try:
        if _gpu_screen_mat is None:
            _gpu_screen_mat = cv2.cuda_GpuMat()
        _gpu_screen_mat.upload(screen_bgr)

        h0, w0 = template_orig.shape[:2]
        best = None
        for scale in scales:
            w, h = int(w0 * scale), int(h0 * scale)
            if w < 5 or h < 5 or w > screen_w or h > screen_h:
                continue

            cache_key = (asset_key, scale)
            tmpl_gpu = _GPU_TEMPLATE_CACHE.get(cache_key)
            if tmpl_gpu is None:
                resized = cv2.resize(template_orig, (w, h), interpolation=cv2.INTER_AREA)
                tmpl_gpu = cv2.cuda_GpuMat()
                tmpl_gpu.upload(resized)
                _GPU_TEMPLATE_CACHE[cache_key] = tmpl_gpu

            matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC3, cv2.TM_CCOEFF_NORMED)
            result_gpu = matcher.match(_gpu_screen_mat, tmpl_gpu)
            result = result_gpu.download()
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if best is None or max_val > best["score"]:
                best = {"x": max_loc[0], "y": max_loc[1], "w": w, "h": h,
                        "score": max_val, "scale": scale}
            if stop_early and max_val >= threshold:
                break
        return best
    except Exception as e:
        print(f"  [gpu] match failed ({e}), falling back to CPU for this frame")
        return _match_template_cpu(screen_bgr, template_orig, scales,
                                    screen_w, screen_h, threshold, stop_early)


def center_of(match):
    return match["x"] + match["w"] // 2, match["y"] + match["h"] // 2


def wait_and_click(win, cfg, asset_key, label, timeout=STEP_TIMEOUT,
                   allow_fallback=True):
    print(f"Looking for '{label}'...")
    start = time.time()
    region = get_region(win)
    attempt = 0
    best = None

    while time.time() - start < timeout:
        attempt += 1
        region = get_region(win)
        screen = capture(region)
        match = match_template(screen, asset_key, cfg)

        if "error" in match:
            print(f"  [{attempt}] {asset_key}: {match['error']}")
        else:
            status = "FOUND" if match["found"] else "miss"
            print(f"  [{attempt}] {asset_key}: {status} score={match['score']:.3f} "
                  f"(need {match['threshold']:.2f}) scale={match['scale']:.2f}")
            if best is None or match["score"] > best["score"]:
                best = match

            if match["found"]:
                log_attempt(label, asset_key, match, region)
                cfg.setdefault("preferred_scale", {})[asset_key] = match["scale"]
                save_config(cfg)
                click_match(region, match)
                return True
        time.sleep(SCAN_INTERVAL)

    if (allow_fallback and best is not None
            and best["score"] >= best["threshold"] - FALLBACK_MARGIN):
        print(f"  -> fallback click (score {best['score']:.3f})")
        log_attempt(label, asset_key, best, region, note="FALLBACK_CLICK")
        click_match(region, best)
        return True

    log_attempt(label, asset_key, best or {"error": "no attempt"}, region, note="TIMEOUT")
    return False


def wait_until_gone(win, cfg, asset_key, label, timeout=MATCHMAKING_TIMEOUT):
    print(f"Waiting for '{label}' to clear...")
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        region = get_region(win)
        screen = capture(region)
        match = match_template(screen, asset_key, cfg)
        if "error" in match:
            return True
        status = "still showing" if match["found"] else "cleared"
        print(f"  [{attempt}] {asset_key}: {status} score={match['score']:.3f} "
              f"(need {match['threshold']:.2f})")
        if not match["found"]:
            return True
        time.sleep(SCAN_INTERVAL)
    print(f"  ! timeout waiting for '{label}' to clear")
    return False


def click_surrender_okay(win, cfg):
    region = get_region(win)
    screen = capture(region)
    match = match_template(screen, "surrender_dialog", cfg)
    log_attempt("surrender_okay", "surrender_dialog", match, region)
    if "error" in match or not match["found"]:
        print("  ! surrender dialog not found")
        return False
    ox = match["x"] + int(match["w"] * SURRENDER_OKAY_OFFSET[0])
    oy = match["y"] + int(match["h"] * SURRENDER_OKAY_OFFSET[1])
    if HUMANIZE:
        time.sleep(random.uniform(*HUMAN_REACTION))
        ox += int(random.gauss(0, match["w"] * 0.04))
        oy += int(random.gauss(0, match["h"] * 0.05))
    click_region_point(region, ox, oy)
    return True


# =========================================================================
# DEPLOY
# =========================================================================
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def zoom_out_and_center(win=None, times=ZOOM_OUT_TIMES):
    print(f"Zooming out ({times} steps) before deploy...")
    if win is not None:
        region = get_region(win)
        cx = region[0] + region[2] // 2
        cy = region[1] + region[3] // 2
    else:
        screen_w, screen_h = pyautogui.size()
        cx, cy = screen_w // 2, screen_h // 2

    try:
        pyautogui.moveTo(cx, cy, duration=0.10)
        time.sleep(0.15)
    except pyautogui.FailSafeException:
        print("  ! fail-safe on move to center")
        return

    for i in range(times):
        try:
            pyautogui.keyDown("ctrl")
            pyautogui.press("-")
            pyautogui.keyUp("ctrl")
        except Exception as e:
            print(f"  ! zoom step {i+1} failed: {e}")
            return
        time.sleep(ZOOM_STEP_DELAY)

    print("  zoom-out done.")


def select_dragon_card(win, cfg, timeout=25):
    print("Selecting dragon troop card...")
    match = {"error": "no attempt"}
    best = None
    region = get_region(win)
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        region = get_region(win)
        screen = capture(region)
        match = match_template(screen, "dragon", cfg)
        if "error" in match:
            print(f"  [{attempt}] dragon: {match['error']}")
        else:
            status = "FOUND" if match["found"] else "miss"
            print(f"  [{attempt}] dragon: {status} score={match['score']:.3f} "
                  f"(need {match['threshold']:.2f}) scale={match['scale']:.2f}")
            if best is None or match["score"] > best["score"]:
                best = match
            if match["found"]:
                break
        time.sleep(SCAN_INTERVAL)

    if "error" not in match and match.get("found"):
        log_attempt("dragon_select", "dragon", match, region)
        click_match(region, match)
        hsleep(0.15)
        return True
    if best is not None and best["score"] >= best["threshold"] - FALLBACK_MARGIN:
        print(f"  -> fallback select dragon (score {best['score']:.3f})")
        log_attempt("dragon_select", "dragon", best, region, note="FALLBACK_CLICK")
        click_match(region, best)
        hsleep(0.15)
        return True
    log_attempt("dragon_select", "dragon", best or match, region, note="TIMEOUT")
    print("  ! could not find dragon card")
    return False


def _jittered(spot, j=DRAGON_SPOT_JITTER):
    return (_clamp(spot[0] + random.randint(-j, j), DEPLOY_MIN_X, DEPLOY_MAX_X),
            _clamp(spot[1] + random.randint(-j, j), DEPLOY_MIN_Y, DEPLOY_MAX_Y))


def deploy_dragons(spot):
    """Exactly DRAGON_COUNT taps at one spot = DRAGON_COUNT dragons."""
    global _battle_t0
    print(f"\n=== Deploying {DRAGON_COUNT} dragons at {spot} ===")
    for i in range(DRAGON_COUNT):
        x, y = _jittered(spot)
        try:
            deploy_tap(x, y)
        except pyautogui.FailSafeException:
            print("  ! fail-safe on dragon tap")
            continue
        if _battle_t0 is None:
            _battle_t0 = time.time()      # battle clock starts at first deploy
        hsleep(DRAGON_TAP_DELAY, 0.4)


def deploy_siege(spot):
    print("\n=== Deploying siege machine (Z) ===")
    try:
        deploy_tap(*SIEGE_BUTTON)
        hsleep(HERO_SELECT_DELAY)
        deploy_tap(*_jittered(spot, 4))
    except pyautogui.FailSafeException:
        print("  ! fail-safe on siege")
    hsleep(HERO_BETWEEN_DELAY)


def deploy_spells(spot):
    """Drop spells along the line from the deploy spot toward the base center."""
    plan = [(SPELL_A_BUTTON, SPELL_A_COUNT, "A"), (SPELL_S_BUTTON, SPELL_S_COUNT, "S")]
    for btn, count, name in plan:
        if count <= 0:
            continue
        print(f"\n=== Spells {name} x{count} ===")
        try:
            deploy_tap(*btn)
        except pyautogui.FailSafeException:
            print(f"  ! fail-safe selecting spell {name}")
            continue
        hsleep(HERO_SELECT_DELAY)
        for i in range(count):
            t = (i + 1) / (count + 1) * 0.7        # fractions along the path
            x = int(spot[0] + (BASE_CENTER[0] - spot[0]) * t)
            y = int(spot[1] + (BASE_CENTER[1] - spot[1]) * t)
            try:
                deploy_tap(*_jittered((x, y), 10))
            except pyautogui.FailSafeException:
                print(f"  ! fail-safe on spell {name}")
                break
            hsleep(SPELL_TAP_DELAY)


def wait_until_surrender_time():
    """Sleep until the battle clock shows SURRENDER_AT_REMAINING seconds left."""
    target = BATTLE_TOTAL_SECONDS - SURRENDER_AT_REMAINING
    elapsed = (time.time() - _battle_t0) if _battle_t0 else 0
    left = max(1.0, target - elapsed)
    if HUMANIZE:
        left = max(1.0, left + random.uniform(*HUMAN_BATTLE_JITTER))
    print(f"Battle clock {elapsed:.0f}s elapsed -> surrender in {left:.0f}s "
          f"(at ~{SURRENDER_AT_REMAINING // 60}m{SURRENDER_AT_REMAINING % 60:02d}s left)")
    try:
        time.sleep(left)
    except KeyboardInterrupt:
        print("  [Ctrl+C] skipping wait...")


def deploy_heroes(spot=None):
    print("\n=== Deploying heroes ===")
    spot = spot or random.choice(GOBLIN_LOOP)
    print(f"  All heroes -> {spot}")
    for num, hero_btn in enumerate(HERO_BUTTONS, 1):
        print(f"  Hero {num}")
        try:
            deploy_tap(*hero_btn)
        except pyautogui.FailSafeException:
            print(f"    ! fail-safe on hero {num} select")
            continue
        hsleep(HERO_SELECT_DELAY)
        for _ in range(HERO_TAPS_PER_SPOT):
            try:
                deploy_tap(*_jittered(spot, 4))
            except pyautogui.FailSafeException:
                print(f"    ! fail-safe on hero {num} tap")
                break
            hsleep(HERO_TAP_DELAY)
        hsleep(HERO_BETWEEN_DELAY)


def activate_hero_abilities():
    print(f"\n=== Activating hero abilities (waiting {HERO_ABILITY_WAIT}s first) ===")
    time.sleep(HERO_ABILITY_WAIT)
    for i, hero_btn in enumerate(HERO_BUTTONS, 1):
        print(f"  Ability {i} -> {hero_btn}")
        try:
            deploy_tap(*hero_btn)
        except pyautogui.FailSafeException:
            print(f"    ! fail-safe on ability {i}")
            continue
        hsleep(HERO_ABILITY_DELAY)


def full_deploy(win, cfg):
    global _battle_t0
    _battle_t0 = None
    print(f"Waiting {BASE_PREVIEW_WAIT}s on base preview...")
    time.sleep(BASE_PREVIEW_WAIT)

    zoom_out_and_center(win, times=ZOOM_OUT_TIMES)

    if not select_dragon_card(win, cfg):
        print("  ! dragon card not found, skipping deploy")
        return False

    spot = random.choice(GOBLIN_LOOP)       # one perimeter spot for the whole army
    deploy_dragons(spot)
    deploy_siege(spot)
    deploy_heroes(spot)
    deploy_spells(spot)
    activate_hero_abilities()

    print("=== Deploy complete ===\n")
    return True


# =========================================================================
# STORAGE CHECK
# =========================================================================
def is_storage_full(res):
    if not res:
        return False
    gold   = res.get("gold")   or 0
    elixir = res.get("elixir") or 0
    return gold >= MAX_GOLD or elixir >= MAX_ELIXIR


# =========================================================================
# ATTACK CYCLE
# =========================================================================
def run_one_attack_cycle(win, cfg, resource_regions, storage_check, auto_deploy,
                         wall_upgrade=False):
    """auto_deploy=True  -> goblins/heroes/abilities (modes 1 & 2)
       auto_deploy=False -> just wait MANUAL_DEPLOY_WAIT, you play (mode 3)"""
    print("\n" + "#" * 60)
    print("# Starting attack cycle")
    print("#" * 60)

    before = None
    if storage_check:
        print("Scanning resources before attack...")
        before = read_resources(resource_regions, verbose=True)
        print_resources("BEFORE", before)

        if is_storage_full(before):
            gold   = (before.get("gold")   or 0) if before else 0
            elixir = (before.get("elixir") or 0) if before else 0
            print(f"\n*** Storage full: gold={gold:,}  elixir={elixir:,} ***")
            if not wall_upgrade:
                print("*** Stopping. ***\n")
                return False
            ok = upgrade_walls(win, cfg, resource_regions, before)
            before = read_resources(resource_regions, verbose=True)
            print_resources("AFTER WALLS", before)
            if not ok or is_storage_full(before):
                print("*** Wall upgrade could not free storage. Stopping. ***\n")
                return False
            print("*** Storage freed, continuing attacks. ***\n")

        if before and before.get("gold") is not None and before.get("elixir") is not None:
            print(f"[loot] remaining to cap: "
                  f"gold={MAX_GOLD - before['gold']:,}  "
                  f"elixir={MAX_ELIXIR - before['elixir']:,}")

    if not wait_and_click(win, cfg, "attack_menu", "Home Attack! icon"):
        return True
    hsleep(POST_CLICK_PAUSE)

    if not wait_and_click(win, cfg, "find_match", "Find a Match"):
        return True

    wait_until_gone(win, cfg, "find_match", "matchmaking", timeout=MATCHMAKING_TIMEOUT)
    print("Base found, confirming attack...")
    hsleep(POST_CLICK_PAUSE)

    if not wait_and_click(win, cfg, "attack_go", "Attack! (base preview)"):
        return True

    if auto_deploy:
        full_deploy(win, cfg)
        wait_until_surrender_time()
    else:
        # Manual deploy window.
        print(f"\n>>> YOU HAVE THE CONTROLS. Deploy manually now.")
        print(f">>> Bot resumes in {MANUAL_DEPLOY_WAIT:.0f}s "
              f"(Ctrl+C skips the rest of the wait and proceeds).\n")
        try:
            time.sleep(MANUAL_DEPLOY_WAIT)
        except KeyboardInterrupt:
            print("  [Ctrl+C] skipping manual-deploy wait...")

    if not wait_and_click(win, cfg, "end_battle", "End Battle"):
        return True
    hsleep(POST_CLICK_PAUSE)

    if not click_surrender_okay(win, cfg):
        return True
    hsleep(POST_CLICK_PAUSE)

    if not wait_and_click(win, cfg, "return_home", "Return Home", timeout=15):
        return True

    if storage_check:
        time.sleep(1.2)
        after = read_resources(resource_regions, verbose=True)
        print_resources("AFTER", after)

        if before and after:
            def delta(k):
                a = after.get(k)
                b = before.get(k)
                if a is None or b is None:
                    return None
                return a - b

            gained = {k: delta(k) for k in
                      ("gold", "elixir", "dark_elixir", "gems")}

            def fmt_g(v):
                return f"{v:+,}" if v is not None else "n/a"

            print(f"[loot] GAINED  gold={fmt_g(gained['gold'])}  "
                  f"elixir={fmt_g(gained['elixir'])}  "
                  f"de={fmt_g(gained['dark_elixir'])}  "
                  f"gems={fmt_g(gained['gems'])}")

            if gained["gold"] is not None or gained["elixir"] is not None:
                log_loot_row(before, after, gained)

    print("=== Cycle complete, back home ===\n")
    return True


def mode_full_auto(cfg, storage_check, auto_deploy=True, wall_upgrade=False):
    if storage_check:
        print("\nFULL AUTO + STORAGE CHECK")
        print(f"Stop condition: gold >= {MAX_GOLD:,} or elixir >= {MAX_ELIXIR:,}\n")
    elif auto_deploy:
        print("\nFULL AUTO (no storage check)")
        print("Running forever. Ctrl+C to return to menu.\n")
    else:
        print("\nAUTO ATTACK ONLY (no deploy)")
        print(f"Waits {MANUAL_DEPLOY_WAIT:.0f}s each cycle for your manual deploy.")
        print("Ctrl+C to return to menu.\n")

    if wall_upgrade:
        missing = wall_assets_missing()
        if missing:
            print("\n[!] Missing wall templates (put them next to main.py):")
            for m in missing:
                print(f"      - {m}")
            print("    Crop them like the other assets, see README. Aborting.\n")
            return
        print("Auto wall upgrade: ON\n")

    resource_regions = None
    if storage_check:
        resource_regions = load_regions()
        print(f"Resource regions: {resource_regions}\n")

    win = ensure_game_running()

    try:
        while True:
            cfg = load_config()
            keep_going = run_one_attack_cycle(win, cfg, resource_regions,
                                              storage_check, auto_deploy,
                                              wall_upgrade)
            if not keep_going:
                print("Storage full - back to menu.")
                break
            human_cycle_pause()
    except KeyboardInterrupt:
        print("\nCtrl+C - back to menu.")


# =========================================================================
# TEST / CALIBRATION MODE
# =========================================================================
def mode_test(cfg, rounds=15):
    """Scans for every asset repeatedly WITHOUT clicking anything, printing
    score + how long each match_template() call took. Use this to:
      - tune cfg['confidence'] thresholds (raise if you get false positives,
        lower if a real icon never crosses the threshold)
      - see the real effect of --gpu / --fast on speed
      - sanity-check that assets still match after a UI update
    """
    print("\nTEST / CALIBRATE (no clicks, Ctrl+C to stop early)")
    print(f"gpu={'ON' if (USE_GPU and GPU_READY) else 'off'}  "
          f"fast={'ON' if FAST_MODE else 'off'}  "
          f"capture={'mss' if mss_mod else 'pyautogui'}\n")

    win = ensure_game_running()
    totals = {k: {"scores": [], "times_ms": [], "hits": 0} for k in ASSETS}

    try:
        for r in range(1, rounds + 1):
            region = get_region(win)
            screen = capture(region)
            print(f"--- round {r}/{rounds} ---")
            for asset_key in [k for k in ASSETS if os.path.exists(ASSET_PATHS[k])]:
                t0 = time.perf_counter()
                match = match_template(screen, asset_key, cfg, thorough=True)
                dt_ms = (time.perf_counter() - t0) * 1000

                if "error" in match:
                    print(f"  {asset_key:<16} ERROR: {match['error']}")
                    continue

                totals[asset_key]["scores"].append(match["score"])
                totals[asset_key]["times_ms"].append(dt_ms)
                if match["found"]:
                    totals[asset_key]["hits"] += 1

                status = "FOUND" if match["found"] else "miss "
                print(f"  {asset_key:<16} {status} score={match['score']:.3f} "
                      f"(need {match['threshold']:.2f}) scale={match['scale']:.2f} "
                      f"{dt_ms:6.1f} ms")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n  [Ctrl+C] stopping test early...")

    print("\n=== Summary (avg score / avg time / hit rate) ===")
    for asset_key, d in totals.items():
        if not d["scores"]:
            print(f"  {asset_key:<16} no successful reads")
            continue
        n = len(d["scores"])
        avg_score = sum(d["scores"]) / n
        avg_ms = sum(d["times_ms"]) / n
        hit_rate = 100.0 * d["hits"] / n
        print(f"  {asset_key:<16} avg_score={avg_score:.3f}  "
              f"avg_time={avg_ms:6.1f} ms  hit_rate={hit_rate:5.1f}%  (n={n})")
    print()


# =========================================================================
# MAIN
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Clash of Clans Auto-Attack Bot")
    p.add_argument("--gpu", action="store_true",
                    help="use cv2.cuda template matching if a CUDA device + "
                         "opencv-contrib-python (CUDA build) are available; "
                         "silently falls back to CPU otherwise")
    p.add_argument("--fast", action="store_true",
                    help="stop the multi-scale search at the first scale "
                         "that clears the confidence threshold instead of "
                         "always checking all 16 scales (faster, slightly "
                         "less accurate reported score - use mode 4 first "
                         "to make sure your thresholds are solid)")
    p.add_argument("--no-human", action="store_true",
                    help="disable humanizer (fixed delays, exact-center clicks, no breaks)")
    p.add_argument("--rounds", type=int, default=15,
                    help="number of scan rounds for mode 4 (test/calibrate)")
    return p.parse_args()


def main():
    global USE_GPU, FAST_MODE, HUMANIZE

    args = parse_args()
    USE_GPU = args.gpu
    FAST_MODE = args.fast
    HUMANIZE = not args.no_human

    print("Loading config...")
    cfg = load_config()
    ensure_log_header()
    if USE_GPU or FAST_MODE:
        print(f"Flags: gpu={USE_GPU}  fast={FAST_MODE}  "
              f"(GPU is only confirmed once a game mode loads OpenCV)")

    while True:
        print_menu()
        try:
            choice = input("Choose [0-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice == "0":
            print("Exiting.")
            break

        elif choice == "1":
            if not _ensure_full_deps(need_ocr=False):
                continue
            mode_full_auto(cfg, storage_check=False, auto_deploy=True)

        elif choice == "2":
            if not _ensure_full_deps(need_ocr=True):
                continue
            mode_full_auto(cfg, storage_check=True, auto_deploy=True)

        elif choice == "3":
            if not _ensure_full_deps(need_ocr=False):
                continue
            mode_full_auto(cfg, storage_check=False, auto_deploy=False)

        elif choice == "4":
            if not _ensure_full_deps(need_ocr=False):
                continue
            mode_test(cfg, rounds=args.rounds)

        elif choice == "5":
            if not _ensure_full_deps(need_ocr=True):
                continue
            mode_full_auto(cfg, storage_check=True, auto_deploy=True,
                           wall_upgrade=True)

        elif choice == "6":
            if not _ensure_full_deps(need_ocr=False):
                continue
            mode_wall_test(cfg)

        else:
            print(f"Unknown choice: {choice!r}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        # Silences the "Exception ignored on threading shutdown" warning
        # that pygetwindow / pyautogui trigger on Ctrl+C during exit.
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass