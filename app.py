# -*- coding: utf-8 -*-
"""
羽毛球高远球动作智能化教学辅助系统 v3.0
=========================================
功能：
  1. 单人动作分析：多拍切分 + 逐拍动作识别 + 骨骼标注 + 角度曲线 + 评分 + 文字诊断建议
  2. 标准动作对比：上传标准视频与学员视频，击球时刻自动对齐，曲线叠加对比
  3. 动作标准库（v3 新增）：分析标准视频一键保存为"动作模板"；
     学员视频与模板做 DTW（动态时间规整）匹配，给出相似度分数与逐阶段偏差诊断
依赖：streamlit==1.31.0 / mediapipe==0.10.9 / opencv-python==4.9.0.80 / numpy==1.26.4 / pandas==2.2.0
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

# ============ 页面配置 ============
st.set_page_config(page_title="羽毛球高远球智能教学系统", page_icon="🏸", layout="wide")

# ============ 界面美化 CSS ============
st.markdown(
    """
    <style>
    .big-title{background:linear-gradient(90deg,#1f6f54,#2eaa7a);color:#fff;
        padding:22px 28px;border-radius:14px;margin-bottom:6px;
        box-shadow:0 4px 14px rgba(46,170,122,.35);}
    .big-title h1{margin:0;font-size:30px;}
    .big-title p{margin:6px 0 0;font-size:14px;opacity:.9;}
    .metric-card{background:#fff;border:1px solid #e6e9ef;border-radius:12px;
        padding:14px 18px;box-shadow:0 2px 8px rgba(0,0,0,.06);text-align:center;}
    .metric-card .val{font-size:30px;font-weight:800;color:#1f6f54;}
    .metric-card .lab{font-size:13px;color:#6b7280;margin-top:2px;}
    div[data-testid="stMetric"]{background:#fff;border-radius:12px;
        border:1px solid #e6e9ef;box-shadow:0 2px 8px rgba(0,0,0,.05);padding:12px 8px;}
    .hint{color:#6b7280;font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ 动作专项规则库 ============
# 每种动作：名称、识别阈值区间、诊断理想区间
ACTION_PROFILES = {
    "clear": {   # 高远球
        "name": "高远球", "emoji": "🛫",
        "ideal": {
            "elbow_peak": (150, 175), "shoulder_peak": (135, 170),
            "swing_speed": 300, "impact_lift": 0.0,
        },
    },
    "smash": {   # 杀球
        "name": "杀球", "emoji": "💥",
        "ideal": {
            "elbow_peak": (155, 178), "shoulder_peak": (140, 175),
            "swing_speed": 500, "impact_lift": 0.10, "post_drop": 0.02,
        },
    },
    "drop": {    # 吊球
        "name": "吊球", "emoji": "🎯",
        "ideal": {
            "elbow_peak": (140, 170), "shoulder_peak": (130, 165),
            "swing_speed": 200, "swing_speed_max": 330, "impact_lift": 0.02,
            "elbow_ready": (75, 110),
        },
    },
    "lift": {    # 挑球
        "name": "挑球", "emoji": "⤴️",
        "ideal": {
            "elbow_peak": (120, 160), "swing_speed": 200,
            "post_rise": 0.04,
        },
    },
}
MAX_FRAMES = 240   # 最多分析的采样帧数（约 16 秒 @30fps 隔帧采样）
FRAME_STEP = 2     # 隔帧采样
ALIGN_PRE, ALIGN_POST = 15, 15  # 对比时击球帧前后窗口
# —— 多动作分段参数 ——
IMPACT_THR = 0.02       # 击球检测：手腕向上速度阈值（归一化/采样帧）
IMPACT_MIN_GAP = 10     # 两次击球最小间隔（采样帧）
SEG_PRE, SEG_POST = 12, 10  # 每个动作片段的击球前后窗口
MAX_ACTIONS = 6         # 最多识别的连续动作数
# —— 模板匹配参数 ——
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
TEMPLATE_SCALE = 0.15   # DTW 距离→相似度的映射系数（距离 15 ≈ 85 分）


# ============ DTW 模板匹配 ============
def dtw_distance(s1, s2):
    """DTW 形状距离：z-score 标准化后比对，只反映曲线"形状"（与整体幅度无关）。
    注意：形状距离无法区分"整体平移"，需配合 magnitude_similarity 使用。"""
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0

    def znorm(x):
        sd = x.std()
        return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)

    a, b = znorm(s1), znorm(s2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    # 剪枝带宽：至少覆盖两条序列的长度差，否则快慢差异大时无法正确对齐
    band = max(8, abs(n - m) + 4)
    for i in range(1, n + 1):
        j_lo, j_hi = max(1, i - band), min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def magnitude_similarity(s1, s2):
    """幅度相似度：比较两条曲线的均值与动态范围（排除"形状一样但整体差几十度"的情况）。
    返回 (0~100, 均值差, 幅度差)。"""
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    mean_diff = abs(s1.mean() - s2.mean())
    rng_diff = abs((s1.max() - s1.min()) - (s2.max() - s2.min()))
    # 关节角均值差 12° / 动态范围差 15° 即记为 0 分
    # （生物力学上，关节角差 12° 已属明显不同的动作模式）
    m_score = max(0.0, 1.0 - mean_diff / 12.0)
    r_score = max(0.0, 1.0 - rng_diff / 15.0)
    # 取两者最小值为主：整体平移(范围差=0)或幅度缩放(均值差=0)任一项严重偏离都应拉低总分
    return 100.0 * min(m_score, r_score), mean_diff, rng_diff


def joint_similarity(s1, s2):
    """单关节综合相似度 = 形状(35%) + 幅度(65%)。
    形状用 DTW 弹性对齐（抗快慢差异），幅度用均值/动态范围，取较严的一项（识别整体角度偏差）。
    幅度权重更高：动作节奏因人而异可以宽容，但关节角度脱离标准就是实打实的技术问题。"""
    shape = dtw_similarity(dtw_distance(s1, s2))
    mag, d_mean, d_rng = magnitude_similarity(s1, s2)
    return 0.35 * shape + 0.65 * mag, shape, mag, d_mean, d_rng


def dtw_similarity(dist):
    """形状距离 → 0~100 相似度（指数映射，距离越大衰减越快）"""
    return max(0.0, min(100.0, 100.0 * np.exp(-TEMPLATE_SCALE * dist)))


def load_templates():
    """读取本地标准库：{动作key: {"name":…, "elbow":[…], "shoulder":[…], …}}"""
    if not os.path.isdir(TEMPLATE_DIR):
        return {}
    out = {}
    for fn in sorted(os.listdir(TEMPLATE_DIR)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(TEMPLATE_DIR, fn), "r", encoding="utf-8") as f:
                    out[fn[:-5]] = json.load(f)
            except Exception:
                pass
    return out


def save_template(action_key, data):
    """把一个动作模板写入标准库（同名覆盖）"""
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    path = os.path.join(TEMPLATE_DIR, action_key + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def delete_template(action_key):
    path = os.path.join(TEMPLATE_DIR, action_key + ".json")
    if os.path.exists(path):
        os.remove(path)


def parse_angle_csv(df_csv):
    """识别 CSV 中的关节角度列（支持中英文列名，六关节任意组合，至少需肘部）。
    返回 (({关节key: 序列}), None) 或 (None, 错误信息)。"""
    cols = [str(c) for c in df_csv.columns]
    alias = {  # 关节key: 列名匹配关键词
        "elbow": ["elbow", "肘"], "shoulder": ["shoulder", "肩"], "wrist": ["wrist", "腕"],
        "hip": ["hip", "髋"], "knee": ["knee", "膝"], "ankle": ["ankle", "踝"],
    }
    found = {}
    for key, kws in alias.items():
        col = next((c for c in cols if any(kw.lower() in c.lower() for kw in kws)), None)
        if col is not None:
            vals = pd.to_numeric(df_csv[col], errors="coerce").dropna().tolist()
            if len(vals) >= 6:
                found[key] = vals
    if "elbow" not in found:
        return None, "未找到肘部角度列（列名需含 elbow 或「肘」，如 elbow / 肘部角度）"
    n = min(len(v) for v in found.values())
    if n < 6:
        return None, "有效数据点不足（至少 6 个采样点才能构成动作曲线）"
    return {k: v[:n] for k, v in found.items()}, None


def template_evaluate(seg, tpl):
    """学员片段 vs 标准模板 → 六关节加权相似度（形状+幅度）+ 动力链分环节诊断。
    返回 (总分, 各关节相似度dict, 反馈列表)"""
    fb = []
    sims = {}
    details = {}
    total_w = 0.0
    weighted = 0.0
    for k in JOINT_KEYS:
        if k in tpl and k in seg:
            sim, shape, mag, d_mean, d_rng = joint_similarity(seg[k], tpl[k])
            sims[k] = sim
            details[k] = (shape, mag, d_mean, d_rng)
            weighted += JOINT_W[k] * sim
            total_w += JOINT_W[k]
    total = round(weighted / total_w) if total_w > 0 else 0

    # —— 按动力链环节诊断（下肢 → 转体 → 挥臂 → 手腕）——
    chain = [
        ("ankle", "下肢蹬转（踝）", "蹬地发力是动力链起点，角度偏差会影响整条链的力量传导。"),
        ("knee", "下肢蹬伸（膝）", "膝角偏差说明蹬地蓄力不够，击球高度和力量都会受限。"),
        ("hip", "转髋（髋）", "转髋不足意味着只用手臂发力，长期容易肩肘劳损。"),
        ("shoulder", "转体挥臂（肩）", "肩臂幅度不足则借不上躯干力量。"),
        ("elbow", "鞭打伸展（肘）", "肘部是鞭打链条末端，直接影响出球速度。"),
        ("wrist", "手腕控制（腕）", "手腕偏差会影响拍面角度与出球方向。"),
    ]
    for k, label, why in chain:
        if k not in sims:
            continue
        v = sims[k]
        shape, mag, d_mean, d_rng = details[k]
        # 区分"形状不像"和"幅度不像"——两者训练重点完全不同
        if v < 60:
            if shape < 55 and mag >= 60:
                fb.append(("err", "%s 动作时序/形状与标准差异明显（相似度 %.0f）：%s" % (label, v, why)))
            elif mag < 55 and shape >= 60:
                fb.append(("err", "%s 角度幅度偏离标准（均值差 %.0f°，相似度 %.0f）：动作时序接近但活动范围不对，注意该关节的幅度。" % (label, d_mean, v)))
            else:
                fb.append(("err", "%s 与标准差异明显（形状 %.0f / 幅度 %.0f）：%s" % (label, shape, mag, why)))
        elif v < 78:
            fb.append(("warn", "%s 略有偏差（形状 %.0f / 幅度 %.0f），可对照曲线微调。" % (label, shape, mag)))
        else:
            fb.append(("ok", "%s 与标准吻合（形状 %.0f / 幅度 %.0f）。" % (label, shape, mag)))

    # —— 显著性提示：找出偏差最大的环节 ——
    if sims:
        worst_k = min(sims, key=lambda x: sims[x])
        if sims[worst_k] < 70:
            fb.insert(0, ("tip", "本拍最需改进的是【%s】（相似度 %.0f），建议优先针对该环节练习。"
                          % (JOINT_CN[worst_k], sims[worst_k])))

    # —— 峰值角度绝对值对比（肘部为鞭打核心）——
    if "elbow" in tpl:
        tpl_peak = float(np.max(tpl["elbow"]))
        if seg["peak_elbow"] < tpl_peak - 12:
            fb.append(("err", "肘部峰值 %.0f° 低于标准 %.0f°：鞭打末端未充分展开。" % (seg["peak_elbow"], tpl_peak)))
        elif seg["peak_elbow"] > tpl_peak + 12:
            fb.append(("warn", "肘部峰值 %.0f° 高于标准 %.0f°，注意避免过度伸直。" % (seg["peak_elbow"], tpl_peak)))
        else:
            fb.append(("ok", "肘部峰值 %.0f° 与标准 %.0f° 基本一致。" % (seg["peak_elbow"], tpl_peak)))
    return total, sims, fb

# ============ 六关节动力链配置 ============
# 每项：(显示名, MediaPipe 关键点三元组(定义点A,顶点B,定义点C), DTW权重, 全部是右侧点的索引)
# MediaPipe Pose 索引：肩11/12 肘13/14 腕15/16 髋23/24 膝25/26 踝27/28
JOINTS = [
    ("肘部", "elbow",    0.24, "鞭打链条末端，决定力量传导与出球速度"),
    ("肩部", "shoulder", 0.18, "躯干转体与肩臂挥动幅度"),
    ("手腕", "wrist",    0.16, "拍面控制与最后鞭打发力"),
    ("髋部", "hip",      0.16, "下肢到上肢的转体中转站"),
    ("膝部", "knee",     0.16, "蹬地蓄力与重心控制"),
    ("踝部", "ankle",    0.10, "蹬转起点与步伐稳定性"),
]
JOINT_KEYS = [j[1] for j in JOINTS]
JOINT_W = {j[1]: j[2] for j in JOINTS}
JOINT_CN = {j[1]: j[0] for j in JOINTS}
# 右侧关键点索引（MediaPipe 左右成对：右侧为奇数，左侧=右侧+1）
KP = {
    "shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27,
    "index_finger": 19, "foot_index": 31,
}
# 每个角度的三点定义（右侧）：(A, B顶点, C)
ANGLE_DEF = {
    "elbow":    ("shoulder", "elbow", "wrist"),          # 肩-肘-腕
    "shoulder": ("hip", "shoulder", "elbow"),            # 髋-肩-肘
    "wrist":    ("elbow", "wrist", "index_finger"),      # 肘-腕-食指
    "hip":      ("shoulder", "hip", "knee"),             # 肩-髋-膝
    "knee":     ("hip", "knee", "ankle"),                # 髋-膝-踝
    "ankle":    ("knee", "ankle", "foot_index"),         # 膝-踝-脚尖
}


def angle_triplet(key, side):
    """返回某关节在指定持拍侧的关键点索引三元组 (A, B顶点, C)"""
    off = 0 if side == "right" else 1
    a, b, c = ANGLE_DEF[key]
    return (KP[a] + off, KP[b] + off, KP[c] + off)


def calc_joint_angle(lm, key, side):
    """计算某关节角度；关键点可见度不足时返回 None"""
    ia, ib, ic = angle_triplet(key, side)
    if lm[ia].visibility < 0.3 or lm[ib].visibility < 0.3 or lm[ic].visibility < 0.3:
        return None
    return calc_angle(lm2pt(lm[ia]), lm2pt(lm[ib]), lm2pt(lm[ic]))


# ============ 工具函数 ============
def calc_angle(a, b, c):
    """三点夹角（度）：a-b-c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.clip(np.arccos(np.clip(cosv, -1, 1)), 0, 180)))


def lm2pt(lm):
    return [lm.x, lm.y]


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


@st.cache_resource
def get_pose():
    return mp_pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True
    )


def pick_side(lm):
    """按关键点可见度判定持拍侧：综合上肢（肩肘腕）+下肢（髋膝踝）的置信度"""
    def vis(off):
        return (lm[11 + off].visibility + lm[13 + off].visibility + lm[15 + off].visibility
                + lm[23 + off].visibility + lm[25 + off].visibility + lm[27 + off].visibility)
    return "right" if vis(0) >= vis(1) else "left"


# 屏幕上标注的角度 → 挂在哪个关键点上（右侧基准）
LABEL_AT = {
    "elbow": ("elbow", -10), "shoulder": ("shoulder", 22), "wrist": ("wrist", 30),
    "hip": ("hip", -10), "knee": ("knee", -10), "ankle": ("ankle", 26),
}
LBL_COLOR = {"elbow": (0, 0, 255), "shoulder": (255, 0, 0), "wrist": (0, 160, 255),
             "hip": (200, 0, 200), "knee": (0, 200, 0), "ankle": (150, 100, 0)}


def draw_annotated(frame_bgr, landmarks, angles, side="right"):
    """画骨骼 + 多关节角度数字（角度：{关节key: 度数}）"""
    mp_drawing.draw_landmarks(
        frame_bgr, landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2),
    )
    h, w = frame_bgr.shape[:2]
    off = 0 if side == "right" else 1
    for key, deg in angles.items():
        if deg is None or key not in LABEL_AT:
            continue
        kp_name, dy = LABEL_AT[key]
        kp = landmarks.landmark[KP[kp_name] + off]
        if kp.visibility < 0.3:
            continue
        x, y = int(kp.x * w), int(kp.y * h)
        text = "%s%d" % (JOINT_CN[key][0], int(deg))  # 取关节名首字，如"肘168"
        cv2.putText(frame_bgr, text, (x + 8, y + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, LBL_COLOR.get(key, (0, 0, 0)), 2)
    return frame_bgr


# ============ 核心分析 ============
def analyze_video(video_bytes, label=""):
    """逐帧姿态估计。返回 dict：
    jpgs(标注帧) elbow/shoulder(角度序列) impact(击球帧索引) fps 及派生指标"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp = f.name
    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened():
        os.unlink(tmp)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = get_pose()

    jpgs = []
    # 六关节角度序列 + 手腕/肩 y 轨迹
    series = {k: [] for k in JOINT_KEYS}
    wrist_y, sh_y = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_STEP == 0 and len(jpgs) < MAX_FRAMES:
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            lm = res.pose_landmarks
            if lm:
                side = pick_side(lm.landmark)
                ang = {k: calc_joint_angle(lm.landmark, k, side) for k in JOINT_KEYS}
                for k in JOINT_KEYS:
                    series[k].append(ang[k] if ang[k] is not None else np.nan)
                off = 0 if side == "right" else 1
                wrist_y.append(lm.landmark[KP["wrist"] + off].y)
                sh_y.append(lm.landmark[KP["shoulder"] + off].y)
                anno = draw_annotated(frame.copy(), lm, ang, side)
                ok2, buf = cv2.imencode(".jpg", anno, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok2:
                    jpgs.append(buf.tobytes())
            else:
                for k in JOINT_KEYS:
                    series[k].append(np.nan)
                wrist_y.append(np.nan)
                sh_y.append(np.nan)
                jpgs.append(None)
        idx += 1
    cap.release()
    os.unlink(tmp)

    if len([v for v in series["elbow"] if not np.isnan(v)]) < 8:
        return None

    for k in JOINT_KEYS:
        series[k] = pd.Series(series[k]).interpolate(limit=4).ffill().bfill().tolist()

    # 手腕 y 序列（用于多击球检测）
    wy = pd.Series(wrist_y).interpolate(limit=4).ffill().bfill()
    sh = pd.Series(sh_y).interpolate(limit=4).ffill().bfill()

    dt = FRAME_STEP / fps  # 每采样帧的秒数

    # —— 检测全部击球时刻，逐段计算指标 ——
    impacts = detect_impacts(wy)
    segments = []
    for imp in impacts:
        m = segment_metrics(series, wy, sh, imp, dt)
        if m is None:
            continue
        m["action"] = classify_action(m)
        segments.append(m)

    if not segments:
        return None

    # 整段视频的汇总指标（取最强一次击球，兼容对比模式）
    best = max(segments, key=lambda s: s["swing_speed"])

    out = {
        "label": label, "jpgs": jpgs, "fps": fps, "dt": dt,
        "wrist_y": wy.tolist(), "sh_y": sh.tolist(),
        "impact": segments[0]["impact"], "segments": segments,
        "peak_elbow": best["peak_elbow"], "min_elbow": best["min_elbow"],
        "peak_shoulder": best["peak_shoulder"], "swing_speed": best["swing_speed"],
        "impact_lift": best["impact_lift"],
        "post_move": best["post_move"], "pre_move": best["pre_move"],
    }
    for k in JOINT_KEYS:  # 全序列（整段曲线图用）
        out[k] = series[k]
    return out


def detect_impacts(wy):
    """检测视频中每一次击球：手腕向上速度的局部极大值。
    返回按时间排序的击球帧索引列表（最多 MAX_ACTIONS 个）。"""
    vy = wy.diff()
    up = (-vy).fillna(0.0).values  # 正值 = 手腕向上运动的速度
    n = len(up)
    if n < 4:
        return [n // 2] if n else []
    cands = []
    for i in range(1, n - 1):
        if up[i] > up[i - 1] and up[i] >= up[i + 1] and up[i] > IMPACT_THR:
            cands.append((i, up[i]))
    if not cands:  # 兜底：至少取全局最大向上速度处
        return [int(np.argmax(up))]
    # 按强度降序贪心选择，保证两次击球间隔 ≥ IMPACT_MIN_GAP
    cands.sort(key=lambda x: -x[1])
    chosen = []
    for i, v in cands:
        if all(abs(i - j) >= IMPACT_MIN_GAP for j, _ in chosen):
            chosen.append((i, v))
        if len(chosen) >= MAX_ACTIONS:
            break
    chosen.sort()  # 按时间排序
    return [i for i, _ in chosen]


def segment_metrics(series, wy, sh, impact, dt):
    """计算单个动作片段（击球前后窗口）的六关节运动学指标"""
    elbow_s = series["elbow"]
    n = len(elbow_s)
    if impact is None or impact < 0 or impact >= n:
        return None
    lo = max(0, impact - SEG_PRE)
    hi = min(n - 1, impact + SEG_POST)
    elbow_win = elbow_s[lo:hi + 1]
    if len(elbow_win) < 4:
        return None
    # 挥速只统计击球爆发窗口（前4帧~后3帧），排除远端跟拍抖动干扰
    b_lo = max(0, impact - lo - 4)
    b_hi = min(len(elbow_win) - 1, impact - lo + 3)
    diffs = [abs(elbow_win[k + 1] - elbow_win[k]) for k in range(b_lo, b_hi)]
    p_lo, p_hi = impact + 1, min(n - 1, impact + 7)
    r_lo, r_hi = max(0, impact - 7), impact - 1

    m = {
        "impact": impact, "win_lo": lo, "win_hi": hi,
        "swing_speed": (max(diffs) / dt) if diffs else 0.0,
        "impact_lift": float(sh.iloc[impact] - wy.iloc[impact]),
        "post_move": float(wy.iloc[p_hi] - wy.iloc[p_lo]) if p_hi > p_lo else 0.0,
        "pre_move": float(wy.iloc[r_hi] - wy.iloc[r_lo]) if r_hi > r_lo else 0.0,
    }
    # 六关节：片段序列 + 峰值/最低值
    for k in JOINT_KEYS:
        win = series[k][lo:hi + 1]
        m[k] = win
        m["peak_" + k] = float(max(win))
        m["min_" + k] = float(min(win))
    return m


def classify_action(a):
    """动作分类决策树（基于运动学特征，MediaPipe y 轴向下为正）：
    1) 击球点远高于肩 + 挥速极快 → 杀球
    2) 击球后手腕仍持续上升（y 减小，post_move<0）→ 挑球（下手位向上发力）
    3) 挥速慢 → 吊球（减力切削）
    4) 其余 → 高远球兜底（本项目主动作）"""
    lift, spd = a["impact_lift"], a["swing_speed"]
    post = a["post_move"]

    if lift > 0.12 and spd >= 420:
        return "smash"
    if post < -0.03:  # 击球后手腕继续上升 → 下手位向上发力
        return "lift"
    if spd < 260:
        return "drop"
    return "clear"


def score_and_feedback(a, action="clear"):
    """按动作专项规则的评分引擎 + 诊断文字建议"""
    P = ACTION_PROFILES[action]
    ideal = P["ideal"]
    score = 100
    fb = []

    lo, hi = ideal["elbow_peak"]
    if a["peak_elbow"] < lo:
        d = lo - a["peak_elbow"]
        score -= min(28, d * 0.8)
        fb.append(("err", "肘部最大伸展角 %.0f°（理想 %d–%d°）：鞭打末端未完全展开，力量传导受阻。" % (a["peak_elbow"], lo, hi)))
        fb.append(("tip", "练习：持拍做慢速「引拍—顶肘—挥臂」分解动作，每次在镜子前确认手臂几乎伸直。"))
    elif a["peak_elbow"] > hi:
        score -= 8
        fb.append(("warn", "肘部几乎过度伸直（%.0f°），注意保持微屈，避免肘部劳损。" % a["peak_elbow"]))
    else:
        fb.append(("ok", "肘部鞭打伸展充分（峰值 %.0f°），力量传导良好。" % a["peak_elbow"]))

    if "shoulder_peak" in ideal:
        lo, hi = ideal["shoulder_peak"]
        if a["peak_shoulder"] < lo:
            score -= min(20, (lo - a["peak_shoulder"]) * 0.5)
            fb.append(("warn", "肩部挥动幅度偏小（峰值 %.0f°，理想 %d°+）：躯干转体不足，借不上身体力量。" % (a["peak_shoulder"], lo)))
            fb.append(("tip", "练习：侧身架拍时让左肩对网（右手持拍者），击球时重心从右脚转到左脚。"))
        else:
            fb.append(("ok", "转体充分，肩臂挥动幅度达标（峰值 %.0f°）。" % a["peak_shoulder"]))

    spd_min = ideal.get("swing_speed", 300)
    spd_max = ideal.get("swing_speed_max", 100000)
    if a["swing_speed"] < spd_min:
        penalty = 12 if action == "drop" else 18
        score -= penalty
        fb.append(("warn", "挥拍角速度 %.0f°/s 偏慢（该动作理想 ≥%d°/s）：击球爆发力不足。" % (a["swing_speed"], spd_min)))
        fb.append(("tip", "练习：对墙做 20 次/组的快速鞭打挥拍，体会「放松引拍—瞬间发力」。"))
    elif a["swing_speed"] > spd_max:
        score -= 10
        fb.append(("warn", "挥拍角速度 %.0f°/s 偏快（该动作宜 ≤%d°/s）：吊球讲究减力控制，过快难切削。" % (a["swing_speed"], spd_max)))
    else:
        fb.append(("ok", "挥拍角速度 %.0f°/s，与该动作发力特点匹配。" % a["swing_speed"]))

    lift_min = ideal.get("impact_lift", 0.0)
    if action == "lift":
        if a["post_move"] > -ideal.get("post_rise", 0.04):
            score -= 18
            fb.append(("err", "击球后手腕未持续上送（净变化 %.0f%%）：挑球应自下向上送拍，弧线拉不起来。" % (a["post_move"] * 100)))
            fb.append(("tip", "练习：多球训练低手位挑球，强调「拍面朝上、小臂带腕上送」。"))
        else:
            fb.append(("ok", "击球后球拍随挥上送充分（%.0f%%），挑球弧线饱满。" % (-a["post_move"] * 100)))
    else:
        if a["impact_lift"] <= lift_min + (0.02 if action == "drop" else 0.0):
            score -= 18
            fb.append(("err", "击球点偏低：击球瞬间手腕未高于肩部，回球弧线受压制。"))
            fb.append(("tip", "练习：把羽毛球用线吊在高处，练习在最高点迎球，养成「早引拍、高点击球」习惯。"))
        elif action == "smash" and a["impact_lift"] < ideal.get("impact_lift", 0.0):
            score -= 8
            fb.append(("warn", "杀球击球点不够高（提升量 %.0f%%）：下压角度变缓，杀球威胁下降。" % (a["impact_lift"] * 100)))
        else:
            fb.append(("ok", "击球点高于肩部（提升量 %.0f%%），出球角度理想。" % (a["impact_lift"] * 100)))

    if "elbow_ready" in ideal:
        lo, hi = ideal["elbow_ready"]
        if a["min_elbow"] < lo or a["min_elbow"] > hi:
            score -= 6
            fb.append(("warn", "引拍肘角 %.0f°（理想 %d–%d°）：引拍过直或过屈都会削弱动作隐蔽性。" % (a["min_elbow"], lo, hi)))

    return max(0, round(score)), fb


def angle_chart_df(a, prefix="", keys=None):
    """六关节角度曲线 DataFrame；keys 指定要画的关节（默认全部）"""
    keys = keys or JOINT_KEYS
    data = {}
    for k in keys:
        if k in a:
            data[prefix + JOINT_CN[k] + "角度"] = a[k]
    n = len(next(iter(data.values()))) if data else 0
    return pd.DataFrame(data, index=np.arange(n))


def aligned_series(a, key):
    """以击球帧为 0 点对齐，取前后窗口"""
    s = pd.Series(a[key])
    center = a["impact"]
    lo = max(0, center - ALIGN_PRE)
    hi = min(len(s), center + ALIGN_POST + 1)
    seg = s.iloc[lo:hi]
    seg.index = seg.index - center
    return seg


def render_fb(fb):
    for level, text in fb:
        if level == "ok":
            st.markdown("✅ " + text)
        elif level == "tip":
            st.markdown("💡 " + text)
        elif level == "warn":
            st.markdown("⚠️ " + text)
        else:
            st.markdown("❌ " + text)


def load_video_bytes(key, caption):
    """上传 或 选择本地 videos/ 目录下的视频"""
    c1, c2 = st.columns([3, 2])
    with c1:
        f = st.file_uploader(caption, type=["mp4", "mov", "avi", "mkv"], key=key)
        if f is not None:
            return f.getvalue()
    with c2:
        vdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
        vids = []
        if os.path.isdir(vdir):
            vids = [x for x in sorted(os.listdir(vdir))
                    if x.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        if vids:
            sel = st.selectbox("或选择本地视频（videos/ 文件夹）", vids, key=key + "_lb")
            if sel:
                with open(os.path.join(vdir, sel), "rb") as fp:
                    return fp.read()
        else:
            st.markdown('<span class="hint">未发现本地 videos/ 视频库</span>', unsafe_allow_html=True)
    return None


def show_frame_browser(jpgs, impact, key):
    """逐帧浏览（jpgs 可为整段或某动作片段的切片）"""
    valid = [(i, j) for i, j in enumerate(jpgs) if j]
    if not valid:
        st.warning("未能生成标注帧，请换一段人物更清晰的视频。")
        return
    default = min(impact, valid[-1][0]) if impact is not None else valid[0][0]
    valid_indices = [i for i, _ in valid]
    default_idx = valid_indices.index(default) if default in valid_indices else 0
    fi = st.slider("拖动查看动作过程（骨架 + 实时角度）", 0, len(valid) - 1,
                   default_idx, key="browser_" + key)
    st.image(valid[fi][1], use_column_width=True)
    real = valid[fi][0]
    if impact is not None and abs(real - impact) <= 1:
        st.markdown("🎯 **↑ 当前为击球时刻**")


def show_metrics(a, score=None):
    cols = st.columns(5)
    data = [
        ("综合评分", "%.0f 分" % score if score is not None else "-"),
        ("肘部峰值", "%.0f°" % a["peak_elbow"]),
        ("引拍肘角", "%.0f°" % a["min_elbow"]),
        ("挥拍角速度", "%.0f°/s" % a["swing_speed"]),
        ("击球点提升", "%.0f%%" % (a["impact_lift"] * 100)),
    ]
    for c, (lab, val) in zip(cols, data):
        c.metric(lab, val)


def eval_segment(seg, action):
    """统一评分入口：有该动作的标准模板 → DTW 模板匹配；无 → 常模规则。
    返回 (总分, 反馈列表, 评价模式, 模板名, 各关节相似度)"""
    tpl_map = load_templates()
    tpl = tpl_map.get(action)
    if tpl is not None:
        total, sims, fb = template_evaluate(seg, tpl)
        return total, fb, "模板匹配", tpl.get("name", action), sims
    s, fb = score_and_feedback(seg, action)
    return s, fb, "常模规则", None, None


def render_mode_badge(mode, tpl_name):
    if mode == "模板匹配":
        st.markdown("🎯 **评价基准：标准动作模板「%s」**（六关节加权 DTW 匹配）" % tpl_name)
        st.caption("与「动作标准库」中保存的标准曲线逐关节比对，评价的是动作形状与标准的一致性。")
    else:
        st.markdown("📏 **评价基准：运动生物力学常模规则**（未找到该动作的标准模板）")
        st.caption("如需以自定义标准动作评价，请到「动作标准库」页建立该动作的模板。")


def show_chain_sims(sims):
    """动力链各环节相似度可视化"""
    if not sims:
        return
    st.markdown("##### 🦵 动力链分环节相似度")
    order = ["ankle", "knee", "hip", "shoulder", "elbow", "wrist"]
    cols = st.columns(len(order))
    for c, k in zip(cols, order):
        if k in sims:
            c.metric(JOINT_CN[k], "%.0f" % sims[k])
    st.caption("动力链顺序：踝（蹬转）→ 膝（蹬伸）→ 髋（转体）→ 肩（挥臂）→ 肘（鞭打）→ 腕（控拍）")


# ============ 页头 ============
st.markdown(
    """
    <div class="big-title">
      <h1>🏸 羽毛球动作智能化教学辅助系统</h1>
      <p>基于 MediaPipe 六关节动力链分析与 DTW 模板匹配诊断 · v4.0 · 大学生创新创业训练项目</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["📊 单人动作分析", "⚖️ 标准动作对比", "🎯 动作标准库", "ℹ️ 系统说明"])

# ============ Tab1：单人分析 ============
with tab1:
    st.markdown("上传一段羽毛球击球视频（单一动作或连续多个动作均可），系统自动切分每一次击球、识别动作类型，逐段给出关节角度与专项评分。")
    data = load_video_bytes("solo", "上传学员视频")
    if data and st.button("🚀 开始分析", key="btn_solo"):
        with st.spinner("正在逐帧姿态估计与动作分割……"):
            a = analyze_video(data, label="solo")
        if a is None:
            st.error("未能检测到人体关键点，请确认视频中人物全身清晰、光线充足。")
        else:
            st.session_state["solo_result"] = a

    if "solo_result" in st.session_state:
        a = st.session_state["solo_result"]
        segs = a["segments"]

        # —— 多动作汇总清单 ——
        scored = []
        for k, seg in enumerate(segs):
            act = seg["action"]
            s, fb, mode, tpl_name, sims = eval_segment(seg, act)
            scored.append((k, seg, act, s, fb, mode, tpl_name, sims))

        if len(scored) > 1:
            st.markdown("### 📋 本段视频共检测到 **%d 次击球**" % len(scored))
            cards = st.columns(min(len(scored), 6))
            for c, (k, seg, act, s, fb, mode, tpl_name, sims) in zip(cards, scored):
                P = ACTION_PROFILES[act]
                c.metric("%s 第%d拍·%s" % (P["emoji"], k + 1, P["name"]), "%d 分" % s)
            st.caption("识别依据：击球点高度、挥拍角速度、击球前后手腕轨迹特征。点击下方选择要查看详情的一次击球。")
            labels = ["第%d拍 · %s（%d 分）" % (k + 1, ACTION_PROFILES[act]["name"], s)
                      for k, seg, act, s, fb, mode, tpl_name, sims in scored]
            st.selectbox("选择第几拍查看详情", labels, key="seg_sel")
            k, seg, act, s, fb, mode, tpl_name, sims = scored[labels.index(st.session_state["seg_sel"])]
            st.markdown("#### %s 第%d拍详情：%s" % (ACTION_PROFILES[act]["emoji"], k + 1, ACTION_PROFILES[act]["name"]))
        else:
            k, seg, act, s, fb, mode, tpl_name, sims = scored[0]
            P = ACTION_PROFILES[act]
            st.markdown("### %s 系统识别动作：**%s**" % (P["emoji"], P["name"]))
            st.caption("识别依据：击球点高度、挥拍角速度、击球前后手腕轨迹特征。")

        render_mode_badge(mode, tpl_name)
        show_metrics(seg, s)
        if sims:
            show_chain_sims(sims)
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("动作过程回放")
            st.caption("骨架图上标注六关节实时角度（肘/肩/腕/髋/膝/踝）")
            seg_jpgs = a["jpgs"][seg["win_lo"]: seg["win_hi"] + 1]
            show_frame_browser(seg_jpgs, seg["impact"] - seg["win_lo"], "seg%d" % k)
        with c2:
            st.subheader("本拍六关节角度曲线")
            show_keys = st.multiselect(
                "选择要显示的关节", JOINT_KEYS, default=["elbow", "shoulder", "wrist"],
                format_func=lambda kk: JOINT_CN[kk], key="curve_keys%d" % k)
            show_keys = show_keys or JOINT_KEYS
            if mode == "模板匹配":
                tpl = load_templates()[act]
                tpl_part = angle_chart_df(tpl, keys=show_keys)
                stu_part = angle_chart_df(seg, keys=show_keys, prefix="学员·")
                cmp_df = pd.concat([stu_part, tpl_part.add_prefix("标准·")], axis=1)
                st.line_chart(cmp_df, height=300)
                st.caption("「学员·」为本次动作，「标准·」为模板曲线（已按击球时刻对齐）。")
            else:
                st.line_chart(angle_chart_df(seg, keys=show_keys), height=300)
                st.caption("横轴为片段内采样帧序号，可观察引拍（角度低谷）→ 击球（峰值）的完整时序。")
        if len(scored) > 1:
            with st.expander("📈 查看整段视频的完整角度曲线（含全部击球）"):
                st.line_chart(angle_chart_df(a, keys=["elbow", "shoulder", "wrist"]), height=280)
                st.caption("每个「低谷→高峰」的波动对应一次完整的挥拍周期。")
        st.subheader("诊断报告")
        render_fb(fb)

# ============ Tab2：标准对比 ============
with tab2:
    st.markdown("同时提供标准示范与学员动作，系统在**击球时刻自动对齐**两条曲线，量化差距。")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**① 标准示范视频**（教练 / 专业运动员）")
        std = load_video_bytes("std", "上传标准视频")
    with cB:
        st.markdown("**② 学员视频**")
        stu = load_video_bytes("stu", "上传学员视频")

    if std and stu and st.button("🚀 开始对比分析", key="btn_cmp"):
        p1 = st.progress(0, text="分析标准视频……")
        a_std = analyze_video(std, label="std")
        p1.progress(60, text="分析学员视频……")
        a_stu = analyze_video(stu, label="stu")
        p1.progress(100, text="完成")
        if a_std and a_stu:
            st.session_state["cmp"] = (a_std, a_stu)
        else:
            st.error("至少一段视频未能识别出人体，请检查画面清晰度。")

    if "cmp" in st.session_state:
        a_std, a_stu = st.session_state["cmp"]
        # 对比模式取各自视频的第一拍（若检测到多次击球则提示）
        act_std = a_std["segments"][0]["action"]
        act_stu = a_stu["segments"][0]["action"]
        a_std["action"], a_stu["action"] = act_std, act_stu
        if len(a_std["segments"]) > 1 or len(a_stu["segments"]) > 1:
            st.caption("ℹ️ 视频中检测到多次击球（标准 %d 拍 / 学员 %d 拍），对比采用各自的第 1 拍。"
                       % (len(a_std["segments"]), len(a_stu["segments"])))
        if act_std != act_stu:
            st.warning("⚠️ 两段视频识别出的动作类型不同（标准：%s / 学员：%s），曲线对比仅供参考。"
                       % (ACTION_PROFILES[act_std]["name"], ACTION_PROFILES[act_stu]["name"]))
        else:
            st.markdown("### %s 双方动作一致：**%s**" % (ACTION_PROFILES[act_stu]["emoji"], ACTION_PROFILES[act_stu]["name"]))
        score_std, _ = score_and_feedback(a_std, act_std)
        score_stu, fb_stu = score_and_feedback(a_stu, act_stu)

        st.subheader("关键指标对比")
        rows = [
            ("综合评分", "%.0f" % score_std, "%.0f" % score_stu),
            ("肘部峰值角度", "%.0f°" % a_std["peak_elbow"], "%.0f°" % a_stu["peak_elbow"]),
            ("引拍肘角", "%.0f°" % a_std["min_elbow"], "%.0f°" % a_stu["min_elbow"]),
            ("挥拍角速度", "%.0f°/s" % a_std["swing_speed"], "%.0f°/s" % a_stu["swing_speed"]),
            ("击球点提升", "%.0f%%" % (a_std["impact_lift"] * 100), "%.0f%%" % (a_stu["impact_lift"] * 100)),
        ]
        tdf = pd.DataFrame(rows, columns=["指标", "标准动作", "学员动作"])
        tdf["差距"] = [
            "%s" % ("↑ 更好" if b >= a_ else "↓ 需改进")
            for a_, b in [(score_std, score_stu), (a_std["peak_elbow"], a_stu["peak_elbow"]),
                          (a_std["min_elbow"], a_stu["min_elbow"]),
                          (a_std["swing_speed"], a_stu["swing_speed"]),
                          (a_std["impact_lift"], a_stu["impact_lift"])]
        ]
        st.dataframe(tdf, use_container_width=True)

        st.subheader("击球时刻对齐的角度曲线（0 = 击球瞬间）")
        df = pd.concat(
            [aligned_series(a_std, "elbow").rename("标准·肘部"),
             aligned_series(a_stu, "elbow").rename("学员·肘部"),
             aligned_series(a_std, "shoulder").rename("标准·肩部"),
             aligned_series(a_stu, "shoulder").rename("学员·肩部")],
            axis=1,
        )
        st.line_chart(df, height=340)
        st.caption("若学员曲线的「肘角冲高峰」更平缓或滞后，说明鞭打发力不充分或节奏偏慢。")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**标准示范 · 击球帧**")
            if a_std["jpgs"][min(a_std["impact"], len(a_std["jpgs"]) - 1)]:
                st.image(a_std["jpgs"][min(a_std["impact"], len(a_std["jpgs"]) - 1)],
                         use_column_width=True)
        with c2:
            st.markdown("**学员动作 · 击球帧**")
            if a_stu["jpgs"][min(a_stu["impact"], len(a_stu["jpgs"]) - 1)]:
                st.image(a_stu["jpgs"][min(a_stu["impact"], len(a_stu["jpgs"]) - 1)],
                         use_column_width=True)

        st.subheader("针对学员的改进建议")
        render_fb(fb_stu)

# ============ Tab3：动作标准库（v3 新增） ============
with tab3:
    st.markdown("### 🎯 以「正确动作」为基准的个性化评价")
    st.markdown(
        "分析一段标准动作视频（教练/专业运动员，或你自己认可的规范动作），选择其中一拍"
        "**保存为该动作的标准模板**。之后学员视频将自动与模板做 DTW 形状匹配——"
        "评价基准从「教科书常模」变为「你定义的标准动作」。"
    )
    tpls = load_templates()

    # —— 已有模板管理 ——
    if tpls:
        st.markdown("#### 📚 已保存的标准模板")
        for key, t in tpls.items():
            c1, c2, c3 = st.columns([2, 2, 1])
            P = ACTION_PROFILES.get(t.get("action", key), {"name": t.get("name", key), "emoji": "📁"})
            with c1:
                st.markdown("%s **%s**" % (P["emoji"], t.get("name", key)))
            with c2:
                st.caption("保存于 %s · %d 个采样点" % (t.get("saved_at", "-"), len(t.get("elbow", []))))
            with c3:
                if st.button("删除", key="del_" + key):
                    delete_template(key)
                    st.rerun()
        st.markdown("---")

    # —— 上传标准视频建立模板 ——
    st.markdown("#### 🎬 方式一：从标准动作视频建立模板")
    std_data = load_video_bytes("tpl", "上传标准动作视频")
    if std_data and st.button("🚀 分析标准视频", key="btn_tpl"):
        with st.spinner("正在逐帧姿态估计与动作分割……"):
            ta = analyze_video(std_data, label="tpl")
        if ta is None:
            st.error("未能检测到人体关键点，请确认视频中人物全身清晰、光线充足。")
        else:
            st.session_state["tpl_result"] = ta

    if "tpl_result" in st.session_state:
        ta = st.session_state["tpl_result"]
        tsegs = ta["segments"]
        st.markdown("检测到 **%d 次击球**，请选择一拍作为标准模板：" % len(tsegs))
        tlabels = ["第%d拍 · %s（肘峰 %.0f°，挥速 %.0f°/s）"
                   % (i + 1, ACTION_PROFILES[s["action"]]["name"], s["peak_elbow"], s["swing_speed"])
                   for i, s in enumerate(tsegs)]
        tsel = st.selectbox("选择标准拍", tlabels, key="tpl_sel")
        si = tlabels.index(tsel)
        seg = tsegs[si]
        act = seg["action"]

        cA, cB = st.columns([1, 1])
        with cA:
            st.subheader("该拍骨架回放")
            seg_jpgs = ta["jpgs"][seg["win_lo"]: seg["win_hi"] + 1]
            show_frame_browser(seg_jpgs, seg["impact"] - seg["win_lo"], "tplseg%d" % si)
        with cB:
            st.subheader("该拍角度曲线（将保存为模板）")
            st.line_chart(angle_chart_df(seg), height=300)

        tpl_name = st.text_input("模板名称", value=ACTION_PROFILES[act]["name"] + "·标准")
        if st.button("💾 保存为「%s」的标准模板" % ACTION_PROFILES[act]["name"], key="btn_save_tpl"):
            tpl_data = {"name": tpl_name, "action": act,
                        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
            for kk in JOINT_KEYS:
                tpl_data[kk] = [round(float(v), 2) for v in seg[kk]]
            save_template(act, tpl_data)
            st.success("已保存！此后「单人动作分析」中识别为%s的片段将以该模板（六关节）为评价基准。"
                       % ACTION_PROFILES[act]["name"])
            st.rerun()

    # —— 从角度数据 CSV 直接导入模板（方式二） ——
    st.markdown("#### 📄 方式二：从关节角度数据导入（CSV）")
    st.caption(
        "适用于已有标准动作角度数据的情况（测角仪采集、文献数据、其他系统导出）。"
        "CSV 每行一个采样点，**至少需包含肘部角度列**（列名含 elbow 或「肘」），"
        "肩/腕/髋/膝/踝列可选（列名含 shoulder/肩、wrist/腕、hip/髋、knee/膝、ankle/踝 均可）。"
        "数据需覆盖**引拍 → 击球 → 收拍**的完整动作周期。"
    )
    csv_f = st.file_uploader("上传角度数据 CSV", type=["csv"], key="tpl_csv")
    if csv_f is not None:
        try:
            df_csv = pd.read_csv(csv_f)
            parsed, err = parse_angle_csv(df_csv)
        except Exception:
            parsed, err = None, "CSV 文件读取失败，请确认是标准 CSV 格式（逗号分隔）"
        if err:
            st.error(err)
        else:
            n_pts = len(next(iter(parsed.values())))
            st.line_chart(angle_chart_df(parsed), height=260)
            st.caption("预览：%d 个采样点，识别到 %s。曲线应呈现「引拍低谷 → 击球峰值 → 收拍回落」形态。"
                       % (n_pts, "、".join(JOINT_CN[k] for k in parsed)))
            cA, cB = st.columns(2)
            with cA:
                act_sel = st.selectbox(
                    "该标准对应哪种动作",
                    list(ACTION_PROFILES.keys()),
                    format_func=lambda kk: ACTION_PROFILES[kk]["name"],
                    key="csv_act",
                )
            with cB:
                csv_name = st.text_input("模板名称", value=ACTION_PROFILES[act_sel]["name"] + "·数据导入",
                                         key="csv_tpl_name")
            if st.button("💾 保存为标准模板", key="btn_csv_tpl"):
                tpl_data = {"name": csv_name, "action": act_sel,
                            "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
                for kk, vv in parsed.items():
                    tpl_data[kk] = [round(float(v), 2) for v in vv]
                save_template(act_sel, tpl_data)
                st.success("已保存！「单人动作分析」中识别为%s的片段将以该数据为评价基准。"
                           % ACTION_PROFILES[act_sel]["name"])
                st.rerun()

# ============ Tab4：系统说明 ============
with tab4:
    st.markdown(
        """
#### 技术架构
- **姿态估计**：MediaPipe Pose（33 个身体关键点，含置信度滤波与平滑）
- **六关节动力链**：肘、肩、腕、髋、膝、踝——覆盖「踝蹬转 → 膝蹬伸 → 髋转体 → 肩挥臂 → 肘鞭打 → 腕控拍」完整发力链
- **动作分割**：基于手腕垂直速度多峰检测，自动切分视频中的每一次击球（最多 6 拍）
- **动作识别**：基于运动学特征（击球点高度、挥拍角速度、击球前后手腕轨迹）的决策树分类，支持高远球/杀球/吊球/挑球四类
- **评分引擎（双基准）**：
  - 有标准模板 → **六关节加权 DTW 模板匹配**（z-score 归一化后比曲线形状，按动力链环节逐段诊断）
  - 无标准模板 → 运动生物力学**常模规则库**（肘部伸展、转体幅度、挥速、击球点高度、随挥轨迹）
- **标准模板来源（两种）**：① 标准动作视频分析后一键保存；② 直接导入关节角度数据（CSV，六关节任意组合，支持中英文列名）
- **交互界面**：Streamlit 单页应用，支持逐拍清单、逐帧回放、六关节曲线勾选与标准库管理

#### 六关节与 DTW 权重
| 关节 | 权重 | 生物力学意义 |
|---|---|---|
| 肘部 | 0.24 | 鞭打链条末端，决定力量传导与出球速度 |
| 肩部 | 0.18 | 躯干转体与肩臂挥动幅度 |
| 手腕 | 0.16 | 拍面控制与最后鞭打发力 |
| 髋部 | 0.16 | 下肢到上肢的转体中转站 |
| 膝部 | 0.16 | 蹬地蓄力与重心控制 |
| 踝部 | 0.10 | 蹬转起点与步伐稳定性 |

#### DTW 模板匹配是什么
动态时间规整（Dynamic Time Warping）是语音识别与运动分析领域的经典算法：
把两条**快慢不同**的动作曲线在时间轴上弹性对齐后再比较形状，因此学员挥拍慢一点、教练快一点，
也能公平比较"动作做得像不像"。

**单关节相似度 = 形状 35% + 幅度 65%**：
- **形状维度**（DTW，z-score 归一化）：只比曲线的起伏模式，容忍快慢差异
- **幅度维度**（均值差 + 动态范围差）：识别"形状像但整体角度差几十度"的情况

这样设计的理由：动作节奏因人而异可以宽容，但关节角度脱离标准（如膝角差 45°）就是实打实的
技术问题——两者必须分开评价，诊断建议也会区分"时序问题"和"幅度问题"。

#### 四种动作的识别逻辑
| 动作 | 关键特征 |
|---|---|
| 杀球 | 击球点远高于肩 + 挥拍角速度极快（≥420°/s）|
| 挑球 | 击球后手腕持续上升（下手位向上发力）|
| 吊球 | 挥拍角速度慢（<260°/s，减力切削）|
| 高远球 | 完整鞭打动作：挥速快 + 击球后随挥下落 |

#### 使用建议
- 视频要求：侧面机位（持拍侧朝镜头）、全身入镜、光线充足
- 标准模板：同机位同角度录制，模板质量直接决定评价可信度
- CSV 导入：数据需覆盖完整动作周期（引拍→击球→收拍），列名含 elbow/肘 与 shoulder/肩 即可
- 对比模式：标准视频建议选取教练或专业运动员的**同类**动作

#### 版本
- v4.0：六关节动力链分析——特征扩展到肘/肩/腕/髋/膝/踝；骨架图标注多关节实时角度；曲线可勾选关节；
        模板匹配升级为六关节加权 DTW；诊断报告按动力链环节（下肢→转体→挥臂→手腕）逐段反馈
- v3.0：动作标准库——标准视频一键建模板 / CSV 角度数据导入；评分引擎升级为 DTW 模板匹配
- v2.2：连续多动作分割——一段视频多次击球自动逐拍切分、识别、评分
- v2.1：动作自动识别（四类）、专项规则库评分、动作一致性校验
- v2.0：标准动作对比、击球时刻自动对齐、角度–时间曲线、逐帧浏览器与诊断报告
        """
    )
