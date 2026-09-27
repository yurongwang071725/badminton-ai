import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import mediapipe as mp


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility
            })
        return landmarks

    def get_key_points(self, landmarks):
        if not landmarks:
            return None
        return {
            'left_shoulder': (landmarks[11]['x'], landmarks[11]['y']),
            'right_shoulder': (landmarks[12]['x'], landmarks[12]['y']),
            'left_elbow': (landmarks[13]['x'], landmarks[13]['y']),
            'right_elbow': (landmarks[14]['x'], landmarks[14]['y']),
            'left_wrist': (landmarks[15]['x'], landmarks[15]['y']),
            'right_wrist': (landmarks[16]['x'], landmarks[16]['y']),
            'left_hip': (landmarks[23]['x'], landmarks[23]['y']),
            'right_hip': (landmarks[24]['x'], landmarks[24]['y']),
        }

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        return frame

    def close(self):
        self.pose.close()


class BadmintonAnalyzer:
    # 动作标准库：每种击球动作的理想参数区间与技术要点
    STANDARD_LIBRARY = {
        '高远球': {
            'desc': '侧身架拍充分，击球点在头顶后上方，手腕充分伸展，力量从脚→腰→肩→腕依次传递。',
            'ranges': {
                'elbow_angle': (120, 170),
                'shoulder_angle': (80, 120),
                'wrist_height': (0.2, 0.6),
                'hip_rotation': (15, 45),
            }
        },
        '杀球': {
            'desc': '充分引拍至头顶后方，身体后仰成弓形，击球点高且靠前，甩腕爆发用力。',
            'ranges': {
                'elbow_angle': (90, 140),
                'shoulder_angle': (60, 110),
                'wrist_height': (0.5, 0.9),
                'hip_rotation': (30, 60),
            }
        },
        '吊球': {
            'desc': '动作隐蔽，击球点在身体前上方，手腕轻切，控制落点在对方前场。',
            'ranges': {
                'elbow_angle': (110, 150),
                'shoulder_angle': (70, 110),
                'wrist_height': (0.3, 0.6),
                'hip_rotation': (15, 40),
            }
        },
        '平高球': {
            'desc': '击球点略低于高远球，拍面稍前倾，弧度平快，压制对方后场。',
            'ranges': {
                'elbow_angle': (110, 160),
                'shoulder_angle': (80, 120),
                'wrist_height': (0.3, 0.55),
                'hip_rotation': (15, 45),
            }
        },
        '网前搓球': {
            'desc': '重心降低，前臂贴近身体，手腕灵活搓切，使球在网前翻滚过网。',
            'ranges': {
                'elbow_angle': (90, 130),
                'shoulder_angle': (40, 90),
                'wrist_height': (0.0, 0.3),
                'hip_rotation': (0, 20),
            }
        },
    }

    METRIC_NAMES = {
        'elbow_angle': '肘部角度 (°)',
        'shoulder_angle': '肩部角度 (°)',
        'wrist_height': '击球点高度',
        'hip_rotation': '髋部旋转 (°)',
    }

    @staticmethod
    def calculate_angle(p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def analyze_frame(self, key_points):
        if not key_points:
            return None
        shoulder = key_points['right_shoulder']
        elbow = key_points['right_elbow']
        wrist = key_points['right_wrist']
        hip = key_points['right_hip']

        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = self.calculate_angle(hip, shoulder, elbow)
        wrist_height = 1.0 - wrist[1]
        hip_rotation = abs(hip[0] - key_points['left_hip'][0]) * 100

        return {
            'elbow_angle': elbow_angle,
            'shoulder_angle': shoulder_angle,
            'wrist_height': wrist_height,
            'hip_rotation': hip_rotation,
        }

    def score_action(self, metrics, action_type='高远球'):
        if not metrics:
            return 0, []
        ranges = self.STANDARD_LIBRARY[action_type]['ranges']
        scores = []
        feedback = []
        for metric, (low, high) in ranges.items():
            if metric not in metrics:
                continue
            value = metrics[metric]
            name = self.METRIC_NAMES[metric]
            if low <= value <= high:
                scores.append(100)
                feedback.append(f"✅ {name} {value:.1f} 处于标准区间 [{low}, {high}]")
            elif value < low:
                scores.append(60)
                feedback.append(f"⚠️ {name} {value:.1f} 偏低（标准 {low}~{high}）")
            else:
                scores.append(70)
                feedback.append(f"⚠️ {name} {value:.1f} 偏高（标准 {low}~{high}）")
        score = sum(scores) / len(scores) if scores else 0
        return score, feedback

    def generate_report(self, scores_list):
        if not scores_list:
            return "未能检测到有效动作"
        avg_score = sum(scores_list) / len(scores_list)
        max_score = max(scores_list)
        min_score = min(scores_list)

        report = f"""
## 综合评估报告

- **平均得分**: {avg_score:.1f} / 100
- **最高得分**: {max_score:.1f}
- **最低得分**: {min_score:.1f}
- **检测帧数**: {len(scores_list)}
"""
        if avg_score >= 85:
            report += "\n**优秀** - 动作标准，继续保持！\n"
        elif avg_score >= 70:
            report += "\n**良好** - 动作基本合格，仍有提升空间\n"
        elif avg_score >= 60:
            report += "\n**合格** - 建议加强基础动作练习\n"
        else:
            report += "\n**需改进** - 建议从基础动作开始练习\n"
        return report


def extract_video_frames(video_path, max_frames=150):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h))


st.set_page_config(page_title="羽毛球 AI 教学系统", page_icon="🏸", layout="wide")
st.title("🏸 羽毛球高远球动作智能化教学辅助系统")
st.markdown("**基于姿态估计的实时诊断与反馈系统** — 大学生创新创业训练计划项目")
st.divider()

# ── 动作类型分类选择 ───────────────────────────────────────────────
ACTION_TYPES = list(BadmintonAnalyzer.STANDARD_LIBRARY.keys())
action_type = st.selectbox(
    "选择要分析的动作类型",
    ACTION_TYPES,
    index=0,
    help="不同击球动作有各自的标准参数，选择后评测将按该动作标准进行"
)

# ── 动作标准库 ─────────────────────────────────────────────────────
with st.expander("📚 动作标准库（点击展开）"):
    st.markdown(f"**当前所选「{action_type}」技术要点：**  {BadmintonAnalyzer.STANDARD_LIBRARY[action_type]['desc']}")
    st.markdown("---")
    rows = []
    for atype, info in BadmintonAnalyzer.STANDARD_LIBRARY.items():
        rng = info['ranges']
        rows.append({
            '动作类型': atype,
            '肘部角度': f"{rng['elbow_angle'][0]}~{rng['elbow_angle'][1]}",
            '肩部角度': f"{rng['shoulder_angle'][0]}~{rng['shoulder_angle'][1]}",
            '击球点高度': f"{rng['wrist_height'][0]}~{rng['wrist_height'][1]}",
            '髋部旋转': f"{rng['hip_rotation'][0]}~{rng['hip_rotation'][1]}",
        })
    st.table(rows)
    st.caption("数值说明：角度单位为度；击球点高度以归一化坐标表示（越接近 1 表示越靠近画面顶部/越高）；髋部旋转为左右髋水平分离度的相对值。")

uploaded_file = st.file_uploader(
    "上传视频文件（点击下方按钮选择）",
    type=['mp4', 'avi', 'mov'],
    help="建议上传 5-15 秒的羽毛球动作视频"
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始视频")
        st.video(uploaded_file)

    if st.button("🚀 开始分析", type="primary"):
        with st.spinner("AI 正在分析你的动作..."):
            progress_bar = st.progress(0)
            detector = PoseDetector()
            analyzer = BadmintonAnalyzer()

            frames = extract_video_frames(video_path)
            total_frames = len(frames)

            if total_frames == 0:
                st.error("无法读取视频，请检查文件格式")
            else:
                scores_list = []
                feedback_list = []
                analyzed_frames = []
                metrics_list = []

                for i, frame in enumerate(frames):
                    frame = resize_frame(frame, width=640)
                    results = detector.detect(frame)
                    landmarks = detector.get_landmarks(results)
                    key_points = detector.get_key_points(landmarks)
                    metrics = analyzer.analyze_frame(key_points)

                    if metrics:
                        metrics_list.append(metrics)
                        score, feedback = analyzer.score_action(metrics, action_type)
                        scores_list.append(score)
                        feedback_list.extend(feedback)
                        frame = detector.draw_landmarks(frame, results)
                        cv2.putText(frame, f"Score: {score:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    analyzed_frames.append(frame)
                    progress_bar.progress((i + 1) / total_frames)

                detector.close()

                with col2:
                    st.subheader("AI 分析结果")
                    if analyzed_frames:
                        mid_frame = analyzed_frames[len(analyzed_frames) // 2]
                        st.image(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB),
                                 caption="姿态估计可视化")

                st.divider()
                report = analyzer.generate_report(scores_list)
                st.markdown(report)

                # ── 标准动作对比 ─────────────────────────────────────
                if metrics_list:
                    st.subheader(f"🎯 标准动作对比（{action_type}）")
                    ranges = analyzer.STANDARD_LIBRARY[action_type]['ranges']
                    comp_rows = []
                    for metric, name in analyzer.METRIC_NAMES.items():
                        vals = [m[metric] for m in metrics_list if metric in m]
                        user_val = sum(vals) / len(vals) if vals else None
                        rng = ranges[metric]
                        if user_val is None:
                            status = "无数据"
                        elif rng[0] <= user_val <= rng[1]:
                            status = "✅ 达标"
                        elif user_val < rng[0]:
                            status = "⚠️ 偏低"
                        else:
                            status = "⚠️ 偏高"
                        comp_rows.append({
                            '指标': name,
                            '用户实测(均值)': f"{user_val:.1f}" if user_val is not None else "-",
                            '标准区间': f"{rng[0]}~{rng[1]}",
                            '评价': status,
                        })
                    st.table(comp_rows)

                st.subheader("详细动作反馈")
                if feedback_list:
                    seen = set()
                    for fb in feedback_list:
                        if fb not in seen:
                            st.write(fb)
                            seen.add(fb)
                else:
                    st.info("未检测到有效动作关键点，请调整拍摄角度后重试")

                st.subheader("评分曲线")
                if scores_list:
                    st.line_chart(scores_list)

                st.success("分析完成！")

    try:
        os.unlink(video_path)
    except Exception:
        pass

else:
    st.info("👆 请上传视频文件开始分析")

    with st.expander("使用说明"):
        st.markdown("""
        1. 选择要分析的动作类型（高远球 / 杀球 / 吊球 / 平高球 / 网前搓球）
        2. 录制 5-15 秒的羽毛球动作视频
        3. 建议正面或侧面拍摄，背景简洁
        4. 保持身体完整在画面内
        5. 点击"开始分析"按钮
        6. 等待 AI 分析完成，查看评分、标准对比与反馈
        """)

    with st.expander("系统功能"):
        st.markdown("""
        - 基于 MediaPipe 的实时姿态估计
        - **动作类型分类**：支持 5 种常见击球动作
        - **动作标准库**：内置每种动作的理想参数与技术要点
        - **标准动作对比**：将你的动作与标准区间逐项比对
        - 肘部角度、肩部角度、击球点高度、髋部旋转多维度评分
        - 自动评分与个性化反馈建议
        """)

st.divider()
st.markdown("羽毛球高远球 AI 教学辅助系统 | 大创项目作品")
