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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility})
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
    IDEAL_RANGES = {
        'elbow_angle': (120, 170),
        'shoulder_angle': (80, 120),
        'wrist_height': (0.2, 0.6),
    }

    @staticmethod
    def calculate_angle(p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
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
            'hip_rotation': hip_rotation
        }

    def score_action(self, metrics):
        if not metrics:
            return 0, []
        scores = []
        feedback = []

        angle = metrics['elbow_angle']
        if self.IDEAL_RANGES['elbow_angle'][0] <= angle <= self.IDEAL_RANGES['elbow_angle'][1]:
            scores.append(100)
            feedback.append(f"挥拍肘部角度良好 ({angle:.1f}°)")
        elif angle < self.IDEAL_RANGES['elbow_angle'][0]:
            scores.append(60)
            feedback.append(f"肘部角度偏小 ({angle:.1f}°)，挥拍时肘部应充分抬起")
        else:
            scores.append(70)
            feedback.append(f"肘部角度过大 ({angle:.1f}°)，注意控制挥拍幅度")

        angle = metrics['shoulder_angle']
        if self.IDEAL_RANGES['shoulder_angle'][0] <= angle <= self.IDEAL_RANGES['shoulder_angle'][1]:
            scores.append(100)
            feedback.append(f"肩部外展角度合适 ({angle:.1f}°)")
        else:
            scores.append(70)
            feedback.append(f"肩部角度需要调整 ({angle:.1f}°)")

        height = metrics['wrist_height']
        if height >= self.IDEAL_RANGES['wrist_height'][1]:
            scores.append(100)
            feedback.append(f"击球点高度充分 ({height:.2f})")
        else:
            scores.append(60)
            feedback.append(f"击球点偏低 ({height:.2f})，应充分引拍")

        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score, feedback

    def generate_report(self, scores_list):
        if not scores_list:
            return "未能检测到有效动作"
        avg = sum(scores_list) / len(scores_list)
        max_score = max(scores_list)
        min_score = min(scores_list)

        report = f"""
## 综合评估报告

- **平均得分**: {avg:.1f} / 100
- **最高得分**: {max_score:.1f}
- **最低得分**: {min_score:.1f}
- **检测帧数**: {len(scores_list)}
"""
        if avg >= 85:
            report += "\n**优秀** - 动作标准，继续保持！\n"
        elif avg >= 70:
            report += "\n**良好** - 动作基本合格，仍有提升空间\n"
        elif avg >= 60:
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

uploaded_file = st.file_uploader(
    "上传视频文件（点击下方按钮选择）",
    type=['mp4', 'avi', 'mov'],
    help="建议上传 5-15 秒的羽毛球高远球动作视频"
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

                for i, frame in enumerate(frames):
                    frame = resize_frame(frame, width=640)
                    results = detector.detect(frame)
                    landmarks = detector.get_landmarks(results)
                    key_points = detector.get_key_points(landmarks)
                    metrics = analyzer.analyze_frame(key_points)

                    if metrics:
                        score, feedback = analyzer.score_action(metrics)
                        scores_list.append(score)
                        frame = detector.draw_landmarks(frame, results)
                        cv2.putText(frame, f"Score: {score:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if i == total_frames - 1:
                            feedback_list = feedback

                    analyzed_frames.append(frame)
                    progress_bar.progress((i + 1) / total_frames)

                detector.close()

                with col2:
                    st.subheader("AI 分析结果")
                    if analyzed_frames:
                        mid_frame = analyzed_frames[len(analyzed_frames)//2]
                        st.image(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB),
                                caption="姿态估计可视化")

                st.divider()
                report = analyzer.generate_report(scores_list)
                st.markdown(report)

                st.subheader("详细动作反馈")
                for fb in feedback_list:
                    st.write(fb)

                st.subheader("评分曲线")
                if scores_list:
                    st.line_chart(scores_list)

                st.success("分析完成！")

    try:
        os.unlink(video_path)
    except:
        pass

else:
    st.info("👆 请上传视频文件开始分析")

    with st.expander("使用说明"):
        st.markdown("""
        1. 录制 5-15 秒的羽毛球高远球动作视频
        2. 建议正面或侧面拍摄，背景简洁
        3. 保持身体完整在画面内
        4. 点击"开始分析"按钮
        5. 等待 AI 分析完成
        """)

    with st.expander("系统功能"):
        st.markdown("""
        - 基于 MediaPipe 的实时姿态估计
        - 肘部角度、肩部角度、击球点高度等多维度分析
        - 自动评分与个性化反馈建议

        **分析维度**：
        - 挥拍肘部角度（理想 120-170°）
        - 肩部外展角度（理想 80-120°）
        - 击球点高度
        - 髋部旋转角度
        """)

st.divider()
st.markdown("羽毛球高远球 AI 教学辅助系统 | 大创项目作品")