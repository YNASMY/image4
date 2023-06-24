import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math

# Mediapipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# タイトルを表示
st.title("フォームチェッカー")

# STEP1: モデル画像の姿勢推定と角度計算
st.header("STEP1: モデル画像の姿勢推定")

# モデル画像ファイルをアップロードする
uploaded_model_file = st.file_uploader("モデル画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# モデル画像ファイルがアップロードされた場合の処理
if uploaded_model_file is not None:
    # モデル画像ファイルをOpenCVで読み込む
    model_image = cv2.imdecode(np.frombuffer(uploaded_model_file.read(), np.uint8), 1)

    # MediapipeのPose推定のインスタンスを作成
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # モデル画像をRGBに変換
        model_image_rgb = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)

        # モデル画像をPose推定に渡す
        model_results = pose.process(model_image_rgb)

        # 推定結果を描画する
        annotated_model_image = model_image.copy()
        mp_drawing.draw_landmarks(annotated_model_image, model_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                               thickness=2,
                                                                               circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                  thickness=2))

        # 肩、肘、手首、腰の座標を取得
        model_shoulder_px = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * annotated_model_image.shape[1])
        model_shoulder_py = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * annotated_model_image.shape[0])
        model_elbow_px = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * annotated_model_image.shape[1])
        model_elbow_py = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * annotated_model_image.shape[0])
        model_wrist_px = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * annotated_model_image.shape[1])
        model_wrist_py = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * annotated_model_image.shape[0])
        model_hip_px = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * annotated_model_image.shape[1])
        model_hip_py = int(model_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * annotated_model_image.shape[0])

        # 肩、肘、手首、腰の角度を計算
        model_angle = math.degrees(math.atan2(model_wrist_py - model_elbow_py, model_wrist_px - model_elbow_px) -
                                   math.atan2(model_shoulder_py - model_elbow_py, model_shoulder_px - model_elbow_px))
        model_hip_angle = math.degrees(math.atan2(model_shoulder_py - model_hip_py, model_shoulder_px - model_hip_px))

        # 推定結果と角度を表示
        st.image(annotated_model_image, channels="BGR", use_column_width=True, caption='Model Pose Estimation Result')
        st.write(f"肩-肘-手首の角度: {model_angle:.2f} 度")
        st.write(f"肩-腰の角度: {model_hip_angle:.2f} 度")
else:
    st.warning("モデル画像をアップロードしてください")

# STEP2: 本人画像の姿勢推定と角度計算
st.header("STEP2: 本人画像の姿勢推定")

# 本人画像ファイルをアップロードする
uploaded_user_file = st.file_uploader("本人画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# 本人画像ファイルがアップロードされた場合の処理
if uploaded_user_file is not None:
    # 本人画像ファイルをOpenCVで読み込む
    user_image = cv2.imdecode(np.frombuffer(uploaded_user_file.read(), np.uint8), 1)

    # MediapipeのPose推定のインスタンスを作成
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # 本人画像をRGBに変換
        user_image_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

        # 本人画像をPose推定に渡す
        user_results = pose.process(user_image_rgb)

        # 推定結果を描画する
        annotated_user_image = user_image.copy()
        mp_drawing.draw_landmarks(annotated_user_image, user_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                               thickness=2,
                                                                               circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                  thickness=2))

        # 肩、肘、手首、腰の座標を取得
        user_shoulder_px = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * annotated_user_image.shape[1])
        user_shoulder_py = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * annotated_user_image.shape[0])
        user_elbow_px = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * annotated_user_image.shape[1])
        user_elbow_py = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * annotated_user_image.shape[0])
        user_wrist_px = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * annotated_user_image.shape[1])
        user_wrist_py = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * annotated_user_image.shape[0])
        user_hip_px = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * annotated_user_image.shape[1])
        user_hip_py = int(user_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * annotated_user_image.shape[0])

        # 肩、肘、手首、腰の角度を計算
        user_angle = math.degrees(math.atan2(user_wrist_py - user_elbow_py, user_wrist_px - user_elbow_px) -
                                  math.atan2(user_shoulder_py - user_elbow_py, user_shoulder_px - user_elbow_px))
        user_hip_angle = math.degrees(math.atan2(user_shoulder_py - user_hip_py, user_shoulder_px - user_hip_px))

        # 推定結果と角度を表示
        st.image(annotated_user_image, channels="BGR", use_column_width=True, caption='User Pose Estimation Result')
        st.write(f"肩-肘-手首の角度: {user_angle:.2f} 度")
        st.write(f"肩-腰の角度: {user_hip_angle:.2f} 度")
else:
    st.warning("本人画像をアップロードしてください")

# STEP3: モデル画像と本人画像の角度の違いを表示
st.header("STEP3: モデル画像と本人画像の角度の違い")

# モデル画像と本人画像の角度の違いを計算
angle_diff = model_angle - user_angle

# 角度の違いを表示
st.write(f"モデル画像と本人画像の角度の違い: {angle_diff:.2f} 度")

# STEP4: 本人画像にモデル画像と同じ角度を描画
st.header("STEP4: 本人画像にモデル画像と同じ角度を描画")

# 本人画像のコピーを作成
annotated_user_image_with_model = user_image.copy()

# モデル画像の角度と本人画像の角度が同じになるように点と線を描画
user_wrist_px_with_model = int(user_shoulder_px +
                               math.cos(math.radians(angle_diff)) * (user_wrist_px - user_shoulder_px) -
                               math.sin(math.radians(angle_diff)) * (user_wrist_py - user_shoulder_py))
user_wrist_py_with_model = int(user_shoulder_py +
                               math.sin(math.radians(angle_diff)) * (user_wrist_px - user_shoulder_px) +
                               math.cos(math.radians(angle_diff)) * (user_wrist_py - user_shoulder_py))

user_hip_px_with_model = int(user_shoulder_px +
                             math.cos(math.radians(angle_diff)) * (user_hip_px - user_shoulder_px) -
                             math.sin(math.radians(angle_diff)) * (user_hip_py - user_shoulder_py))
user_hip_py_with_model = int(user_shoulder_py +
                             math.sin(math.radians(angle_diff)) * (user_hip_px - user_shoulder_px) +
                             math.cos(math.radians(angle_diff)) * (user_hip_py - user_shoulder_py))

# 本人画像にモデル画像と同じ角度の点と線を描画
cv2.circle(annotated_user_image_with_model, (user_shoulder_px, user_shoulder_py), 2, (0, 255, 255), -1)
cv2.line(annotated_user_image_with_model, (user_shoulder_px, user_shoulder_py),
         (user_elbow_px, user_elbow_py), (0, 255, 255), 2)
cv2.line(annotated_user_image_with_model, (user_elbow_px, user_elbow_py),
         (user_wrist_px_with_model, user_wrist_py_with_model), (0, 255, 255), 2)
cv2.line(annotated_user_image_with_model, (user_shoulder_px, user_shoulder_py),
         (user_hip_px_with_model, user_hip_py_with_model), (0, 255, 255), 2)

# 本人画像と描画結果を並べて表示
st.image(np.concatenate((annotated_user_image, annotated_user_image_with_model), axis=1), channels="BGR", use_column_width=True, caption='User Image with Model Pose')
