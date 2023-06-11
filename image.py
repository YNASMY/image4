import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Mediapipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# サイドバーにタイトルを表示
st.sidebar.title("姿勢推定アプリ")

# 画像ファイルをアップロードする
uploaded_file = st.sidebar.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"])

# 画像ファイルがアップロードされた場合の処理
if uploaded_file is not None:
    # 画像ファイルをOpenCVで読み込む
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # MediapipeのPose推定のインスタンスを作成
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # 画像をRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 画像をPose推定に渡す
        results = pose.process(image_rgb)

        # 推定結果を描画する
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                               thickness=2,
                                                                               circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                  thickness=2))

    # 描画結果を表示する
    st.image(annotated_image, channels="BGR", use_column_width=True, caption='Pose Estimation Result')

else:
    st.sidebar.info("画像ファイルをアップロードしてください")

