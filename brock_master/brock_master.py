import os
import time
import threading
from collections import deque

import cv2
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32

from ultralytics import YOLO

from .webcam_utils import LatestFrameReader


# ============================================================
# ROSトピック
# ============================================================

IMAGE_TOPIC = "brock_Webcam_BBOX"
INFO_TOPIC = "brocks_info"
COLOR_TOPIC = "brock_color"
CONF_TOPIC = "brock_conf"


# ============================================================
# YOLOモデル設定
# ============================================================

MODEL_FILENAMES = [
    "red_brock.pt",
    "blue_brock.pt",
]

CONF_THRES = 0.01


# ============================================================
# Webカメラ設定
# ============================================================

WEBCAM_INDEX = 4
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720


# ============================================================
# 映像配信FPS
# ============================================================

IMAGE_FPS = 30.0


# ============================================================
# YOLO推論FPS
#
# 例えば10FPSなら約0.1秒に1回YOLOを実行
# ============================================================

YOLO_FPS = 10.0


# ============================================================
# 距離推定設定
# ============================================================

BOX_REAL_WIDTH_M = 0.22
BOX_REAL_HEIGHT_M = 0.30

WEBCAM_FOCAL_LENGTH_PX = 791.7

DISTANCE_HISTORY_SIZE = 4


# ============================================================
# アスペクト比設定
# ============================================================

EXPECTED_ASPECT_RATIO = (
    BOX_REAL_WIDTH_M / BOX_REAL_HEIGHT_M
)

ASPECT_RATIO_TOLERANCE = 0.4


# ============================================================
# 描画設定
# ============================================================

# BBOX
BOX_COLOR = (0, 165, 255)

# 目標X軸
# BGRで (0, 255, 0) = 緑
TARGET_X_COLOR = (0, 255, 0)

TARGET_X_THICKNESS = 2


# ============================================================
# モデル設定確認
# ============================================================

if len(MODEL_FILENAMES) < 1 or len(MODEL_FILENAMES) > 2:
    raise ValueError(
        "MODEL_FILENAMES は1つまたは2つ設定してください。"
    )


# ============================================================
# モデルパス
# ============================================================

MODEL_PATHS = [
    os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    for filename in MODEL_FILENAMES
]


# ============================================================
# モデル名
# ============================================================

MODEL_NAMES = [
    os.path.splitext(
        os.path.basename(path)
    )[0]
    for path in MODEL_PATHS
]


# ============================================================
# 色とモデルの対応
#
# red_brock.pt  -> red
# blue_brock.pt -> blue
# ============================================================

COLOR_TO_MODEL_INDEX = {
    name.split("_")[0].lower(): index
    for index, name in enumerate(MODEL_NAMES)
}


# ============================================================
# 距離推定
# ============================================================

def estimate_distance_from_height(pixel_height):

    if (
        pixel_height <= 0
        or WEBCAM_FOCAL_LENGTH_PX <= 0
    ):
        return 0.0

    distance = (
        BOX_REAL_HEIGHT_M
        * WEBCAM_FOCAL_LENGTH_PX
    ) / pixel_height

    return distance


# ============================================================
# アスペクト比の信頼性判定
# ============================================================

def is_aspect_ratio_reliable(
    pixel_width,
    pixel_height,
):

    if (
        pixel_width <= 0
        or pixel_height <= 0
    ):
        return False

    observed_ratio = (
        pixel_width / pixel_height
    )

    deviation = abs(
        observed_ratio
        / EXPECTED_ASPECT_RATIO
        - 1.0
    )

    return deviation <= ASPECT_RATIO_TOLERANCE


# ============================================================
# 段数を割り振る
# ============================================================

def assign_tiers(box_data_list):

    sorted_by_bottom = sorted(
        box_data_list,
        key=lambda box: box["y2"],
        reverse=True,
    )

    tier_by_id = {
        id(box): tier
        for tier, box in enumerate(
            sorted_by_bottom,
            start=1,
        )
    }

    return tier_by_id


# ============================================================
# メインノード
# ============================================================

class BrockMasterNode(Node):

    def __init__(self):

        super().__init__("brock_master")


        # ====================================================
        # Confidence
        # ====================================================

        self.conf_thres = CONF_THRES


        # ====================================================
        # スレッド用Lock
        #
        # YOLO側と映像配信側が
        # latest_box_dataを同時に触るため使用
        # ====================================================

        self.data_lock = threading.Lock()


        # ====================================================
        # 最新フレーム
        # ====================================================

        self.latest_frame = None


        # ====================================================
        # 最新YOLO検出結果
        # ====================================================

        self.latest_box_data = []


        # ====================================================
        # YOLOの最新情報
        # ====================================================

        self.latest_info_lines = []


        # ====================================================
        # YOLO推論時刻
        # ====================================================

        self.last_yolo_time = 0.0


        # ====================================================
        # 終了フラグ
        # ====================================================

        self.running = True


        # ====================================================
        # モデル
        # ====================================================

        self._load_models()


        # ====================================================
        # Webカメラ
        # ====================================================

        self._init_webcam()


        # ====================================================
        # Publisher / Subscriber
        # ====================================================

        self._init_publishers_and_subscribers()


        self.get_logger().info(
            "Starting inference..."
        )

        self.get_logger().info(
            f"Initial YOLO confidence: "
            f"{self.conf_thres:.2f}"
        )

        self.get_logger().info(
            f"Image FPS target: {IMAGE_FPS:.1f}"
        )

        self.get_logger().info(
            f"YOLO FPS target: {YOLO_FPS:.1f}"
        )


    # ========================================================
    # モデル読み込み
    # ========================================================

    def _load_models(self):
        """YOLOモデルを読み込む。"""

        self.get_logger().info(
            f"Loading {len(MODEL_PATHS)} model(s)..."
        )

        self.models = [
            YOLO(path)
            for path in MODEL_PATHS
        ]

        for model_name in MODEL_NAMES:

            self.get_logger().info(
                f"loaded model: {model_name}"
            )


        # 最初に使用するモデル
        self.active_model_index = 0

        self.get_logger().info(
            "Default active model: "
            f"{MODEL_NAMES[self.active_model_index]}"
        )


    # ========================================================
    # Webカメラ初期化
    # ========================================================

    def _init_webcam(self):
        """Webカメラと画像関連の初期化。"""

        self.webcam = LatestFrameReader(
            WEBCAM_INDEX,
            cv2.CAP_V4L2,
            WEBCAM_WIDTH,
            WEBCAM_HEIGHT,
        )


        # ====================================================
        # 距離履歴
        # ====================================================

        self.distance_history = deque(
            maxlen=DISTANCE_HISTORY_SIZE
        )


    # ========================================================
    # Publisher / Subscriber初期化
    # ========================================================

    def _init_publishers_and_subscribers(self):
        """ROSのPublisherとSubscriberを初期化する。"""


        # ----------------------------------------------------
        # Callback Group
        # ----------------------------------------------------

        self.color_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.conf_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.camera_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.yolo_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )


        # ----------------------------------------------------
        # Image Publisher
        # ----------------------------------------------------

        self.publisher = self.create_publisher(
            Image,
            IMAGE_TOPIC,
            10,
        )


        # ----------------------------------------------------
        # Info Publisher
        # ----------------------------------------------------

        self.info_publisher = self.create_publisher(
            String,
            INFO_TOPIC,
            10,
        )


        # ----------------------------------------------------
        # brock_color Subscriber
        # ----------------------------------------------------

        self.color_subscription = (
            self.create_subscription(
                String,
                COLOR_TOPIC,
                self.on_brock_color,
                10,
                callback_group=(
                    self.color_callback_group
                ),
            )
        )

        self.get_logger().info(
            f"Subscribed to {COLOR_TOPIC}"
        )


        # ----------------------------------------------------
        # brock_conf Subscriber
        # ----------------------------------------------------

        self.conf_subscription = (
            self.create_subscription(
                Float32,
                CONF_TOPIC,
                self.on_brock_conf,
                10,
                callback_group=(
                    self.conf_callback_group
                ),
            )
        )

        self.get_logger().info(
            f"Subscribed to {CONF_TOPIC}"
        )


        # ====================================================
        # カメラ取得タイマー
        #
        # ここではカメラ画像を取得するだけ
        # ====================================================

        self.camera_timer = self.create_timer(
            1.0 / IMAGE_FPS,
            self.on_camera_timer,
            callback_group=(
                self.camera_callback_group
            ),
        )


        # ====================================================
        # YOLOタイマー
        #
        # カメラとは別速度で推論
        # ====================================================

        self.yolo_timer = self.create_timer(
            1.0 / YOLO_FPS,
            self.on_yolo_timer,
            callback_group=(
                self.yolo_callback_group
            ),
        )


        self.get_logger().info(
            f"Publishing image on {IMAGE_TOPIC}"
        )

        self.get_logger().info(
            f"Publishing detection info on {INFO_TOPIC}"
        )


    # ========================================================
    # brock_color Callback
    # ========================================================

    def on_brock_color(self, msg):
        """
        brock_colorを受信したときに、
        使用するYOLOモデルを切り替える。
        """

        color = msg.data.strip().lower()


        # ----------------------------------------------------
        # 色チェック
        # ----------------------------------------------------

        if color not in COLOR_TO_MODEL_INDEX:

            self.get_logger().warning(
                f"Unknown color '{color}' received "
                f"on {COLOR_TOPIC}. "
                f"Valid colors: "
                f"{list(COLOR_TO_MODEL_INDEX.keys())}"
            )

            return


        # ----------------------------------------------------
        # モデル変更
        # ----------------------------------------------------

        new_index = COLOR_TO_MODEL_INDEX[color]

        if new_index != self.active_model_index:

            self.active_model_index = new_index

            self.get_logger().info(
                "Active model switched to: "
                f"{MODEL_NAMES[new_index]} "
                f"(color={color})"
            )


    # ========================================================
    # brock_conf Callback
    # ========================================================

    def on_brock_conf(self, msg):
        """
        brock_confを受信して、
        YOLOのConfidence thresholdを変更する。
        """

        confidence = float(msg.data)


        # 0.0～1.0に制限
        confidence = max(
            0.0,
            min(1.0, confidence),
        )


        # Confidence更新
        self.conf_thres = confidence


        self.get_logger().info(
            "YOLO confidence threshold "
            f"changed to: {self.conf_thres:.2f}"
        )


    # ========================================================
    # 距離平滑化
    # ========================================================

    def get_smoothed_distance(
        self,
        pixel_height,
        reliable,
    ):

        # 現在の距離を計算
        distance = estimate_distance_from_height(
            pixel_height
        )


        # 信頼できる距離だけ履歴に追加
        if reliable and distance > 0:

            self.distance_history.append(
                distance
            )


        # 履歴がなければ現在値を返す
        if not self.distance_history:

            return distance


        # 履歴の中央値を使用
        sorted_history = sorted(
            self.distance_history
        )

        middle_index = (
            len(sorted_history) // 2
        )

        return sorted_history[middle_index]


    # ========================================================
    # カメラタイマー
    #
    # YOLOとは完全に分離
    # ========================================================

    def on_camera_timer(self):

        # ----------------------------------------------------
        # カメラ画像取得
        # ----------------------------------------------------

        ret, frame = self.webcam.read()

        if not ret:
            return


        # ----------------------------------------------------
        # 最新フレームを保存
        # ----------------------------------------------------

        with self.data_lock:

            self.latest_frame = frame.copy()


        # ----------------------------------------------------
        # 最新BBOXを取得
        # ----------------------------------------------------

        with self.data_lock:

            box_data_list = [
                box.copy()
                for box in self.latest_box_data
            ]


        # ----------------------------------------------------
        # 段数計算
        # ----------------------------------------------------

        tier_by_id = assign_tiers(
            box_data_list
        )


        # ----------------------------------------------------
        # BBOX + 緑線描画
        # ----------------------------------------------------

        self._draw_boxes(
            frame,
            box_data_list,
            tier_by_id,
        )


        # ----------------------------------------------------
        # Image publish
        # ----------------------------------------------------

        self._publish_image(frame)


    # ========================================================
    # YOLOタイマー
    #
    # 映像配信とは別に動作
    # ========================================================

    def on_yolo_timer(self):

        # ----------------------------------------------------
        # 最新フレームを取得
        # ----------------------------------------------------

        with self.data_lock:

            if self.latest_frame is None:
                return

            frame = self.latest_frame.copy()


        # ----------------------------------------------------
        # YOLO推論
        # ----------------------------------------------------

        start_time = time.perf_counter()

        box_data_list = self._run_inference(
            frame
        )

        inference_time = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # 段数計算
        # ----------------------------------------------------

        tier_by_id = assign_tiers(
            box_data_list
        )


        # ----------------------------------------------------
        # 情報作成
        # ----------------------------------------------------

        info_lines = (
            self._build_info(
                box_data_list,
                tier_by_id,
            )
        )


        # ----------------------------------------------------
        # 最新BBOXを更新
        # ----------------------------------------------------

        with self.data_lock:

            self.latest_box_data = box_data_list

            self.latest_info_lines = info_lines


        # ----------------------------------------------------
        # brocks_info送信
        # ----------------------------------------------------

        self._publish_info(
            info_lines
        )


        # ----------------------------------------------------
        # 推論時間ログ
        # ----------------------------------------------------

        self.get_logger().debug(
            f"YOLO inference: "
            f"{inference_time * 1000:.1f} ms"
        )


    # ========================================================
    # YOLO推論
    # ========================================================

    def _run_inference(self, frame):

        # ----------------------------------------------------
        # 現在使用しているモデル
        # ----------------------------------------------------

        model_index = self.active_model_index

        model = self.models[model_index]

        model_name = MODEL_NAMES[model_index]


        # ----------------------------------------------------
        # YOLO推論
        # ----------------------------------------------------

        results = model(
            frame,
            conf=self.conf_thres,
            verbose=False,
        )


        box_data_list = []


        # ----------------------------------------------------
        # 検出結果を処理
        # ----------------------------------------------------

        for result in results:

            for box in result.boxes:

                # --------------------------------------------
                # BBOX座標
                # --------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )


                # --------------------------------------------
                # BBOXサイズ
                # --------------------------------------------

                pixel_width = x2 - x1
                pixel_height = y2 - y1


                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = float(
                    box.conf[0].item()
                )


                # --------------------------------------------
                # アスペクト比
                # --------------------------------------------

                reliable = (
                    is_aspect_ratio_reliable(
                        pixel_width,
                        pixel_height,
                    )
                )


                # --------------------------------------------
                # 距離
                # --------------------------------------------

                distance = (
                    self.get_smoothed_distance(
                        pixel_height,
                        reliable,
                    )
                )


                # --------------------------------------------
                # 中心X
                # --------------------------------------------

                center_x = (
                    x1 + x2
                ) // 2


                # --------------------------------------------
                # データ保存
                # --------------------------------------------

                box_data_list.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "pixel_height": pixel_height,
                        "distance": distance,
                        "center_x": center_x,
                        "confidence": confidence,
                        "reliable": reliable,
                        "model_name": model_name,
                    }
                )


        return box_data_list


    # ========================================================
    # BBOX描画
    #
    # YOLO推論とは分離
    # ========================================================

    def _draw_boxes(
        self,
        frame,
        box_data_list,
        tier_by_id,
    ):

        # ====================================================
        # 目標X軸
        # ====================================================

        target_x = frame.shape[1] // 2


        cv2.line(
            frame,
            (target_x, 0),
            (
                target_x,
                frame.shape[0],
            ),
            TARGET_X_COLOR,
            TARGET_X_THICKNESS,
        )


        cv2.putText(
            frame,
            f"TARGET X = {target_x}",
            (
                target_x + 10,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TARGET_X_COLOR,
            2,
        )


        # ====================================================
        # BBOX
        # ====================================================

        for box_data in box_data_list:

            # ------------------------------------------------
            # 座標
            # ------------------------------------------------

            x1 = box_data["x1"]
            y1 = box_data["y1"]
            x2 = box_data["x2"]
            y2 = box_data["y2"]


            # ------------------------------------------------
            # 段数
            # ------------------------------------------------

            tier = tier_by_id[
                id(box_data)
            ]


            # ------------------------------------------------
            # BBOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                2,
            )


            # ------------------------------------------------
            # BBOX中心線
            # ------------------------------------------------

            cv2.line(
                frame,
                (
                    box_data["center_x"],
                    y1,
                ),
                (
                    box_data["center_x"],
                    y2,
                ),
                BOX_COLOR,
                2,
            )


            # ------------------------------------------------
            # 表示文字
            # ------------------------------------------------

            label_text = (
                f"距離:{box_data['distance']:.3f}m "
                f"段:{tier} "
                f"モデル:{box_data['model_name']} "
                f"高さ:{box_data['pixel_height']}px "
                f"信頼度:{box_data['confidence']:.2f}"
            )


            # ------------------------------------------------
            # アスペクト比
            # ------------------------------------------------

            if not box_data["reliable"]:

                label_text += " [傾き大]"


            # ------------------------------------------------
            # 文字描画
            # ------------------------------------------------

            cv2.putText(
                frame,
                label_text,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                BOX_COLOR,
                2,
            )


    # ========================================================
    # brocks_info用データ作成
    # ========================================================

    def _build_info(
        self,
        box_data_list,
        tier_by_id,
    ):

        info_lines = []


        for box_data in box_data_list:

            tier = tier_by_id[
                id(box_data)
            ]


            info_lines.append(
                f"distance="
                f"{box_data['distance']:.3f},"
                f"tier="
                f"{tier},"
                f"model="
                f"{box_data['model_name']},"
                f"height="
                f"{box_data['pixel_height']},"
                f"center_x="
                f"{box_data['center_x']}"
            )


        return info_lines


    # ========================================================
    # brocks_info送信
    # ========================================================

    def _publish_info(self, info_lines):

        if not info_lines:
            return


        info_msg = String()

        info_msg.data = "\n".join(
            info_lines
        )


        self.info_publisher.publish(
            info_msg
        )


    # ========================================================
    # Image送信
    #
    # cv_bridge不要
    # ========================================================

    def _publish_image(self, frame):

        # ----------------------------------------------------
        # ROS Imageメッセージ
        # ----------------------------------------------------

        msg = Image()


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )


        # ----------------------------------------------------
        # サイズ
        # ----------------------------------------------------

        height, width, channels = frame.shape

        msg.height = height
        msg.width = width


        # ----------------------------------------------------
        # BGR
        # ----------------------------------------------------

        msg.encoding = "bgr8"


        # ----------------------------------------------------
        # エンディアン
        # ----------------------------------------------------

        msg.is_bigendian = 0


        # ----------------------------------------------------
        # 1行のバイト数
        # ----------------------------------------------------

        msg.step = width * channels


        # ----------------------------------------------------
        # ndarray → bytes
        # ----------------------------------------------------

        msg.data = frame.tobytes()


        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.publisher.publish(msg)


    # ========================================================
    # 終了処理
    # ========================================================

    def destroy_node(self):

        self.running = False

        self.webcam.release()

        super().destroy_node()


# ============================================================
# main
# ============================================================

def main(args=None):

    # --------------------------------------------------------
    # ROS 2初期化
    # --------------------------------------------------------

    rclpy.init(args=args)


    # --------------------------------------------------------
    # ノード作成
    # --------------------------------------------------------

    node = BrockMasterNode()


    # --------------------------------------------------------
    # MultiThreadedExecutor
    # --------------------------------------------------------

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)


    try:

        # ----------------------------------------------------
        # ROS 2処理開始
        # ----------------------------------------------------

        executor.spin()


    except KeyboardInterrupt:

        pass


    finally:

        # ----------------------------------------------------
        # ノード終了
        # ----------------------------------------------------

        node.destroy_node()


        # ----------------------------------------------------
        # ROS 2終了
        # ----------------------------------------------------

        rclpy.shutdown()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()