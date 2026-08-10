import os
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
)

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32

from cv_bridge import CvBridge
from ultralytics import YOLO

from .webcam_utils import LatestFrameReader


# =========================================================
# トピック名
# =========================================================

IMAGE_TOPIC = "brock_Webcam_BBOX"
INFO_TOPIC = "brocks_info"
COLOR_TOPIC = "brock_color"
CONF_TOPIC = "brock_conf"


# =========================================================
# YOLOモデル設定
# =========================================================

MODEL_FILENAMES = [
    "red_brock.pt",
    "blue_brock.pt",
]

if len(MODEL_FILENAMES) < 1 or len(MODEL_FILENAMES) > 2:
    raise ValueError(
        "MODEL_FILENAMES は1つまたは2つ設定してください。"
    )


MODEL_PATHS = [
    os.path.join(
        os.path.dirname(__file__),
        filename
    )
    for filename in MODEL_FILENAMES
]


MODEL_NAMES = [
    os.path.splitext(
        os.path.basename(path)
    )[0]
    for path in MODEL_PATHS
]


# =========================================================
# モデル名から色を取得
#
# red_brock.pt  -> red
# blue_brock.pt -> blue
# =========================================================

COLOR_TO_MODEL_INDEX = {
    name.split("_")[0].lower(): idx
    for idx, name in enumerate(MODEL_NAMES)
}


# =========================================================
# YOLO confidence 初期値
# =========================================================

CONF_THRES = 0.2


# =========================================================
# Webカメラ設定
# =========================================================

WEBCAM_INDEX = 4
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720


# =========================================================
# 距離推定設定
# =========================================================

BOX_REAL_WIDTH_M = 0.22
BOX_REAL_HEIGHT_M = 0.30

WEBCAM_FOCAL_LENGTH_PX = 791.7


# =========================================================
# アスペクト比
# =========================================================

EXPECTED_ASPECT_RATIO = (
    BOX_REAL_WIDTH_M / BOX_REAL_HEIGHT_M
)

ASPECT_RATIO_TOLERANCE = 0.4


# =========================================================
# 距離履歴
# =========================================================

DISTANCE_HISTORY_SIZE = 4


# =========================================================
# 描画色 BGR
# =========================================================

BOX_COLOR = (0, 165, 255)


# =========================================================
# 距離推定
# =========================================================

def estimate_distance_from_height(pixel_height):

    if (
        pixel_height <= 0
        or WEBCAM_FOCAL_LENGTH_PX <= 0
    ):
        return 0.0

    return (
        BOX_REAL_HEIGHT_M
        * WEBCAM_FOCAL_LENGTH_PX
    ) / pixel_height


# =========================================================
# アスペクト比の信頼性判定
# =========================================================

def is_aspect_ratio_reliable(
    pixel_width,
    pixel_height
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


# =========================================================
# 段数を割り振る
# =========================================================

def assign_tiers(box_data_list):

    sorted_by_bottom = sorted(
        box_data_list,
        key=lambda b: b["y2"],
        reverse=True
    )

    return {
        id(b): tier
        for tier, b in enumerate(
            sorted_by_bottom,
            start=1
        )
    }


# =========================================================
# メインノード
# =========================================================

class BrockMasterNode(Node):

    def __init__(self):

        super().__init__("brock_master")

        # =====================================================
        # Confidence
        # =====================================================

        self.conf_thres = CONF_THRES

        # =====================================================
        # モデル
        # =====================================================

        self._load_models()

        # =====================================================
        # Webカメラ
        # =====================================================

        self._init_webcam()

        # =====================================================
        # Publisher / Subscriber
        # =====================================================

        self._init_publishers_and_subscribers()

        self.get_logger().info(
            "Starting inference..."
        )

        self.get_logger().info(
            f"Initial YOLO confidence: "
            f"{self.conf_thres:.2f}"
        )

    # =========================================================
    # モデル読み込み
    # =========================================================

    def _load_models(self):

        self.get_logger().info(
            f"Loading {len(MODEL_PATHS)} model(s)..."
        )

        self.models = [
            YOLO(path)
            for path in MODEL_PATHS
        ]

        for name in MODEL_NAMES:

            self.get_logger().info(
                f"loaded model: {name}"
            )

        # =====================================================
        # 初期モデル
        # =====================================================

        self.active_model_index = 0

        self.get_logger().info(
            "Default active model: "
            f"{MODEL_NAMES[self.active_model_index]}"
        )

    # =========================================================
    # Webカメラ初期化
    # =========================================================

    def _init_webcam(self):

        self.webcam = LatestFrameReader(
            WEBCAM_INDEX,
            cv2.CAP_V4L2,
            WEBCAM_WIDTH,
            WEBCAM_HEIGHT
        )

        self.bridge = CvBridge()

        self.distance_history = deque(
            maxlen=DISTANCE_HISTORY_SIZE
        )

    # =========================================================
    # Publisher / Subscriber
    # =========================================================

    def _init_publishers_and_subscribers(self):

        # =====================================================
        # Callback Group
        # =====================================================

        self.timer_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.color_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.conf_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        # =====================================================
        # Image Publisher
        # =====================================================

        self.publisher = self.create_publisher(
            Image,
            IMAGE_TOPIC,
            10
        )

        # =====================================================
        # Info Publisher
        # =====================================================

        self.info_publisher = self.create_publisher(
            String,
            INFO_TOPIC,
            10
        )

        # =====================================================
        # brock_color Subscriber
        # =====================================================

        self.color_subscription = (
            self.create_subscription(
                String,
                COLOR_TOPIC,
                self.on_brock_color,
                10,
                callback_group=(
                    self.color_callback_group
                )
            )
        )

        self.get_logger().info(
            f"Subscribed to {COLOR_TOPIC}"
        )

        # =====================================================
        # brock_conf Subscriber
        # =====================================================

        self.conf_subscription = (
            self.create_subscription(
                Float32,
                CONF_TOPIC,
                self.on_brock_conf,
                10,
                callback_group=(
                    self.conf_callback_group
                )
            )
        )

        self.get_logger().info(
            f"Subscribed to {CONF_TOPIC}"
        )

        # =====================================================
        # 推論タイマー
        # =====================================================

        self.timer = self.create_timer(
            0.001,
            self.on_timer,
            callback_group=(
                self.timer_callback_group
            )
        )

        self.get_logger().info(
            f"Publishing image on {IMAGE_TOPIC}"
        )

        self.get_logger().info(
            f"Publishing detection info on {INFO_TOPIC}"
        )

    # =========================================================
    # brock_color
    # =========================================================

    def on_brock_color(self, msg):

        color = msg.data.strip().lower()

        # =====================================================
        # 色チェック
        # =====================================================

        if color not in COLOR_TO_MODEL_INDEX:

            self.get_logger().warning(
                f"Unknown color '{color}' received "
                f"on {COLOR_TOPIC}. "
                f"Valid colors: "
                f"{list(COLOR_TO_MODEL_INDEX.keys())}"
            )

            return

        # =====================================================
        # モデル変更
        # =====================================================

        new_index = (
            COLOR_TO_MODEL_INDEX[color]
        )

        if (
            new_index
            != self.active_model_index
        ):

            self.active_model_index = new_index

            self.get_logger().info(
                "Active model switched to: "
                f"{MODEL_NAMES[new_index]} "
                f"(color={color})"
            )

    # =========================================================
    # brock_conf
    # =========================================================

    def on_brock_conf(self, msg):

        # =====================================================
        # Float32を取得
        # =====================================================

        conf = float(msg.data)

        # =====================================================
        # 0.0～1.0に制限
        # =====================================================

        conf = max(
            0.0,
            min(1.0, conf)
        )

        # =====================================================
        # Confidence更新
        # =====================================================

        self.conf_thres = conf

        self.get_logger().info(
            "YOLO confidence threshold "
            f"changed to: {self.conf_thres:.2f}"
        )

    # =========================================================
    # 距離平滑化
    # =========================================================

    def get_smoothed_distance(
        self,
        pixel_height,
        reliable
    ):

        distance = (
            estimate_distance_from_height(
                pixel_height
            )
        )

        # =====================================================
        # 信頼できる距離だけ履歴へ追加
        # =====================================================

        if reliable and distance > 0:

            self.distance_history.append(
                distance
            )

        # =====================================================
        # 履歴がなければ現在値
        # =====================================================

        if not self.distance_history:

            return distance

        # =====================================================
        # 中央値
        # =====================================================

        sorted_history = sorted(
            self.distance_history
        )

        return sorted_history[
            len(sorted_history) // 2
        ]

    # =========================================================
    # 推論ループ
    # =========================================================

    def on_timer(self):

        ret, frame = (
            self.webcam.read()
        )

        if not ret:

            return

        # =====================================================
        # YOLO推論
        # =====================================================

        box_data_list = (
            self._run_inference(frame)
        )

        # =====================================================
        # 段数計算
        # =====================================================

        tier_by_id = assign_tiers(
            box_data_list
        )

        # =====================================================
        # 描画 + 情報作成
        # =====================================================

        info_lines = (
            self._draw_boxes_and_build_info(
                frame,
                box_data_list,
                tier_by_id
            )
        )

        # =====================================================
        # 情報送信
        # =====================================================

        self._publish_info(
            info_lines
        )

        # =====================================================
        # 画像送信
        # =====================================================

        self._publish_image(
            frame
        )

    # =========================================================
    # YOLO推論
    # =========================================================

    def _run_inference(self, frame):

        # =====================================================
        # 現在選択されているモデル
        # =====================================================

        model_index = (
            self.active_model_index
        )

        model = (
            self.models[model_index]
        )

        model_name = (
            MODEL_NAMES[model_index]
        )

        # =====================================================
        # 現在のConfidenceを使用
        # =====================================================

        results = model(
            frame,
            conf=self.conf_thres
        )

        box_data_list = []

        # =====================================================
        # 検出結果
        # =====================================================

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                pixel_width = (
                    x2 - x1
                )

                pixel_height = (
                    y2 - y1
                )

                # =================================================
                # YOLO Confidence
                # =================================================

                confidence = float(
                    box.conf[0].item()
                )

                # =================================================
                # アスペクト比
                # =================================================

                reliable = (
                    is_aspect_ratio_reliable(
                        pixel_width,
                        pixel_height
                    )
                )

                # =================================================
                # 距離
                # =================================================

                distance = (
                    self.get_smoothed_distance(
                        pixel_height,
                        reliable
                    )
                )

                # =================================================
                # 検出情報
                # =================================================

                box_data_list.append({

                    "x1": x1,
                    "y1": y1,

                    "x2": x2,
                    "y2": y2,

                    "pixel_height":
                        pixel_height,

                    "distance":
                        distance,

                    "center_x":
                        (x1 + x2) // 2,

                    "confidence":
                        confidence,

                    "reliable":
                        reliable,

                    "model_name":
                        model_name,
                })

        return box_data_list

    # =========================================================
    # 描画 + brocks_info作成
    # =========================================================

    def _draw_boxes_and_build_info(
        self,
        frame,
        box_data_list,
        tier_by_id
    ):

        info_lines = []

        for b in box_data_list:

            x1 = b["x1"]
            y1 = b["y1"]
            x2 = b["x2"]
            y2 = b["y2"]

            tier = (
                tier_by_id[id(b)]
            )

            # =================================================
            # 枠
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                2
            )

            # =================================================
            # 中心線
            # =================================================

            cv2.line(
                frame,
                (b["center_x"], y1),
                (b["center_x"], y2),
                BOX_COLOR,
                2
            )

            # =================================================
            # 表示文字
            # =================================================

            label_text = (

                f"距離:{b['distance']:.3f}m "

                f"段:{tier} "

                f"モデル:{b['model_name']} "

                f"高さ:{b['pixel_height']}px "

                f"信頼度:{b['confidence']:.2f}"
            )

            # =================================================
            # 傾き表示
            # =================================================

            if not b["reliable"]:

                label_text += " [傾き大]"

            # =================================================
            # 描画
            # =================================================

            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                BOX_COLOR,
                2
            )

            # =================================================
            # brocks_info
            # =================================================

            info_lines.append(

                f"distance="
                f"{b['distance']:.3f},"

                f"tier="
                f"{tier},"

                f"model="
                f"{b['model_name']},"

                f"height="
                f"{b['pixel_height']},"

                f"center_x="
                f"{b['center_x']}"
            )

        return info_lines

    # =========================================================
    # brocks_info送信
    # =========================================================

    def _publish_info(
        self,
        info_lines
    ):

        if not info_lines:

            return

        info_msg = String()

        info_msg.data = (
            "\n".join(info_lines)
        )

        self.info_publisher.publish(
            info_msg
        )

    # =========================================================
    # 画像送信
    # =========================================================

    def _publish_image(
        self,
        frame
    ):

        msg = (
            self.bridge.cv2_to_imgmsg(
                frame,
                encoding="bgr8"
            )
        )

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        self.publisher.publish(
            msg
        )

    # =========================================================
    # 終了処理
    # =========================================================

    def destroy_node(self):

        self.webcam.release()

        super().destroy_node()


# =========================================================
# main
# =========================================================

def main(args=None):

    rclpy.init(args=args)

    node = BrockMasterNode()

    executor = MultiThreadedExecutor(
        num_threads=3
    )

    executor.add_node(node)

    try:

        executor.spin()

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()