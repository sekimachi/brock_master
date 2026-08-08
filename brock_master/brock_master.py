import os
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

from .webcam_utils import LatestFrameReader


# =========================================================
# トピック名
# =========================================================

IMAGE_TOPIC = "brock_Webcam_BBOX"   # 検出結果を描画した映像を配信するトピック
INFO_TOPIC = "brocks_info"          # 検出結果(距離・段数など)を配信するトピック
COLOR_TOPIC = "brock_color"         # 使用するモデルの色("red" / "blue")を受け取るトピック


# =========================================================
# YOLOモデル設定(最大2つまで設定可能)
# =========================================================

# パッケージ内(このファイルと同じディレクトリ)に置いたモデルファイル名をここに列挙する。
# ファイル名の先頭(アンダースコア区切りの最初の単語)が brock_color トピックで
# 受け取る色名("red" / "blue")と対応する。
MODEL_FILENAMES = [
    "red_brock.pt",
    "blue_brock.pt",
]

if len(MODEL_FILENAMES) < 1 or len(MODEL_FILENAMES) > 2:
    raise ValueError("MODEL_FILENAMES は1つまたは2つ設定してください。")

MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), filename)
    for filename in MODEL_FILENAMES
]

MODEL_NAMES = [
    os.path.splitext(os.path.basename(path))[0]
    for path in MODEL_PATHS
]

# ファイル名の先頭の単語("red_brock" -> "red")を色キーとして、
# brock_color トピックで受け取った文字列からモデルのインデックスを引けるようにする
COLOR_TO_MODEL_INDEX = {
    name.split("_")[0].lower(): idx
    for idx, name in enumerate(MODEL_NAMES)
}

CONF_THRES = 0.2


# =========================================================
# ウェブカメラ設定
# =========================================================

WEBCAM_INDEX = 4
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720


# =========================================================
# 距離推定設定
# =========================================================

# calibrate_webcam.py で計測した値をここに設定する
# 2026-07-10: gosa.txtの実測(正しい距離 vs 表示距離)から最小二乗フィットで補正
#   1回目: 27点(3.5〜6.5m)から 760.6 -> 791.7
#   2回目: 791.7適用後に再計測した21点(4.0〜6.0m)を合算した48点で再フィット -> 788.6だったが、
#          誤差がむしろ拡大し始めたため791.7に戻す
BOX_REAL_WIDTH_M = 0.22
BOX_REAL_HEIGHT_M = 0.30
WEBCAM_FOCAL_LENGTH_PX = 791.7

# 2026-07-10: 箱を左右に傾けると、奥行き(側面)が見える分バウンディングボックスの
# 幅がむしろ広がり「近すぎ」に誤推定することが判明。左右の傾き(ヨー)は高さには
# ほぼ影響しないため、距離推定は幅ではなく高さから行う。
# 正対時の幅/高さ比から外れたフレームは傾き過大とみなし、距離の更新をスキップする。
EXPECTED_ASPECT_RATIO = BOX_REAL_WIDTH_M / BOX_REAL_HEIGHT_M
ASPECT_RATIO_TOLERANCE = 0.4  # 正対時の比から±40%を超えたら信頼しない

DISTANCE_HISTORY_SIZE = 4

# 描画色 (BGR)
BOX_COLOR = (0, 165, 255)


# =========================================================
# 距離推定ヘルパー関数
# =========================================================

def estimate_distance_from_height(pixel_height):
    if pixel_height <= 0 or WEBCAM_FOCAL_LENGTH_PX <= 0:
        return 0.0

    return (BOX_REAL_HEIGHT_M * WEBCAM_FOCAL_LENGTH_PX) / pixel_height


def is_aspect_ratio_reliable(pixel_width, pixel_height):
    if pixel_width <= 0 or pixel_height <= 0:
        return False

    observed_ratio = pixel_width / pixel_height
    deviation = abs(observed_ratio / EXPECTED_ASPECT_RATIO - 1.0)
    return deviation <= ASPECT_RATIO_TOLERANCE


def assign_tiers(box_data_list):
    """段数を割り振る: 画面下(y2が大きい)に近いものほど1、上にあるものほど大きい数字にする"""

    sorted_by_bottom = sorted(
        box_data_list,
        key=lambda b: b["y2"],
        reverse=True
    )

    return {
        id(b): tier
        for tier, b in enumerate(sorted_by_bottom, start=1)
    }


# =========================================================
# メインノード
# =========================================================

class BrockMasterNode(Node):

    def __init__(self):
        super().__init__("brock_master")

        self._load_models()
        self._init_webcam()
        self._init_publishers_and_subscribers()

        self.get_logger().info("Starting inference...")

    # ---------------------------------------------------
    # 初期化まわり
    # ---------------------------------------------------

    def _load_models(self):
        self.get_logger().info(f"Loading {len(MODEL_PATHS)} model(s)...")

        self.models = [YOLO(path) for path in MODEL_PATHS]

        for name in MODEL_NAMES:
            self.get_logger().info(f"  loaded model: {name}")

        # brock_color トピックで指定された色に対応するモデルだけを使う。
        # まだ何も受信していない間はリストの1つ目をデフォルトで使う。
        self.active_model_index = 0
        self.get_logger().info(
            f"Default active model: {MODEL_NAMES[self.active_model_index]}"
        )

    def _init_webcam(self):
        self.webcam = LatestFrameReader(
            WEBCAM_INDEX,
            cv2.CAP_V4L2,
            WEBCAM_WIDTH,
            WEBCAM_HEIGHT
        )
        self.bridge = CvBridge()
        self.distance_history = deque(maxlen=DISTANCE_HISTORY_SIZE)

    def _init_publishers_and_subscribers(self):
        # タイマー(推論ループ)と色トピックの購読を別々のコールバックグループに
        # 分ける。同じグループ(特にReentrant)に入れると、1msごとに発火する
        # タイマーがExecutorのスレッドを埋め尽くし、brock_colorの受信が
        # いつまでも処理されない、という事態になり得るため。
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        color_callback_group = MutuallyExclusiveCallbackGroup()

        self.publisher = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self.info_publisher = self.create_publisher(String, INFO_TOPIC, 10)

        self.color_subscription = self.create_subscription(
            String,
            COLOR_TOPIC,
            self.on_brock_color,
            10,
            callback_group=color_callback_group
        )
        self.get_logger().info(f"Subscribed to color topic: {COLOR_TOPIC}")

        self.timer = self.create_timer(
            0.001,  # カメラ読み取り速度に合わせてなるべく高頻度でループを回す
            self.on_timer,
            callback_group=timer_callback_group
        )
        self.get_logger().info(f"Publishing annotated image on topic: {IMAGE_TOPIC}")
        self.get_logger().info(f"Publishing detection info on topic: {INFO_TOPIC}")

    # ---------------------------------------------------
    # brock_color トピックのコールバック
    # ---------------------------------------------------

    def on_brock_color(self, msg):
        color = msg.data.strip().lower()

        if color not in COLOR_TO_MODEL_INDEX:
            self.get_logger().warn(
                f"Unknown color '{color}' received on {COLOR_TOPIC}. "
                f"Valid colors: {list(COLOR_TO_MODEL_INDEX.keys())}"
            )
            return

        new_index = COLOR_TO_MODEL_INDEX[color]

        if new_index != self.active_model_index:
            self.active_model_index = new_index
            self.get_logger().info(
                f"Active model switched to: {MODEL_NAMES[new_index]} (color={color})"
            )

    # ---------------------------------------------------
    # 距離の平滑化
    # ---------------------------------------------------

    def get_smoothed_distance(self, pixel_height, reliable):
        distance = estimate_distance_from_height(pixel_height)

        if reliable and distance > 0:
            self.distance_history.append(distance)

        if not self.distance_history:
            return distance

        sorted_history = sorted(self.distance_history)
        return sorted_history[len(sorted_history) // 2]

    # ---------------------------------------------------
    # 推論ループ本体
    # ---------------------------------------------------

    def on_timer(self):
        ret, frame = self.webcam.read()

        if not ret:
            return

        box_data_list = self._run_inference(frame)
        tier_by_id = assign_tiers(box_data_list)
        info_lines = self._draw_boxes_and_build_info(frame, box_data_list, tier_by_id)

        self._publish_info(info_lines)
        self._publish_image(frame)

    def _run_inference(self, frame):
        """brock_color トピックで選択された1つのモデルだけで推論する"""

        model_index = self.active_model_index
        model = self.models[model_index]
        model_name = MODEL_NAMES[model_index]

        results = model(frame, conf=CONF_THRES)

        box_data_list = []

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                pixel_width = x2 - x1
                pixel_height = y2 - y1

                reliable = is_aspect_ratio_reliable(pixel_width, pixel_height)
                distance = self.get_smoothed_distance(pixel_height, reliable)

                box_data_list.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "pixel_height": pixel_height,
                    "distance": distance,
                    "center_x": (x1 + x2) // 2,
                    "reliable": reliable,
                    "model_name": model_name,
                })

        return box_data_list

    def _draw_boxes_and_build_info(self, frame, box_data_list, tier_by_id):
        info_lines = []

        for b in box_data_list:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            tier = tier_by_id[id(b)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.line(frame, (b["center_x"], y1), (b["center_x"], y2), BOX_COLOR, 2)

            label_text = (
                f"距離:{b['distance']:.3f}m "
                f"段:{tier} "
                f"モデル:{b['model_name']} "
                f"高さ:{b['pixel_height']}px"
            )

            if not b["reliable"]:
                label_text += " [傾き大]"

            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                BOX_COLOR,
                2
            )

            info_lines.append(
                f"distance={b['distance']:.3f},"
                f"tier={tier},"
                f"model={b['model_name']},"
                f"height={b['pixel_height']},"
                f"center_x={b['center_x']}"
            )

        return info_lines

    def _publish_info(self, info_lines):
        if not info_lines:
            return

        info_msg = String()
        info_msg.data = "\n".join(info_lines)
        self.info_publisher.publish(info_msg)

    def _publish_image(self, frame):
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

    # ---------------------------------------------------
    # 終了処理
    # ---------------------------------------------------

    def destroy_node(self):
        self.webcam.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = BrockMasterNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()