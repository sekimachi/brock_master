import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Bool

from ultralytics import YOLO

from .webcam_utils import LatestFrameReader


# ============================================================
# ROSトピック
# ============================================================

IMAGE_TOPIC = "brock_Webcam_BBOX"
INFO_TOPIC = "brocks_info"
COLOR_TOPIC = "brock_color"
CONF_TOPIC = "brock_conf"

# ★YOLO / HSV切り替えトピック
YOLO_TOPIC = "brock_YOLO"


# ============================================================
# YOLOモデル設定
# ============================================================

MODEL_FILENAMES = [
    "red_brock.pt",
    "blue_brock.pt",
]

CONF_THRES = 0.3


# ============================================================
# Webカメラ設定（推論用の解像度）
# ============================================================

# /dev/videoNの番号はUSB抜き差しや起動順序でずれることがあるため、
# by-id(デバイス名ベース)のパスで固定して指定する。
# 番号を使う場合は `v4l2-ctl --list-devices` で都度確認すること。
WEBCAM_INDEX = "/dev/video0"
WEBCAM_WIDTH = 1920
WEBCAM_HEIGHT = 1080


# ============================================================
# 配信用解像度
# ============================================================

PUBLISH_WIDTH = 320
PUBLISH_HEIGHT = 180


# ============================================================
# 映像配信FPS
# ============================================================

IMAGE_FPS = 30.0


# ============================================================
# YOLO推論FPS
# ============================================================

YOLO_FPS = 10.0


# ============================================================
# 距離推定設定
# ============================================================

BOX_REAL_WIDTH_M = 0.22
BOX_REAL_HEIGHT_M = 0.30

WEBCAM_FOCAL_LENGTH_PX = 1161.12  # 実機YOLO 2m/3m/4m/5mデータで最小二乗フィット

# ------------------------------------------------------------
# カメラ距離オフセット
#
# キャリブレーション時、メジャーの0点(測定基準位置)と
# カメラのレンズ(光学中心)の位置が一致していないことで生じる
# 系統誤差を補正するための値。
#
# distance = (H * focal / pixel_height) - CAMERA_DISTANCE_OFFSET_M
#
# 実機のYOLO検出ログ(2m, 3m, 4m, 5m地点、箱静止確認済み)から
# 最小二乗フィットして算出。RMSE = 約0.10m (2.85%)
# ------------------------------------------------------------

CAMERA_DISTANCE_OFFSET_M = -0.2336

DISTANCE_HISTORY_SIZE = 4


# ============================================================
# アスペクト比設定
# ============================================================

EXPECTED_ASPECT_RATIO = (
    BOX_REAL_WIDTH_M / BOX_REAL_HEIGHT_M
)

ASPECT_RATIO_TOLERANCE = 0.4


# ============================================================
# BBOX重複判定
#
# 重なり面積 / 小さい方のBBOX面積
#
# 70%以上重なった場合、
# 信頼度の低い方を削除する
# ============================================================

BBOX_OVERLAP_THRESHOLD = 0.90


# ============================================================
# ★HSV認識設定
#
# ★YOLO_inference = True  : YOLOで認識
# ★YOLO_inference = False : HSVで認識
# ============================================================

YOLO_inference = True

# ★赤色のHSV範囲
HSV_RED_LOW_1 = np.array([0, 80, 80], dtype=np.uint8)
HSV_RED_HIGH_1 = np.array([10, 255, 255], dtype=np.uint8)
HSV_RED_LOW_2 = np.array([170, 80, 80], dtype=np.uint8)
HSV_RED_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)

# ★青色のHSV範囲
HSV_BLUE_LOW = np.array([90, 80, 50], dtype=np.uint8)
HSV_BLUE_HIGH = np.array([140, 255, 255], dtype=np.uint8)

# ★白色判定
HSV_WHITE_S_MAX = 50
HSV_WHITE_V_MIN = 150

# ★BBOX内に占める白色画素の最低割合
HSV_WHITE_RATIO_THRESHOLD = 0.40

# ★HSVマスクのノイズ除去設定
HSV_MORPH_KERNEL_SIZE = 5
HSV_MIN_CONTOUR_AREA = 500


# ============================================================
# ★YOLO BBOX内部の反対色判定
# ============================================================

# ★YOLOで検出したBBOX内部に、
# ★反対色がこの割合以上含まれていたらBBOXを除外する。
#
# ★赤モデルの場合
# ★    青色が30%以上 → 除外
#
# ★青モデルの場合
# ★    赤色が30%以上 → 除外

YOLO_OPPOSITE_COLOR_RATIO_THRESHOLD = 0.30


# ============================================================
# FPS計測設定
# ============================================================

FPS_HISTORY_SIZE = 10


# ============================================================
# デバッグ計測設定
#
# on_camera_timer内の各ステップ処理時間を
# N回に1回ログ出力する
# ============================================================

DEBUG_TIMING_ENABLED = True
DEBUG_TIMING_INTERVAL = 30


# ============================================================
# 画像publish切り替え設定
#
# publish_call(DDS送信)のコストが原因かどうかを切り分けるため、
# 画像のpublish自体をON/OFFできるようにする。
# Falseにすると、画像は配信されずrqt_image_view等では見れなく
# なるが、brocks_info(距離等のテキスト情報)は影響を受けない。
# ============================================================

ENABLE_IMAGE_PUBLISH = True


# ============================================================
# 描画設定
# ============================================================

# BBOX
BOX_COLOR = (0, 165, 255)

# 目標X軸
TARGET_X_COLOR = (0, 255, 0)
TARGET_X_THICKNESS = 2

# FPS表示
FPS_COLOR = (255, 255, 255)
FPS_BACKGROUND_COLOR = (0, 0, 0)


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
# FPS計測クラス
# ============================================================

class FPSCounter:

    def __init__(self, history_size=10):

        self.timestamps = deque(
            maxlen=history_size
        )

        self.fps = 0.0

        self.lock = threading.Lock()


    def update(self):

        now = time.perf_counter()

        with self.lock:

            self.timestamps.append(now)

            if len(self.timestamps) < 2:
                self.fps = 0.0
                return self.fps

            elapsed = (
                self.timestamps[-1]
                - self.timestamps[0]
            )

            if elapsed > 0:

                self.fps = (
                    (len(self.timestamps) - 1)
                    / elapsed
                )

        return self.fps


    def get_fps(self):

        with self.lock:
            return self.fps


# ============================================================
# 距離推定
#
# メジャーの0点とレンズ位置のズレを補正するため、
# CAMERA_DISTANCE_OFFSET_M を減算する。
# ============================================================

def estimate_distance_from_height(pixel_height):

    if (
        pixel_height <= 0
        or WEBCAM_FOCAL_LENGTH_PX <= 0
    ):
        return 0.0

    distance = (
        (
            BOX_REAL_HEIGHT_M
            * WEBCAM_FOCAL_LENGTH_PX
        ) / pixel_height
    ) - CAMERA_DISTANCE_OFFSET_M

    return max(distance, 0.0)


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
# BBOXの重なり率計算
#
# 「重なっている面積 ÷ 小さい方のBBOX面積」
# ============================================================

def calculate_overlap_ratio(box1, box2):

    # ----------------------------------------------------
    # 交差領域の座標
    # ----------------------------------------------------

    overlap_x1 = max(
        box1["x1"],
        box2["x1"],
    )

    overlap_y1 = max(
        box1["y1"],
        box2["y1"],
    )

    overlap_x2 = min(
        box1["x2"],
        box2["x2"],
    )

    overlap_y2 = min(
        box1["y2"],
        box2["y2"],
    )


    # ----------------------------------------------------
    # 重なっていない場合
    # ----------------------------------------------------

    if (
        overlap_x2 <= overlap_x1
        or overlap_y2 <= overlap_y1
    ):
        return 0.0


    # ----------------------------------------------------
    # 重なり領域
    # ----------------------------------------------------

    overlap_width = (
        overlap_x2 - overlap_x1
    )

    overlap_height = (
        overlap_y2 - overlap_y1
    )

    overlap_area = (
        overlap_width
        * overlap_height
    )


    # ----------------------------------------------------
    # BBOX1の面積
    # ----------------------------------------------------

    width1 = (
        box1["x2"] - box1["x1"]
    )

    height1 = (
        box1["y2"] - box1["y1"]
    )

    area1 = width1 * height1


    # ----------------------------------------------------
    # BBOX2の面積
    # ----------------------------------------------------

    width2 = (
        box2["x2"] - box2["x1"]
    )

    height2 = (
        box2["y2"] - box2["y1"]
    )

    area2 = width2 * height2


    # ----------------------------------------------------
    # 小さい方の面積
    # ----------------------------------------------------

    smaller_area = min(
        area1,
        area2,
    )

    if smaller_area <= 0:
        return 0.0


    # ----------------------------------------------------
    # 重なり率
    # ----------------------------------------------------

    return (
        overlap_area
        / smaller_area
    )


# ============================================================
# BBOX重複除去
#
# 70%以上重なっているBBOXがあれば、
# 信頼度の低い方を削除する。
# ============================================================

def remove_overlapping_boxes(box_data_list):

    if len(box_data_list) <= 1:
        return box_data_list


    # ----------------------------------------------------
    # Confidenceの高い順
    # ----------------------------------------------------

    sorted_boxes = sorted(
        box_data_list,
        key=lambda box: box["confidence"],
        reverse=True,
    )


    kept_boxes = []


    # ----------------------------------------------------
    # 高信頼度のBBOXから確認
    # ----------------------------------------------------

    for current_box in sorted_boxes:

        should_remove = False


        # ------------------------------------------------
        # 既に残っているBBOXと比較
        # ------------------------------------------------

        for kept_box in kept_boxes:

            overlap_ratio = (
                calculate_overlap_ratio(
                    current_box,
                    kept_box,
                )
            )


            # --------------------------------------------
            # 70%以上重なっている
            # --------------------------------------------

            if (
                overlap_ratio
                >= BBOX_OVERLAP_THRESHOLD
            ):

                should_remove = True

                break


        # ------------------------------------------------
        # 重複していなければ残す
        # ------------------------------------------------

        if not should_remove:

            kept_boxes.append(
                current_box
            )


    return kept_boxes


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
        # FPS計測
        # ====================================================

        self.image_fps_counter = FPSCounter(
            FPS_HISTORY_SIZE
        )

        self.yolo_fps_counter = FPSCounter(
            FPS_HISTORY_SIZE
        )


        # ====================================================
        # デバッグ計測用カウンタ
        # ====================================================

        self._camera_dbg_count = 0


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

        self.get_logger().info(
            f"Inference resolution: "
            f"{WEBCAM_WIDTH}x{WEBCAM_HEIGHT}"
        )

        self.get_logger().info(
            f"Publish resolution: "
            f"{PUBLISH_WIDTH}x{PUBLISH_HEIGHT}"
        )

        self.get_logger().info(
            f"BBOX overlap threshold: "
            f"{BBOX_OVERLAP_THRESHOLD * 100:.0f}%"
        )

        self.get_logger().info(
            f"Camera distance offset: "
            f"{CAMERA_DISTANCE_OFFSET_M:.3f} m"
        )

        self.get_logger().info(
            f"Image publish enabled: "
            f"{ENABLE_IMAGE_PUBLISH}"
        )

        # ★現在の認識方式をログ表示
        self.get_logger().info(
            f"Initial recognition mode: "
            f"{'YOLO' if YOLO_inference else 'HSV'}"
        )

        # ★反対色判定の閾値をログ表示
        self.get_logger().info(
            f"YOLO opposite color threshold: "
            f"{YOLO_OPPOSITE_COLOR_RATIO_THRESHOLD * 100:.0f}%"
        )


    # ========================================================
    # モデル読み込み
    # ========================================================

    def _load_models(self):

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


        self.active_model_index = 0

        self.get_logger().info(
            "Default active model: "
            f"{MODEL_NAMES[self.active_model_index]}"
        )


    # ========================================================
    # Webカメラ初期化
    # ========================================================

    def _init_webcam(self):

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

        # ----------------------------------------------------
        # Callback Group
        # ----------------------------------------------------

        self.color_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.conf_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        # ★YOLO切り替え用Callback Group
        self.yolo_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.camera_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )

        self.inference_callback_group = (
            MutuallyExclusiveCallbackGroup()
        )


        # ----------------------------------------------------
        # Image Publisher
        #
        # rqt_image_view等のGUI表示側の描画が追いつかないと、
        # RELIABLE QoS(デフォルト)ではACK待ちでpublish()自体が
        # ブロックされFPS低下の原因になるため、
        # センサーデータ向けのBEST_EFFORT QoSを使用する。
        # 多少のフレーム取りこぼしより最新性を優先する。
        # ----------------------------------------------------

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            Image,
            IMAGE_TOPIC,
            image_qos,
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


        # ----------------------------------------------------
        # ★brock_YOLO Subscriber
        #
        # ★True  -> YOLO推論
        # ★False -> HSV認識
        #
        # ★受信した瞬間に認識方式を変更する。
        # ----------------------------------------------------

        self.yolo_subscription = (
            self.create_subscription(
                Bool,
                YOLO_TOPIC,
                self.on_brock_yolo,
                10,
                callback_group=(
                    self.yolo_callback_group
                ),
            )
        )

        self.get_logger().info(
            f"Subscribed to {YOLO_TOPIC}"
        )


        # ====================================================
        # カメラ取得タイマー
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
        # ====================================================

        self.yolo_timer = self.create_timer(
            1.0 / YOLO_FPS,
            self.on_yolo_timer,
            callback_group=(
                self.inference_callback_group
            ),
        )


        self.get_logger().info(
            f"Publishing image on {IMAGE_TOPIC}"
        )

        self.get_logger().info(
            f"Publishing detection info on {INFO_TOPIC}"
        )


    # ========================================================
    # ★brock_YOLO Callback
    # ========================================================

    def on_brock_yolo(self, msg):

        # ★Boolの値をそのまま使用
        new_mode = bool(msg.data)

        # ★認識方式をリアルタイムで変更
        if new_mode != YOLO_inference:

        
            YOLO_inference = new_mode

            # ★認識結果を一度リセット
            with self.data_lock:

                self.latest_box_data = []
                self.latest_info_lines = []

            # ★距離履歴もリセット
            self.distance_history.clear()

            # ★切り替えログ
            if YOLO_inference:

                self.get_logger().info(
                    "brock_YOLO=True "
                    "-> YOLO inference mode"
                )

            else:

                self.get_logger().info(
                    "brock_YOLO=False "
                    "-> HSV recognition mode"
                )


    # ========================================================
    # brock_color Callback
    # ========================================================

    def on_brock_color(self, msg):

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

        confidence = float(msg.data)


        # 0.0～1.0に制限
        confidence = max(
            0.0,
            min(1.0, confidence),
        )


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

        distance = estimate_distance_from_height(
            pixel_height
        )


        if reliable and distance > 0:

            self.distance_history.append(
                distance
            )


        if not self.distance_history:

            return distance


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
    # カメラ取得と映像配信
    # ========================================================

    def on_camera_timer(self):

        # ----------------------------------------------------
        # デバッグ計測開始
        # ----------------------------------------------------

        if DEBUG_TIMING_ENABLED:
            t0 = time.perf_counter()


        # ----------------------------------------------------
        # カメラ画像取得
        # ----------------------------------------------------

        ret, frame = self.webcam.read()

        if not ret:
            return


        if DEBUG_TIMING_ENABLED:
            t1 = time.perf_counter()


        # ----------------------------------------------------
        # 実際の映像FPSを更新
        # ----------------------------------------------------

        image_fps = (
            self.image_fps_counter.update()
        )


        # ----------------------------------------------------
        # 最新フレームを保存
        #
        # YOLOはこの最新フレームを使用
        # ----------------------------------------------------

        with self.data_lock:

            self.latest_frame = frame.copy()


        if DEBUG_TIMING_ENABLED:
            t2 = time.perf_counter()


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
        # BBOX描画
        # ----------------------------------------------------

        self._draw_boxes(
            frame,
            box_data_list,
            tier_by_id,
        )


        if DEBUG_TIMING_ENABLED:
            t3 = time.perf_counter()


        # ----------------------------------------------------
        # FPS表示
        #
        # 右上に表示
        # ----------------------------------------------------

        self._draw_fps(
            frame,
            image_fps,
            self.yolo_fps_counter.get_fps(),
        )


        if DEBUG_TIMING_ENABLED:
            t4 = time.perf_counter()


        # ----------------------------------------------------
        # 配信用に縮小
        # ----------------------------------------------------

        publish_frame = cv2.resize(
            frame,
            (
                PUBLISH_WIDTH,
                PUBLISH_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )


        if DEBUG_TIMING_ENABLED:
            t5 = time.perf_counter()


        # ----------------------------------------------------
        # Image publish
        #
        # ENABLE_IMAGE_PUBLISH = False にすると、publish_call
        # (DDS送信)のコストが原因かどうかを切り分けられる。
        # ----------------------------------------------------

        if ENABLE_IMAGE_PUBLISH:

            self._publish_image(
                publish_frame
            )


        # ----------------------------------------------------
        # デバッグ計測ログ出力
        #
        # 毎フレーム出すとログ自体が重くなるため、
        # DEBUG_TIMING_INTERVAL回に1回だけ出力する
        # ----------------------------------------------------

        if DEBUG_TIMING_ENABLED:

            t6 = time.perf_counter()

            self._camera_dbg_count += 1

            if (
                self._camera_dbg_count
                % DEBUG_TIMING_INTERVAL
                == 0
            ):

                self.get_logger().info(
                    "on_camera_timer timing: "
                    f"read={(t1 - t0) * 1000:.1f}ms "
                    f"copy={(t2 - t1) * 1000:.1f}ms "
                    f"draw_box={(t3 - t2) * 1000:.1f}ms "
                    f"draw_fps={(t4 - t3) * 1000:.1f}ms "
                    f"resize={(t5 - t4) * 1000:.1f}ms "
                    f"publish={(t6 - t5) * 1000:.1f}ms "
                    f"total={(t6 - t0) * 1000:.1f}ms "
                    f"image_fps={image_fps:.1f}"
                )


    # ========================================================
    # YOLOタイマー
    #
    # 映像配信とは別に動作
    # ========================================================

    def on_yolo_timer(self):

        # ----------------------------------------------------
        # 最新フレーム取得
        # ----------------------------------------------------

        with self.data_lock:

            if self.latest_frame is None:
                return

            frame = self.latest_frame.copy()


        # ----------------------------------------------------
        # 推論
        # ----------------------------------------------------

        start_time = time.perf_counter()

        # ★YOLO / HSV 切り替え
        if YOLO_inference:

            box_data_list = self._run_inference(
                frame
            )

        else:

            box_data_list = self._run_hsv_inference(
                frame
            )

        inference_time = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # 実際のYOLO FPSを更新
        # ----------------------------------------------------

        yolo_fps = (
            self.yolo_fps_counter.update()
        )


        # ====================================================
        # BBOX重複除去
        # ====================================================

        before_count = len(
            box_data_list
        )

        box_data_list = (
            remove_overlapping_boxes(
                box_data_list
            )
        )

        after_count = len(
            box_data_list
        )


        # ----------------------------------------------------
        # 重複除去ログ
        # ----------------------------------------------------

        if before_count != after_count:

            removed_count = (
                before_count
                - after_count
            )

            self.get_logger().debug(
                f"BBOX overlap filtering: "
                f"{before_count} -> "
                f"{after_count} "
                f"({removed_count} removed)"
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

            self.latest_box_data = (
                box_data_list
            )

            self.latest_info_lines = (
                info_lines
            )


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
            f"{inference_time * 1000:.1f} ms, "
            f"YOLO FPS: {yolo_fps:.1f}"
        )


    # ========================================================
    # FPS描画
    #
    # 画面右上に
    #
    # IMAGE FPS : xx.x
    # YOLO FPS  : xx.x
    #
    # を表示
    # ========================================================

    def _draw_fps(
        self,
        frame,
        image_fps,
        yolo_fps,
    ):

        # ----------------------------------------------------
        # 表示文字
        # ----------------------------------------------------

        image_fps_text = (
            f"IMAGE FPS : {image_fps:.1f}"
        )

        yolo_fps_text = (
            f"YOLO FPS  : {yolo_fps:.1f}"
        )


        # ----------------------------------------------------
        # フォント設定
        # ----------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.8

        thickness = 2


        # ----------------------------------------------------
        # 文字サイズ取得
        # ----------------------------------------------------

        (image_text_width, image_text_height), _ = (
            cv2.getTextSize(
                image_fps_text,
                font,
                font_scale,
                thickness,
            )
        )

        (yolo_text_width, yolo_text_height), _ = (
            cv2.getTextSize(
                yolo_fps_text,
                font,
                font_scale,
                thickness,
            )
        )


        # ----------------------------------------------------
        # 最大幅
        # ----------------------------------------------------

        text_width = max(
            image_text_width,
            yolo_text_width,
        )


        # ----------------------------------------------------
        # 右上の位置
        # ----------------------------------------------------

        margin_right = 20

        margin_top = 20

        x = (
            frame.shape[1]
            - text_width
            - margin_right
        )

        y1 = (
            margin_top
            + image_text_height
        )

        y2 = (
            y1
            + yolo_text_height
            + 15
        )


        # ----------------------------------------------------
        # 背景
        #
        # 文字を見やすくするため黒背景を描画
        # ----------------------------------------------------

        background_x1 = x - 10

        background_y1 = (
            margin_top - 10
        )

        background_x2 = (
            frame.shape[1]
            - margin_right
            + 10
        )

        background_y2 = (
            y2 + 10
        )


        cv2.rectangle(
            frame,
            (
                background_x1,
                background_y1,
            ),
            (
                background_x2,
                background_y2,
            ),
            FPS_BACKGROUND_COLOR,
            -1,
        )


        # ----------------------------------------------------
        # IMAGE FPS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            image_fps_text,
            (
                x,
                y1,
            ),
            font,
            font_scale,
            FPS_COLOR,
            thickness,
            cv2.LINE_AA,
        )


        # ----------------------------------------------------
        # YOLO FPS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            yolo_fps_text,
            (
                x,
                y2,
            ),
            font,
            font_scale,
            FPS_COLOR,
            thickness,
            cv2.LINE_AA,
        )


    # ========================================================
    # ★HSV推論
    # ========================================================

    def _run_hsv_inference(self, frame):

        # ★BGR -> HSV
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        # ★現在選択されている色を取得
        model_index = self.active_model_index
        model_name = MODEL_NAMES[model_index]
        color = model_name.split("_")[0].lower()

        # ★色条件からマスクを作成
        if color == "red":

            mask1 = cv2.inRange(
                hsv,
                HSV_RED_LOW_1,
                HSV_RED_HIGH_1,
            )

            mask2 = cv2.inRange(
                hsv,
                HSV_RED_LOW_2,
                HSV_RED_HIGH_2,
            )

            color_mask = cv2.bitwise_or(
                mask1,
                mask2,
            )

        elif color == "blue":

            color_mask = cv2.inRange(
                hsv,
                HSV_BLUE_LOW,
                HSV_BLUE_HIGH,
            )

        else:
            return []

        # ★色マスクのノイズ除去
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                HSV_MORPH_KERNEL_SIZE,
                HSV_MORPH_KERNEL_SIZE,
            ),
        )

        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # ★色領域の輪郭を取得
        contours, _ = cv2.findContours(
            color_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        box_data_list = []

        # ★白色マスクを作成
        white_mask = cv2.inRange(
            hsv,
            np.array(
                [0, 0, HSV_WHITE_V_MIN],
                dtype=np.uint8,
            ),
            np.array(
                [179, HSV_WHITE_S_MAX, 255],
                dtype=np.uint8,
            ),
        )

        for contour in contours:

            # ★小さい色領域は無視
            contour_area = cv2.contourArea(contour)

            if contour_area < HSV_MIN_CONTOUR_AREA:
                continue

            # ★色領域からBBOXを作成
            x, y, w, h = cv2.boundingRect(contour)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            pixel_width = x2 - x1
            pixel_height = y2 - y1

            if pixel_width <= 0 or pixel_height <= 0:
                continue

            # ★BBOX内部の白色割合を計算
            roi_white = white_mask[
                y1:y2,
                x1:x2,
            ]

            total_pixel_count = roi_white.size

            if total_pixel_count <= 0:
                continue

            white_pixel_count = cv2.countNonZero(
                roi_white
            )

            white_ratio = (
                white_pixel_count
                / total_pixel_count
            )

            # ★白色が70%未満ならBBOXを作成しない
            if white_ratio < HSV_WHITE_RATIO_THRESHOLD:
                continue

            # ★アスペクト比
            reliable = is_aspect_ratio_reliable(
                pixel_width,
                pixel_height,
            )

            # ★距離
            distance = self.get_smoothed_distance(
                pixel_height,
                reliable,
            )

            # ★中心X
            center_x = (x1 + x2) // 2

            # ★HSVでは白色割合をConfidenceとして使用
            confidence = white_ratio

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
                    "model_name": f"HSV_{color}",
                    # ★白色割合を保存
                    "white_ratio": white_ratio,
                }
            )

        return box_data_list


    # ========================================================
    # YOLO推論
    # ========================================================

    def _run_inference(self, frame):

        # ----------------------------------------------------
        # 現在使用しているモデル
        # ----------------------------------------------------

        model_index = (
            self.active_model_index
        )

        model = self.models[
            model_index
        ]

        model_name = MODEL_NAMES[
            model_index
        ]

        # ★現在のモデル色
        model_color = model_name.split("_")[0].lower()


        # ----------------------------------------------------
        # ★YOLO BBOX内部の色判定用HSV
        #
        # ★ここはYOLO推論時だけ実行する。
        # ★HSV認識モードでは使用しない。
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        # ★赤色マスク
        red_mask_1 = cv2.inRange(
            hsv,
            HSV_RED_LOW_1,
            HSV_RED_HIGH_1,
        )

        red_mask_2 = cv2.inRange(
            hsv,
            HSV_RED_LOW_2,
            HSV_RED_HIGH_2,
        )

        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2,
        )

        # ★青色マスク
        blue_mask = cv2.inRange(
            hsv,
            HSV_BLUE_LOW,
            HSV_BLUE_HIGH,
        )


        # ----------------------------------------------------
        # YOLO推論
        # ----------------------------------------------------

        results = model(
            frame,
            conf=self.conf_thres,
            imgsz=640,
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
                # BBOX座標を画像範囲内に制限
                # --------------------------------------------

                x1 = max(
                    0,
                    min(x1, frame.shape[1])
                )

                y1 = max(
                    0,
                    min(y1, frame.shape[0])
                )

                x2 = max(
                    0,
                    min(x2, frame.shape[1])
                )

                y2 = max(
                    0,
                    min(y2, frame.shape[0])
                )


                # --------------------------------------------
                # BBOXサイズ
                # --------------------------------------------

                pixel_width = (
                    x2 - x1
                )

                pixel_height = (
                    y2 - y1
                )

                if (
                    pixel_width <= 0
                    or pixel_height <= 0
                ):
                    continue


                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = float(
                    box.conf[0].item()
                )


                # =================================================
                # ★YOLO BBOX内部の反対色判定
                #
                # ★赤モデル
                # ★    BBOX内部の青色割合を計算
                #
                # ★青モデル
                # ★    BBOX内部の赤色割合を計算
                #
                # ★ここで除外されたBBOXは、
                # ★距離計算やbrocks_info送信を行わない。
                # =================================================

                if model_color == "red":

                    # ★赤モデルなので青色を反対色とする
                    opposite_mask = blue_mask

                    # ★BBOX内部の青色画素数
                    opposite_pixel_count = cv2.countNonZero(
                        opposite_mask[
                            y1:y2,
                            x1:x2,
                        ]
                    )

                elif model_color == "blue":

                    # ★青モデルなので赤色を反対色とする
                    opposite_mask = red_mask

                    # ★BBOX内部の赤色画素数
                    opposite_pixel_count = cv2.countNonZero(
                        opposite_mask[
                            y1:y2,
                            x1:x2,
                        ]
                    )

                else:

                    # ★未知の色の場合は反対色判定をしない
                    opposite_pixel_count = 0


                # ★BBOX全体の画素数
                bbox_pixel_count = (
                    pixel_width
                    * pixel_height
                )


                # ★反対色割合
                opposite_ratio = (
                    opposite_pixel_count
                    / bbox_pixel_count
                )


                # ★反対色が30%以上ならBBOXを除外
                if (
                    opposite_ratio
                    >= YOLO_OPPOSITE_COLOR_RATIO_THRESHOLD
                ):

                    self.get_logger().debug(
                        f"YOLO BBOX rejected: "
                        f"model={model_color}, "
                        f"opposite_ratio="
                        f"{opposite_ratio * 100:.1f}%"
                    )

                    continue


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

                        # ★YOLO時の反対色割合を保存
                        "opposite_ratio": opposite_ratio,
                    }
                )


        return box_data_list


    # ========================================================
    # BBOX描画
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

        target_x = (
            frame.shape[1] // 2
        )


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


            # ★HSV認識の場合は白色割合を表示
            if "white_ratio" in box_data:

                label_text += (
                    f" 白:{box_data['white_ratio'] * 100:.1f}%"
                )


            # ★YOLO認識の場合は反対色割合を表示
            if "opposite_ratio" in box_data:

                label_text += (
                    f" 反対色:"
                    f"{box_data['opposite_ratio'] * 100:.1f}%"
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


            # ★HSV認識の場合は白色割合を送信
            if "white_ratio" in box_data:

                info_lines[-1] += (
                    f",white_ratio="
                    f"{box_data['white_ratio']:.3f}"
                )


            # ★YOLO認識の場合は反対色割合を送信
            if "opposite_ratio" in box_data:

                info_lines[-1] += (
                    f",opposite_ratio="
                    f"{box_data['opposite_ratio']:.3f}"
                )


        return info_lines


    # ========================================================
    # brocks_info送信
    # ========================================================

    def _publish_info(
        self,
        info_lines,
    ):

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

    def _publish_image(
        self,
        frame,
    ):

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

        height, width, channels = (
            frame.shape
        )

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

        msg.step = (
            width * channels
        )


        # ----------------------------------------------------
        # ndarray → bytes
        #
        # msg.data = frame.tobytes() のように通常のsetter経由で
        # 代入すると、rclpyが生成するメッセージクラスの検証処理が
        # 全バイト(数百万要素)をPythonループでチェックするため、
        # 1回あたり150〜250ms程度の大きな遅延が発生する。
        # (実測: publish=176〜250ms → 修正後 publish=数ms)
        #
        # 内部属性(_data)に直接代入することでこの検証処理を
        # バイパスする。シリアライズはバッファプロトコル経由で
        # 行われるため、numpy配列(uint8, 1次元)でも問題なく動作する。
        # ----------------------------------------------------

        msg._data = np.asarray(
            frame,
            dtype=np.uint8,
        ).reshape(-1)


        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.publisher.publish(
            msg
        )


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

    rclpy.init(
        args=args
    )


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

    executor.add_node(
        node
    )


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