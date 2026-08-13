import threading
import cv2


class LatestFrameReader:
    # cv2.VideoCaptureは読み取り速度が追いつかないと内部バッファに
    # フレームが溜まり続け、read()のたびに古いフレームを返すようになる。
    # 別スレッドで常にread()し続け、最新の1枚だけを保持することでラグを防ぐ。

    def __init__(self, index, backend, width, height, fourcc="MJPG"):
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # デフォルトのYUY2(無圧縮)はUSB帯域を食い、高解像度では
        # カメラ側が内部的にフレームレートを落としてラグの原因になる。
        # MJPEG(圧縮)に切り替えることで帯域を大きく節約できる。
        # (CAP_DSHOWでは解像度設定の後にFOURCCを設定しないと反映されないカメラがある)

        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        # fourcc/解像度の設定だけだとカメラのデフォルトFPS(5fps等)の
        # ままになることがあるため、FPSも明示的に指定する。
        # (ドライバによってはfourcc/解像度設定の後でないと反映されないため
        #  この順序で呼び出す)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 内部バッファを最小にして、read()時に古いフレームが
        # 溜まって返ってくる遅延を減らす
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(
            f"[LatestFrameReader] index={index} "
            f"fourcc={actual_fourcc_str!r} 解像度={actual_w}x{actual_h} fps={actual_fps:.1f}"
        )

        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()