from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
from ultralytics import settings as yolo_settings


# -----------------------------
# Basic security limits
# -----------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
SAMPLE_PATH = Path("sample.jpg")

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

# Disable Ultralytics analytics and crash-report synchronization.
yolo_settings.update({"sync": False})


st.set_page_config(
    page_title="クラウド物体検出器",
    page_icon="🔍",
    layout="wide",
)


st.title("🔍 クラウド物体検出器")

st.write(
    "事前学習済みの YOLO モデルを使用して、GitHub Codespaces のクラウド上で一般的な画像に対して物体検出を行います。"
)

st.warning(
    "アップロードされた画像は、現在の GitHub Codespace のクラウド上に送られます。"
    "本アプリケーションのコードは、画像を Git リポジトリに書き込んだり、ユーザー指定の URL に送信したりすることはありません。"
    "身分証明書、プライベートな写真、未公開の論文データ、機密性の高いリモートセンシング画像はアップロードしないでください。"
)


def read_safe_image(raw: bytes) -> Image.Image:
    """Validate and convert image bytes without saving them to disk."""

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("画像ファイルは 10 MB を超えないでください。")

    try:
        # First pass: validate the file structure.
        with Image.open(BytesIO(raw)) as probe:
            probe.verify()

        # Second pass: reopen after verify(), then decode safely.
        with Image.open(BytesIO(raw)) as source:
            width, height = source.size

            if width * height > MAX_PIXELS:
                raise ValueError("画像のピクセル数が大きすぎます。")

            # Correct EXIF orientation and convert to RGB.
            # The returned image does not retain the original EXIF metadata.
            result = ImageOps.exif_transpose(source).convert("RGB")
            return result.copy()

    except ValueError:
        raise
    except Exception:
        raise ValueError(
            "この画像を読み込めません。正常な JPG、PNG、または WebP ファイルに差し替えてください。"
        ) from None


def read_cloud_sample() -> Image.Image:
    """
    Read a fixed sample image that was downloaded inside Codespaces.
    This path is fixed and is never taken from user input.
    """

    if not SAMPLE_PATH.is_file():
        raise ValueError(
            "Codespace 内にまだ sample.jpg がありません。"
            "まず手順に従って、クラウドのターミナルでサンプル画像をダウンロードしてください。"
        )

    try:
        raw = SAMPLE_PATH.read_bytes()
    except OSError:
        raise ValueError("クラウドのサンプル画像を読み込めません。") from None

    return read_safe_image(raw)


@st.cache_resource
def load_model() -> YOLO:
    """
    Load the fixed official pretrained model.
    The model is not selected by user input.
    """

    return YOLO("yolo26n.pt")


source_type = st.radio(
    "画像の入手元を選択",
    [
        "Codespace クラウドのサンプル画像",
        "ブラウザから画像をアップロード",
    ],
)


image = None

if source_type == "Codespace クラウドのサンプル画像":
    try:
        image = read_cloud_sample()
        st.caption("このサンプル画像は Codespace のクラウド上にあり、お使いのローカル PC からは取得していません。")
    except ValueError as error:
        st.info(str(error))

else:
    uploaded_file = st.file_uploader(
        "画像を 1 枚選択",
        type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=10,
    )

    if uploaded_file is not None:
        try:
            image = read_safe_image(uploaded_file.getvalue())
        except ValueError as error:
            st.error(str(error))


confidence = st.slider(
    "最小信頼度",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)


if image is not None:
    st.subheader("入力画像")
    st.image(image, caption="検出対象の画像")

    if st.button("検出を開始", type="primary"):
        try:
            with st.spinner(
                "モデルを読み込み、検出を実行しています。初回実行時はモデルファイルをダウンロードするため、しばらくお待ちください……"
            ):
                model = load_model()

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    device="cpu",
                    save=False,
                    verbose=False,
                )

                result = results[0]

        except Exception:
            st.error(
                "検出に失敗しました。ウェブページには詳細なエラー情報は表示されません。"
                "原因を調べた場合は、Codespace のターミナルを確認してください。"
            )
            st.stop()

        # Ultralytics plot() returns a BGR NumPy image.
        # Convert it to RGB for Streamlit.
        annotated_image = result.plot()[:, :, ::-1]

        st.subheader("検出結果")
        st.image(
            annotated_image,
            caption="検出枠付きの結果",
            channels="RGB",
        )

        rows = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()

            for xyxy, score, class_id in zip(
                coordinates,
                scores,
                class_ids,
            ):
                class_id = int(class_id)

                if isinstance(result.names, dict):
                    class_name = result.names.get(
                        class_id,
                        str(class_id),
                    )
                else:
                    class_name = result.names[class_id]

                rows.append(
                    {
                        "クラス": class_name,
                        "信頼度": round(float(score), 3),
                        "左上 X": round(float(xyxy[0]), 1),
                        "左上 Y": round(float(xyxy[1]), 1),
                        "右下 X": round(float(xyxy[2]), 1),
                        "右下 Y": round(float(xyxy[3]), 1),
                    }
                )

        if rows:
            st.success(f"検出された {len(rows)} 個の対象")
            st.subheader("検出の詳細")
            st.dataframe(rows)
        else:
            st.info("現在の信頼度閾値に達した対象は検出されませんでした。")
