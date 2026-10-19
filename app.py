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
    page_title="云端物体检测器",
    page_icon="🔍",
    layout="wide",
)


st.title("🔍 云端物体检测器")

st.write(
    "使用预训练 YOLO 模型，在 GitHub Codespaces 云端对普通图片进行物体检测。"
)

st.warning(
    "上传的图片会进入当前 GitHub Codespace 云端。"
    "本应用代码不会主动把图片写入 Git 仓库或发送到用户指定的 URL。"
    "请不要上传身份证、私人照片、未公开论文数据或敏感遥感原图。"
)


def read_safe_image(raw: bytes) -> Image.Image:
    """Validate and convert image bytes without saving them to disk."""

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("图片文件不能超过 10 MB。")

    try:
        # First pass: validate the file structure.
        with Image.open(BytesIO(raw)) as probe:
            probe.verify()

        # Second pass: reopen after verify(), then decode safely.
        with Image.open(BytesIO(raw)) as source:
            width, height = source.size

            if width * height > MAX_PIXELS:
                raise ValueError("图片像素数量过大。")

            # Correct EXIF orientation and convert to RGB.
            # The returned image does not retain the original EXIF metadata.
            result = ImageOps.exif_transpose(source).convert("RGB")
            return result.copy()

    except ValueError:
        raise
    except Exception:
        raise ValueError(
            "无法读取该图片，请换用正常的 JPG、PNG 或 WebP 文件。"
        ) from None


def read_cloud_sample() -> Image.Image:
    """
    Read a fixed sample image that was downloaded inside Codespaces.
    This path is fixed and is never taken from user input.
    """

    if not SAMPLE_PATH.is_file():
        raise ValueError(
            "Codespace 中还没有 sample.jpg。"
            "请先按照说明在云端终端下载示例图片。"
        )

    try:
        raw = SAMPLE_PATH.read_bytes()
    except OSError:
        raise ValueError("无法读取云端示例图片。") from None

    return read_safe_image(raw)


@st.cache_resource
def load_model() -> YOLO:
    """
    Load the fixed official pretrained model.
    The model is not selected by user input.
    """

    return YOLO("yolo26n.pt")


source_type = st.radio(
    "选择图片来源",
    [
        "Codespace 云端示例图",
        "浏览器上传图片",
    ],
)


image = None

if source_type == "Codespace 云端示例图":
    try:
        image = read_cloud_sample()
        st.caption("该示例图片位于 Codespace 云端，不来自你的本地电脑。")
    except ValueError as error:
        st.info(str(error))

else:
    uploaded_file = st.file_uploader(
        "选择一张普通图片",
        type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=10,
    )

    if uploaded_file is not None:
        try:
            image = read_safe_image(uploaded_file.getvalue())
        except ValueError as error:
            st.error(str(error))


confidence = st.slider(
    "最低置信度",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)


if image is not None:
    st.subheader("输入图片")
    st.image(image, caption="待检测图片")

    if st.button("开始检测", type="primary"):
        try:
            with st.spinner(
                "正在加载模型并进行检测。首次运行会下载模型文件，请稍候……"
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
                "检测失败。网页中没有显示详细错误信息；"
                "如需排查，请查看 Codespace 终端。"
            )
            st.stop()

        # Ultralytics plot() returns a BGR NumPy image.
        # Convert it to RGB for Streamlit.
        annotated_image = result.plot()[:, :, ::-1]

        st.subheader("检测结果")
        st.image(
            annotated_image,
            caption="带检测框的结果",
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
                        "类别": class_name,
                        "置信度": round(float(score), 3),
                        "左上角 X": round(float(xyxy[0]), 1),
                        "左上角 Y": round(float(xyxy[1]), 1),
                        "右下角 X": round(float(xyxy[2]), 1),
                        "右下角 Y": round(float(xyxy[3]), 1),
                    }
                )

        if rows:
            st.success(f"检测到 {len(rows)} 个目标")
            st.subheader("检测明细")
            st.dataframe(rows)
        else:
            st.info("没有检测到达到当前置信度阈值的目标。")