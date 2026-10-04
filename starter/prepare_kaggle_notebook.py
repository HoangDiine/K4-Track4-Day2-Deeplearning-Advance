"""Refresh the source-code snapshot embedded in lab_day2.ipynb."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "starter" / "lab_day2.ipynb"


def main() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.write(ROOT / "eval.py", "eval.py")
        for source in sorted((ROOT / "starter").glob("*.py")):
            if source.name != Path(__file__).name:
                archive.write(source, f"starter/{source.name}")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if cell.get("id") not in {"step3-inference-only-title", "step3-inference-only-code"}
    ]

    def set_cell(index: int, source: str) -> None:
        notebook["cells"][index]["source"] = source.splitlines(keepends=True)

    set_cell(0, """# Lab Day 2 — DeepWeeds on Kaggle / Colab

Notebook tái lập EDA, so sánh backbone, ablation công thức, suy luận và vòng chung kết. Mọi lựa chọn cấu hình dùng fold-0 validation; test chỉ dùng ở vòng chung kết, một lượt cho mỗi seed.

Colab có thể chạy CPU; bật Internet để cài thư viện và tải DeepWeeds. Nếu có GPU, chọn GPU để đo FP16 và tốc độ real-time. Notebook chứa snapshot `eval.py` + `starter/*.py`. Muốn chỉ đánh giá I00–I08, chạy ô inference-only với checkpoint đã có và bỏ qua ô huấn luyện.
""")
    set_cell(1, "## 0. Cài thư viện\n")
    set_cell(3, """## 1. Mã nguồn và môi trường

Cell kế tiếp giải nén snapshot mã nguồn được nhúng trong notebook vào thư mục làm việc. Dataset và checkpoint không nằm trong snapshot.
""")
    set_cell(4, '''from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import base64, json, os, platform, sys, zipfile

from IPython.display import Image, FileLink, display

SOURCE_BUNDLE_BASE64 = """"""
WORK_ROOT = Path("/kaggle/working" if Path("/kaggle/working").is_dir() else "/content")
ROOT = WORK_ROOT / "K4-Track4-Day2-Deeplearning-Advance"
ROOT.mkdir(parents=True, exist_ok=True)
with ZipFile(BytesIO(base64.b64decode(SOURCE_BUNDLE_BASE64))) as archive:
    archive.extractall(ROOT)
assert (ROOT / "eval.py").is_file() and (ROOT / "starter" / "lab_workflow.py").is_file(), ROOT
os.chdir(ROOT)
CODE_DIR = ROOT / "starter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

%pip -q install timm fvcore openpyxl scikit-learn
import numpy as np, pandas as pd, torch, timm
print("Python", platform.python_version(), "| torch", torch.__version__, "| timm", timm.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")
OUTPUT_DIR = WORK_ROOT / "lab-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENVIRONMENT = {"python": platform.python_version(), "torch": torch.__version__,
               "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"}
(OUTPUT_DIR / "environment.json").write_text(json.dumps(ENVIRONMENT, indent=2), encoding="utf-8")
''')
    set_cell(5, "## 2. Tải DeepWeeds và giữ nguyên fold 0\n\nCell kế tiếp cần Internet để tải ảnh từ Zenodo và CSV fold gốc từ GitHub.\n")
    set_cell(7, """## 4. EDA (tuỳ chọn)

Kiểm tra split fold 0 và lưu bảng/ảnh phân bố lớp cùng ví dụ train. Không chạy optimizer hay cập nhật trọng số trong bước này.
""")
    set_cell(8, '''from lab_workflow import prepare_data_and_eda
split_report = prepare_data_and_eda(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)
display(pd.read_csv(OUTPUT_DIR / "eda_class_counts.csv"))
display(Image(filename=str(OUTPUT_DIR / "eda_class_distribution.png")))
display(Image(filename=str(OUTPUT_DIR / "eda_train_examples.png")))
print(split_report)
''')
    set_cell(9, """## 5. Tuỳ chọn — so sánh backbone, ablation huấn luyện và inference tích hợp

Ô này chạy các thí nghiệm huấn luyện nếu checkpoint chưa có trong cache, rồi đo I00–I08 trên validation. Nếu chỉ cần Bước 3 và đã có checkpoint, hãy dùng ô inference-only ở trên; ô đó không gọi code huấn luyện.
""")
    set_cell(11, "## 6. Bước 4–5 — Chung kết, chấm và sản phẩm\n\nChỉ chạy sau khi hoàn thành huấn luyện ở phần tuỳ chọn. Phần này huấn luyện lại F01 và T00 với cùng ba seed; không chạy khi chỉ đánh giá Bước 3. Sau đó chạy eval.py score, eval.py grade và tạo results.xlsx.\n")
    set_cell(13, "## Tải kết quả về máy\n\nColab lưu gói nộp trong `/content`; link tải hiện ở cell dưới. Gói không chứa ảnh DeepWeeds hoặc checkpoint.\n")
    set_cell(14, '''import shutil
submission_zip = shutil.make_archive(str(WORK_ROOT / STUDENT_SLUG), "zip",
                                     root_dir=submission_dir.parent, base_dir=submission_dir.name)
print("Submission archive:", submission_zip)
display(FileLink(submission_zip))
''')

    inference_title = {
        "cell_type": "markdown",
        "id": "step3-inference-only-title",
        "metadata": {},
        "source": """## 3. Bước 3 — I00–I08 trên checkpoint có sẵn (không train lại)

Chạy sau ô tải dữ liệu. Đưa checkpoint đã huấn luyện vào Colab (Files hoặc Drive), điền đường dẫn `.pth` và đúng tên backbone. Có thể thêm tối đa hai checkpoint khác cho I05 (tổng ensemble 2–3 mô hình). Ô này chỉ tính validation, accuracy/calibration và latency p50/p95/p99, throughput ở batch 1 và batch 32; không tạo hay cập nhật trọng số. Bỏ qua EDA và các ô huấn luyện bên dưới nếu chỉ làm Bước 3. Trên CPU, I08 kiểm tra gộp Conv-BN bằng FP32; phần FP16 được đánh dấu chưa chạy.
""".splitlines(keepends=True),
    }
    inference_code = {
        "cell_type": "code",
        "execution_count": None,
        "id": "step3-inference-only-code",
        "metadata": {},
        "outputs": [],
        "source": '''from lab_workflow import run_inference_only

# Đưa checkpoint đã huấn luyện vào Colab rồi điền đường dẫn thực tế.
CHECKPOINT_PATH = None  # Ví dụ: Path("/content/best.pth")
MODEL_BACKBONE = "convnext_tiny"
ENSEMBLE_CHECKPOINTS = [
    # Tối đa hai model khác: {"backbone": "swin_tiny_patch4_window7_224",
    #                         "checkpoint": "/content/other-model/best.pth"}
]

if CHECKPOINT_PATH is None:
    print("Chưa đặt CHECKPOINT_PATH. Không có model nào được train; hãy gắn checkpoint rồi điền đường dẫn.")
else:
    inference_study = run_inference_only(
        CHECKPOINT_PATH, MODEL_BACKBONE, IMAGES_DIR, LABELS_DIR, OUTPUT_DIR,
        ensemble_checkpoints=ENSEMBLE_CHECKPOINTS, batch_size=32, num_workers=2)
    display(pd.DataFrame(inference_study["rows"]))
    print("Realtime candidate:", inference_study["realtime_candidate"])
'''.splitlines(keepends=True),
    }
    data_cell_index = next(
        index for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and "IMAGES_DIR = discover_images_dir(DATA_DIR)" in "".join(cell["source"])
    )
    notebook["cells"][data_cell_index + 1:data_cell_index + 1] = [inference_title, inference_code]

    setup_cell = notebook["cells"][4]
    source = "".join(setup_cell["source"])
    source, replacements = re.subn(
        r'SOURCE_BUNDLE_BASE64 = """.*?"""',
        f'SOURCE_BUNDLE_BASE64 = """{payload}"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Could not find SOURCE_BUNDLE_BASE64 placeholder in setup cell")
    setup_cell["source"] = source.splitlines(keepends=True)

    student_cell = "".join(notebook["cells"][12]["source"])
    student_cell = re.sub(
        r"STUDENT_SLUG = .*?\nassert .*?\n",
        'STUDENT_SLUG = "student_submission"  # Thay bằng MSSV_ten_khong_dau khi nộp.\n',
        student_cell,
        count=1,
    )
    notebook["cells"][12]["source"] = student_cell.splitlines(keepends=True)

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Embedded {len(payload):,} base64 characters from eval.py and starter/*.py")


if __name__ == "__main__":
    main()
