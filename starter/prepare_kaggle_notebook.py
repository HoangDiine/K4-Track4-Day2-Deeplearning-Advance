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

    def set_cell(index: int, source: str) -> None:
        notebook["cells"][index]["source"] = source.splitlines(keepends=True)

    set_cell(0, """# Lab Day 2 — DeepWeeds on Kaggle / Colab

Notebook tái lập EDA, so sánh backbone, ablation công thức, suy luận và vòng chung kết. Mọi lựa chọn cấu hình dùng fold-0 validation; test chỉ dùng ở vòng chung kết, một lượt cho mỗi seed.

Trên Kaggle, bật **GPU Accelerator** và **Internet** trong Settings rồi chạy các ô theo thứ tự. Notebook chứa snapshot `eval.py` + `starter/*.py` và tự tải DeepWeeds; không cần upload thêm mã nguồn hay dataset. Bước suy luận chạy trên checkpoint đã chọn, không cập nhật trọng số.
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
assert torch.cuda.is_available(), "Bật GPU Accelerator trong Kaggle Settings trước khi chạy."
print("GPU:", torch.cuda.get_device_name(0))
OUTPUT_DIR = WORK_ROOT / "lab-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENVIRONMENT = {"python": platform.python_version(), "torch": torch.__version__,
               "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "gpu": torch.cuda.get_device_name(0)}
(OUTPUT_DIR / "environment.json").write_text(json.dumps(ENVIRONMENT, indent=2), encoding="utf-8")
''')
    set_cell(5, "## 2. Tải DeepWeeds và giữ nguyên fold 0\n\nCell kế tiếp cần Internet để tải ảnh từ Zenodo và CSV fold gốc từ GitHub.\n")
    set_cell(9, """## 4. Bước 1–3 — Backbone, công thức huấn luyện và suy luận

Notebook dùng năm backbone và các ablation đã khai báo trong mã. Bước suy luận I00–I08 chạy trên mô hình/checkpoint đã huấn luyện, chọn cấu hình chỉ bằng validation và đo riêng batch 1 và batch 32.
""")
    set_cell(13, "## Tải kết quả về máy\n\nKaggle lưu gói nộp trong `/kaggle/working`; link tải hiện ở cell dưới. Gói không chứa ảnh DeepWeeds hoặc checkpoint.\n")
    set_cell(14, '''import shutil
submission_zip = shutil.make_archive(str(WORK_ROOT / STUDENT_SLUG), "zip",
                                     root_dir=submission_dir.parent, base_dir=submission_dir.name)
print("Submission archive:", submission_zip)
display(FileLink(submission_zip))
''')

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
