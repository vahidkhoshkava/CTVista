
import os

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
from scipy import ndimage


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CTVista",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# TRANSLATIONS
# ============================================================

TEXT = {
    "English": {
        "subtitle": "AI-assisted chest CT review workspace",
        "tagline": "Review • Explain • Compare • Summarize",
        "study": "DEMO-001",
        "scan": "Non-contrast chest CT",
        "analysis": "AI analysis complete",
        "viewer": "CT Image Review",
        "slice": "Axial slice",
        "slice_label": "Slice",
        "window": "Display window",
        "lung": "Lung",
        "soft": "Soft tissue",
        "bone": "Bone",
        "assistant": "AI Review Assistant",
        "progress": "Review progress",
        "region": "Region",
        "ai_note": "AI note",
        "status": "Status",
        "open": "Open",
        "reviewed": "Reviewed",
        "selected_region": "Selected region",
        "inspect": "What to inspect",
        "mark_reviewed": "Mark as reviewed",
        "undo_reviewed": "Remove reviewed status",
        "insight": "AI Insight",
        "similar": "AI Similar Cases",
        "similar_note": (
            "Prototype preview. In a clinical deployment, the AI would retrieve "
            "comparable prior CT studies and reports from the hospital archive."
        ),
        "draft": "AI Draft Report",
        "generate_draft": "Generate draft findings",
        "draft_empty": "Review at least one region before generating the draft.",
        "three_d": "Optional 3D Anatomical Overview",
        "generate_3d": "Generate lightweight 3D overview",
        "three_note": (
            "Approximate air-space visualization for demonstration—not clinical segmentation."
        ),
        "value": "Why CTVista matters",
        "values": [
            ("AI-guided review", "Directs attention to high-value anatomical regions."),
            ("AI explanation", "Explains why a region is suggested for review."),
            ("AI case retrieval", "Connects the current study to comparable prior cases."),
            ("AI draft support", "Transforms reviewed regions into a structured draft."),
        ],
        "warning": (
            "Research and investor demonstration. "
            "All outputs require qualified radiologist verification."
        ),
    },
    "فارسی": {
        "subtitle": "محیط هوشمند برای بررسی سی‌تی‌اسکن قفسه سینه",
        "tagline": "بررسی • توضیح • مقایسه • خلاصه‌سازی",
        "study": "مطالعه نمایشی ۰۰۱",
        "scan": "سی‌تی‌اسکن بدون تزریق قفسه سینه",
        "analysis": "تحلیل هوشمند تکمیل شده است",
        "viewer": "بررسی تصاویر سی‌تی",
        "slice": "برش محوری",
        "slice_label": "برش",
        "window": "پنجره نمایش",
        "lung": "ریه",
        "soft": "بافت نرم",
        "bone": "استخوان",
        "assistant": "دستیار هوشمند بررسی",
        "progress": "پیشرفت بررسی",
        "region": "ناحیه",
        "ai_note": "یادداشت هوشمند",
        "status": "وضعیت",
        "open": "باز",
        "reviewed": "بررسی شد",
        "selected_region": "ناحیه انتخاب‌شده",
        "inspect": "مواردی که باید بررسی شوند",
        "mark_reviewed": "علامت‌گذاری به‌عنوان بررسی‌شده",
        "undo_reviewed": "حذف وضعیت بررسی‌شده",
        "insight": "بینش هوشمند",
        "similar": "موارد مشابه هوشمند",
        "similar_note": (
            "این بخش نمایشی است. در نسخه بالینی، هوش مصنوعی مطالعات و گزارش‌های "
            "مشابه قبلی را از آرشیو بیمارستان بازیابی می‌کند."
        ),
        "draft": "پیش‌نویس هوشمند گزارش",
        "generate_draft": "ایجاد پیش‌نویس یافته‌ها",
        "draft_empty": "پیش از ایجاد پیش‌نویس، حداقل یک ناحیه را بررسی کنید.",
        "three_d": "نمای کلی سه‌بعدی اختیاری",
        "generate_3d": "ایجاد نمای سه‌بعدی سبک",
        "three_note": (
            "نمایش تقریبی فضای هوادار ریه برای نسخه نمایشی است، نه تقسیم‌بندی بالینی."
        ),
        "value": "چرا CTVista اهمیت دارد؟",
        "values": [
            ("بررسی هدایت‌شده هوشمند", "توجه را به ناحیه‌های مهم آناتومیک هدایت می‌کند."),
            ("توضیح هوشمند", "علت پیشنهاد هر ناحیه را توضیح می‌دهد."),
            ("بازیابی هوشمند موارد", "مطالعه فعلی را به موارد مشابه قبلی متصل می‌کند."),
            ("پشتیبانی هوشمند پیش‌نویس", "ناحیه‌های بررسی‌شده را به پیش‌نویس ساختاریافته تبدیل می‌کند."),
        ],
        "warning": (
            "نسخه نمایشی پژوهشی و سرمایه‌گذاری است و تمام خروجی‌ها "
            "باید توسط رادیولوژیست واجد صلاحیت تأیید شوند."
        ),
    },
}


# ============================================================
# REVIEW DATA
# ============================================================

ANATOMY = {
    "Pleural spaces": {
        "fa": "فضاهای جنب",
        "priority": "high",
        "window": "lung",
        "fraction": 0.42,
        "note_en": "Possible pleural-fluid pattern",
        "note_fa": "الگوی احتمالی مایع پلور",
        "inspect_en": ["Pleural effusion", "Pneumothorax", "Pleural thickening", "Pleural mass"],
        "inspect_fa": ["افیوژن پلور", "پنوموتوراکس", "ضخیم‌شدگی پلور", "توده پلور"],
        "why_en": (
            "The scan representation is most similar to text descriptions associated "
            "with pleural fluid. Inspect posterior and lower pleural spaces."
        ),
        "why_fa": (
            "نمایش اسکن بیشترین شباهت را به توصیف‌های مرتبط با مایع پلور دارد. "
            "فضاهای خلفی و تحتانی پلور بررسی شوند."
        ),
    },
    "Lung parenchyma": {
        "fa": "بافت ریه و کدورت‌ها",
        "priority": "high",
        "window": "lung",
        "fraction": 0.50,
        "note_en": "Possible collapse or opacity pattern",
        "note_fa": "الگوی احتمالی کلاپس یا کدورت",
        "inspect_en": ["Atelectasis", "Consolidation", "Ground-glass opacity", "Diffuse opacity"],
        "inspect_fa": ["آتِلکتازی", "کنسولیدیشن", "کدورت شیشه مات", "کدورت منتشر"],
        "why_en": (
            "The scan resembles studies containing volume loss and lung opacity. "
            "Review dependent lower-lobe regions."
        ),
        "why_fa": (
            "اسکن به مطالعات دارای کاهش حجم و کدورت ریوی شباهت دارد. "
            "نواحی وابسته لوب‌های تحتانی بررسی شوند."
        ),
    },
    "Airways": {
        "fa": "راه‌های هوایی",
        "priority": "medium",
        "window": "lung",
        "fraction": 0.58,
        "note_en": "Possible bronchial-wall change",
        "note_fa": "تغییر احتمالی دیواره برونش",
        "inspect_en": ["Wall thickening", "Bronchiectasis", "Mucus plugging", "Airway narrowing"],
        "inspect_fa": ["ضخیم‌شدگی دیواره", "برونشکتازی", "پلاگ موکوسی", "تنگی راه هوایی"],
        "why_en": (
            "The image–text representation resembles descriptions of peribronchial change. "
            "Review central and segmental airways."
        ),
        "why_fa": (
            "نمایش تصویر و متن به توصیف‌های تغییرات اطراف برونش شباهت دارد. "
            "راه‌های هوایی مرکزی و سگمنتال بررسی شوند."
        ),
    },
    "Pulmonary nodules": {
        "fa": "ندول‌های ریوی",
        "priority": "medium",
        "window": "lung",
        "fraction": 0.52,
        "note_en": "Small focal region worth inspecting",
        "note_fa": "ناحیه کانونی کوچک نیازمند بررسی",
        "inspect_en": ["Solid nodules", "Subsolid nodules", "Calcified nodules", "Multiplicity"],
        "inspect_fa": ["ندول جامد", "ندول نیمه‌جامد", "ندول کلسیفیه", "تعداد ندول‌ها"],
        "why_en": (
            "The scan has a moderate semantic match to nodule-related descriptions. "
            "A systematic review of both lungs remains necessary."
        ),
        "why_fa": (
            "اسکن تطابق معنایی متوسطی با توصیف‌های مرتبط با ندول دارد. "
            "بررسی نظام‌مند هر دو ریه ضروری است."
        ),
    },
    "Heart and pericardium": {
        "fa": "قلب و پریکارد",
        "priority": "medium",
        "window": "soft",
        "fraction": 0.38,
        "note_en": "Cardiac and pericardial review",
        "note_fa": "بررسی قلب و پریکارد",
        "inspect_en": ["Cardiac size", "Pericardial fluid", "Coronary calcification", "Cardiac contour"],
        "inspect_fa": ["اندازه قلب", "مایع پریکارد", "کلسیفیکاسیون کرونر", "کانتور قلب"],
        "why_en": (
            "This region is included to ensure systematic review of cardiac size, "
            "pericardium, and visible coronary calcification."
        ),
        "why_fa": (
            "این ناحیه برای تضمین بررسی نظام‌مند اندازه قلب، پریکارد و "
            "کلسیفیکاسیون قابل مشاهده کرونر در نظر گرفته شده است."
        ),
    },
    "Mediastinum": {
        "fa": "مدیاستن و گره‌های لنفاوی",
        "priority": "routine",
        "window": "soft",
        "fraction": 0.60,
        "note_en": "Routine mediastinal review",
        "note_fa": "بررسی روتین مدیاستن",
        "inspect_en": ["Lymph nodes", "Mediastinal mass", "Great vessels", "Esophagus"],
        "inspect_fa": ["گره‌های لنفاوی", "توده مدیاستن", "عروق بزرگ", "مری"],
        "why_en": (
            "The mediastinum should be reviewed systematically even when it is not "
            "among the highest-priority suggestions."
        ),
        "why_fa": (
            "مدیاستن حتی زمانی که در پیشنهادهای اصلی نیست باید به‌صورت نظام‌مند بررسی شود."
        ),
    },
    "Bones": {
        "fa": "استخوان‌ها",
        "priority": "routine",
        "window": "bone",
        "fraction": 0.48,
        "note_en": "Routine skeletal review",
        "note_fa": "بررسی روتین اسکلت",
        "inspect_en": ["Ribs", "Vertebrae", "Sternum", "Focal bone lesions"],
        "inspect_fa": ["دنده‌ها", "مهره‌ها", "جناغ", "ضایعات کانونی استخوان"],
        "why_en": (
            "Bone-window review can reveal fractures and focal osseous abnormalities "
            "that are less apparent on lung windows."
        ),
        "why_fa": (
            "پنجره استخوان می‌تواند شکستگی‌ها و ضایعات کانونی را که "
            "در پنجره ریه کمتر دیده می‌شوند آشکار کند."
        ),
    },
    "Upper abdomen": {
        "fa": "بخش فوقانی شکم",
        "priority": "routine",
        "window": "soft",
        "fraction": 0.12,
        "note_en": "Review included upper-abdominal structures",
        "note_fa": "بررسی ساختارهای فوقانی شکم",
        "inspect_en": ["Liver", "Adrenal glands", "Spleen", "Upper kidneys"],
        "inspect_fa": ["کبد", "غدد فوق کلیوی", "طحال", "بخش فوقانی کلیه‌ها"],
        "why_en": (
            "Chest CT often includes part of the upper abdomen, and visible structures "
            "should not be overlooked."
        ),
        "why_fa": (
            "سی‌تی قفسه سینه معمولاً بخشی از شکم فوقانی را نیز شامل می‌شود "
            "و این ساختارها نباید نادیده گرفته شوند."
        ),
    },
}

ORDER = list(ANATOMY.keys())


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 0.9rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 22px 28px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(110deg, #102b4e, #126a87);
        margin-bottom: 10px;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 30px;
        align-items: center;
    }

    .brand {
        font-size: 2.3rem;
        font-weight: 760;
        line-height: 1.08;
    }

    .version {
        font-size: 0.75rem;
        opacity: 0.8;
        margin-left: 6px;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        margin-top: 8px;
        font-weight: 500;
    }

    .hero-tagline {
        opacity: 0.94;
        margin-top: 8px;
    }

    .developer-card {
        text-align: right;
        padding-left: 28px;
        border-left: 1px solid rgba(255,255,255,.28);
        min-width: 260px;
    }

    .developer-label {
        font-size: 0.82rem;
        opacity: 0.72;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .developer-name {
        font-size: 1.08rem;
        font-weight: 650;
        margin-top: 5px;
    }

    .developer-role {
        font-size: 0.86rem;
        opacity: 0.78;
        margin-top: 3px;
    }

    .study {
        padding: 11px 15px;
        border: 1px solid rgba(128,128,128,.23);
        border-radius: 13px;
        margin-bottom: 12px;
    }

    .priority-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    .dot-high {background: #d73746;}
    .dot-medium {background: #e8911e;}
    .dot-routine {background: #2a915f;}

    .review-table {
        border: 1px solid rgba(128,128,128,.23);
        border-radius: 13px;
        overflow: hidden;
    }

    .review-row {
        display: grid;
        grid-template-columns: 30px 1.1fr 1.6fr 85px;
        align-items: center;
        gap: 8px;
        padding: 9px 10px;
        border-bottom: 1px solid rgba(128,128,128,.15);
    }

    .review-row:last-child {
        border-bottom: none;
    }

    .review-header {
        font-weight: 700;
        background: rgba(128,128,128,.08);
    }

    .status-chip {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        background: rgba(128,128,128,.12);
        font-size: .82rem;
        text-align: center;
    }

    .box {
        padding: 14px;
        border: 1px solid rgba(128,128,128,.23);
        border-radius: 13px;
        min-height: 100%;
    }

    .progress-track {
        height: 10px;
        border-radius: 999px;
        background: rgba(128,128,128,.18);
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: #1887a8;
        border-radius: 999px;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 18px;
        }

        .hero-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }

        .developer-card {
            text-align: left;
            padding-left: 0;
            padding-top: 14px;
            border-left: none;
            border-top: 1px solid rgba(255,255,255,.28);
            min-width: 0;
        }

        .brand {
            font-size: 1.9rem;
        }

        .review-row {
            grid-template-columns: 24px 1fr;
        }

        .review-row .ai-note,
        .review-row .status-cell {
            grid-column: 2;
        }

        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner="Downloading demonstration CT…")
def load_ct():
    token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))

    if not token:
        raise RuntimeError("HF_TOKEN is missing in Streamlit Secrets.")

    path = hf_hub_download(
        repo_id="ibrahimhamamci/CT-RATE",
        repo_type="dataset",
        filename="dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz",
        token=token,
    )

    return path, nib.as_closest_canonical(nib.load(path))


def get_slice(nii, z, vmin, vmax):
    image = np.asarray(
        nii.dataobj[:, :, int(z)],
        dtype=np.float32,
    )

    if image.min() >= 0 and image.max() > 3000:
        image -= 1024.0

    image = np.clip(
        (image - vmin) / (vmax - vmin),
        0,
        1,
    )

    return np.flipud(image.T)


@st.cache_data(show_spinner="Creating lightweight 3D view…")
def create_3d(path):
    nii = nib.as_closest_canonical(nib.load(path))
    step = 5

    raw = np.asarray(
        nii.dataobj[::step, ::step, ::step],
        dtype=np.float32,
    )

    if raw.min() >= 0 and raw.max() > 3000:
        raw -= 1024.0

    air = (raw > -1000) & (raw < -400)

    boundary = np.zeros_like(air, dtype=bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True

    internal = air & ~ndimage.binary_propagation(
        boundary,
        mask=air,
    )

    labels, count = ndimage.label(internal)

    if count == 0:
        raise RuntimeError("No internal air-space regions were detected.")

    sizes = ndimage.sum(
        internal,
        labels,
        index=np.arange(1, count + 1),
    )

    keep = np.argsort(sizes)[-min(5, len(sizes)):] + 1
    points = np.argwhere(np.isin(labels, keep))

    if len(points) > 10000:
        points = points[
            np.random.default_rng(42).choice(
                len(points),
                10000,
                replace=False,
            )
        ]

    xyz = points * np.asarray(
        nii.header.get_zooms()[:3]
    ) * step

    figure = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=2.1,
                opacity=.24,
                color=xyz[:, 2],
                colorscale="Turbo",
                showscale=False,
            ),
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=15, b=0),
        scene=dict(
            aspectmode="data",
            xaxis_title="Left–right",
            yaxis_title="Front–back",
            zaxis_title="Head–feet",
        ),
    )

    return figure


# ============================================================
# SESSION STATE
# ============================================================

if "selected_region" not in st.session_state:
    st.session_state.selected_region = "Pleural spaces"

if "reviewed_regions" not in st.session_state:
    st.session_state.reviewed_regions = []

if "show_3d" not in st.session_state:
    st.session_state.show_3d = False


# ============================================================
# HEADER
# ============================================================

language = st.radio(
    "Language / زبان",
    ["English", "فارسی"],
    horizontal=True,
)

t = TEXT[language]
is_farsi = language == "فارسی"
direction = "rtl" if is_farsi else "ltr"

st.markdown(
    f"""
    <div class="hero" dir="{direction}">
        <div class="hero-grid">
            <div>
                <div class="brand">
                    CTVista™
                    <span class="version">v1.0</span>
                </div>

                <div class="hero-subtitle">
                    {t["subtitle"]}
                </div>

                <div class="hero-tagline">
                    {t["tagline"]}
                </div>
            </div>

            <div class="developer-card">
                <div class="developer-label">
                    {"Developed by" if not is_farsi else "توسعه‌دهنده"}
                </div>

                <div class="developer-name">
                    Vahid Khoshkava, PhD
                </div>

                <div class="developer-role">
                    {"Developer" if not is_farsi else "بنیان‌گذار و توسعه‌دهنده"}
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD STUDY
# ============================================================

try:
    ct_path, nii = load_ct()
except Exception as error:
    st.error(str(error))
    st.stop()

number_of_slices = int(nii.shape[2])

st.markdown(
    f"""
    <div class="study" dir="{direction}">
        <strong>{t["study"]}</strong>
        &nbsp; • &nbsp; {t["scan"]}
        &nbsp; • &nbsp; {number_of_slices} slices
        &nbsp; • &nbsp; ✅ {t["analysis"]}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ACTIVE REGION SETTINGS
# ============================================================

selected_name = st.session_state.selected_region
selected_data = ANATOMY[selected_name]

default_slice = int(
    selected_data["fraction"] * (number_of_slices - 1)
)

window_ranges = {
    "lung": (-1000, 400),
    "soft": (-150, 250),
    "bone": (-200, 1500),
}

window_names = {
    "lung": t["lung"],
    "soft": t["soft"],
    "bone": t["bone"],
}


# ============================================================
# TOP WORKSPACE
# ============================================================

viewer_col, guide_col, similar_col = st.columns(
    [1.02, 1.18, 0.86],
    gap="large",
)


# ---------------- CT VIEWER ----------------

with viewer_col:
    st.subheader(t["viewer"])

    st.markdown(
        f"**{t['slice_label']} {default_slice + 1} / {number_of_slices}**"
    )

    slice_index = st.slider(
        t["slice"],
        0,
        number_of_slices - 1,
        default_slice,
        label_visibility="collapsed",
        key=f"slice_{selected_name}",
    )

    options = [
        t["lung"],
        t["soft"],
        t["bone"],
    ]

    selected_window = st.radio(
        t["window"],
        options,
        index=options.index(
            window_names[selected_data["window"]]
        ),
        horizontal=True,
        key=f"window_{selected_name}",
    )

    reverse_window = {
        t["lung"]: "lung",
        t["soft"]: "soft",
        t["bone"]: "bone",
    }

    active_window = reverse_window[selected_window]

    st.image(
        get_slice(
            nii,
            slice_index,
            *window_ranges[active_window],
        ),
        clamp=True,
        width="stretch",
    )

    st.caption(
        f"{t['slice_label']} {slice_index + 1} / {number_of_slices}"
    )


# ---------------- AI REVIEW ASSISTANT ----------------

with guide_col:
    st.subheader(t["assistant"])

    reviewed_count = len(st.session_state.reviewed_regions)
    progress = int(
        100 * reviewed_count / len(ORDER)
    )

    st.markdown(
        f"""
        <div dir="{direction}">
            <strong>{t["progress"]}</strong>
            — {reviewed_count}/{len(ORDER)} {t["reviewed"]}
            <div class="progress-track">
                <div class="progress-fill" style="width:{progress}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    table_html = f"""
    <div class="review-table" dir="{direction}">
        <div class="review-row review-header">
            <div></div>
            <div>{t["region"]}</div>
            <div class="ai-note">{t["ai_note"]}</div>
            <div class="status-cell">{t["status"]}</div>
        </div>
    """

    for region_name in ORDER:
        data = ANATOMY[region_name]

        display_name = (
            data["fa"]
            if is_farsi
            else region_name
        )

        note = (
            data["note_fa"]
            if is_farsi
            else data["note_en"]
        )

        dot_class = {
            "high": "dot-high",
            "medium": "dot-medium",
            "routine": "dot-routine",
        }[data["priority"]]

        status = (
            t["reviewed"]
            if region_name in st.session_state.reviewed_regions
            else t["open"]
        )

        table_html += f"""
        <div class="review-row">
            <div>
                <span class="priority-dot {dot_class}"></span>
            </div>

            <div>
                <strong>{display_name}</strong>
            </div>

            <div class="ai-note">
                {note}
            </div>

            <div class="status-cell">
                <span class="status-chip">{status}</span>
            </div>
        </div>
        """

    st.markdown(
        table_html + "</div>",
        unsafe_allow_html=True,
    )

    display_options = [
        ANATOMY[name]["fa"] if is_farsi else name
        for name in ORDER
    ]

    current_display_name = (
        selected_data["fa"]
        if is_farsi
        else selected_name
    )

    selected_display = st.selectbox(
        t["selected_region"],
        display_options,
        index=display_options.index(current_display_name),
    )

    display_to_internal = {
        (ANATOMY[name]["fa"] if is_farsi else name): name
        for name in ORDER
    }

    new_region = display_to_internal[selected_display]

    if new_region != st.session_state.selected_region:
        st.session_state.selected_region = new_region
        st.rerun()


# ---------------- AI SIMILAR CASES ----------------

with similar_col:
    st.subheader(t["similar"])
    st.info(t["similar_note"])

    for index, region_name in enumerate(ORDER[:3]):
        data = ANATOMY[region_name]

        display_name = (
            data["fa"]
            if is_farsi
            else region_name
        )

        note = (
            data["note_fa"]
            if is_farsi
            else data["note_en"]
        )

        match = [91, 87, 82][index]

        with st.container(border=True):
            title = (
                f"مورد مشابه {index + 1}"
                if is_farsi
                else f"Similar case {index + 1}"
            )

            st.markdown(
                f"**{title}** — {match}% match"
            )

            st.caption(display_name)
            st.write(note)


# ============================================================
# SECOND ROW: SELECTED REGION + AI INSIGHT + DRAFT
# ============================================================

st.divider()

selected_name = st.session_state.selected_region
selected_data = ANATOMY[selected_name]

display_name = (
    selected_data["fa"]
    if is_farsi
    else selected_name
)

inspection_items = (
    selected_data["inspect_fa"]
    if is_farsi
    else selected_data["inspect_en"]
)

explanation = (
    selected_data["why_fa"]
    if is_farsi
    else selected_data["why_en"]
)

review_col, insight_col, draft_col = st.columns(
    [1.05, 1.05, 1.05],
    gap="large",
)


with review_col:
    st.subheader(t["selected_region"])

    st.markdown(
        f"""
        <div class="box" dir="{direction}">
            <h3>{display_name}</h3>
            <strong>{t["inspect"]}</strong>
            <ul>
                {''.join(f'<li>{item}</li>' for item in inspection_items)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_reviewed = (
        selected_name in st.session_state.reviewed_regions
    )

    if st.button(
        t["undo_reviewed"] if is_reviewed else t["mark_reviewed"],
        type="secondary" if is_reviewed else "primary",
        width="stretch",
    ):
        if is_reviewed:
            st.session_state.reviewed_regions.remove(selected_name)
        else:
            st.session_state.reviewed_regions.append(selected_name)

        st.rerun()


with insight_col:
    st.subheader(t["insight"])

    st.markdown(
        f"""
        <div class="box" dir="{direction}">
            <strong>{display_name}</strong><br><br>
            {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )


with draft_col:
    st.subheader(t["draft"])

    if st.button(
        t["generate_draft"],
        width="stretch",
    ):
        if not st.session_state.reviewed_regions:
            st.info(t["draft_empty"])
        else:
            for region_name in st.session_state.reviewed_regions:
                data = ANATOMY[region_name]

                item_name = (
                    data["fa"]
                    if is_farsi
                    else region_name
                )

                note = (
                    data["note_fa"]
                    if is_farsi
                    else data["note_en"]
                )

                st.write(
                    f"• **{item_name}:** {note}"
                )

    st.warning(t["warning"])


# ============================================================
# PRODUCT VALUE
# ============================================================

st.divider()
st.subheader(t["value"])

value_columns = st.columns(4)

for column, (title, body) in zip(
    value_columns,
    t["values"],
):
    with column:
        st.markdown(
            f"""
            <div class="box" dir="{direction}">
                <strong>{title}</strong><br><br>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# OPTIONAL 3D AT BOTTOM
# ============================================================

st.divider()

with st.expander(
    t["three_d"],
    expanded=False,
):
    if st.button(
        t["generate_3d"],
        type="primary",
        width="stretch",
    ):
        st.session_state.show_3d = True

    if st.session_state.show_3d:
        st.plotly_chart(
            create_3d(ct_path),
            width="stretch",
        )

    st.caption(t["three_note"])


st.caption(t["warning"])
