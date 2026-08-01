import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
from scipy import ndimage

st.set_page_config(page_title="CTVista", page_icon="🫁", layout="wide")

TEXT = {
    "English": {
        "subtitle": "AI-assisted chest CT review workspace",
        "viewer": "CT Slice Viewer",
        "slice": "Axial slice",
        "window": "Display window",
        "lung": "Lung",
        "soft": "Soft tissue",
        "bone": "Bone",
        "three_d": "Lightweight 3D Reconstruction",
        "create": "Create 3D view",
        "summary": "AI Review Summary",
        "first": "Review first",
        "next": "Review subsequently",
        "lower": "Lower model match",
        "score_note": "Relative image–text matches—not disease probabilities or diagnoses.",
        "checklist": "Structured Review Checklist",
        "similar": "Similar Case Retrieval",
        "similar_note": "Prototype preview. These are simulated case cards, not retrieved patients.",
        "draft": "AI-Assisted Draft Findings",
        "draft_button": "Generate review draft",
        "draft_warning": "For radiologist review only—not a final report or diagnosis.",
        "why": "Why CTVista matters",
        "disclaimer": "Research and investor demonstration. Radiologist verification is required.",
    },
    "فارسی": {
        "subtitle": "محیط هوشمند برای بررسی سی‌تی‌اسکن قفسه سینه",
        "viewer": "مشاهده برش‌های سی‌تی‌اسکن",
        "slice": "برش محوری",
        "window": "پنجره نمایش",
        "lung": "ریه",
        "soft": "بافت نرم",
        "bone": "استخوان",
        "three_d": "بازسازی سه‌بعدی سبک",
        "create": "ایجاد نمای سه‌بعدی",
        "summary": "خلاصه بررسی هوشمند",
        "first": "اولویت بررسی",
        "next": "بررسی تکمیلی",
        "lower": "تطابق پایین‌تر",
        "score_note": "امتیازها تطابق نسبی تصویر و متن هستند، نه احتمال قطعی بیماری یا تشخیص.",
        "checklist": "چک‌لیست بررسی ساختاریافته",
        "similar": "بازیابی موارد مشابه",
        "similar_note": "این بخش نمایشی است و موارد آن بیمار واقعی بازیابی‌شده نیستند.",
        "draft": "پیش‌نویس یافته‌های هوشمند",
        "draft_button": "ایجاد پیش‌نویس بررسی",
        "draft_warning": "صرفاً برای بررسی رادیولوژیست است، نه گزارش نهایی یا تشخیص.",
        "why": "چرا CTVista اهمیت دارد؟",
        "disclaimer": "نسخه نمایشی پژوهشی و سرمایه‌گذاری است و تأیید رادیولوژیست ضروری است.",
    },
}

FA = {
    "Pleural effusion": "افیوژن پلور",
    "Atelectasis": "آتِلکتازی",
    "Peribronchial thickening": "ضخیم‌شدگی اطراف برونش",
    "Lung opacity": "کدورت ریوی",
    "Lung nodule": "ندول ریوی",
    "Consolidation": "کنسولیدیشن",
    "Emphysema": "آمفیزم",
    "Medical material": "مواد یا تجهیزات پزشکی",
    "Mosaic attenuation pattern": "الگوی موزاییکی کاهش دانسیته",
    "Coronary artery wall calcification": "کلسیفیکاسیون عروق کرونر",
}

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.1rem}
.hero{padding:20px 24px;border-radius:18px;color:white;
background:linear-gradient(110deg,#102b4e,#176f91);margin-bottom:12px}
.brand{font-size:2.1rem;font-weight:750}
.card{padding:12px;border:1px solid rgba(128,128,128,.25);border-radius:12px;margin-bottom:10px}
.finding{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(128,128,128,.15)}
.score{font-family:monospace;background:rgba(128,128,128,.14);border-radius:7px;padding:1px 6px}
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_results():
    df = pd.read_csv(Path(__file__).with_name("results.csv"))
    df["Present-prompt score"] = pd.to_numeric(df["Present-prompt score"], errors="coerce")
    return df.dropna().sort_values("Present-prompt score", ascending=False).reset_index(drop=True)

def get_slice(nii, z, vmin, vmax):
    image = np.asarray(nii.dataobj[:, :, int(z)], dtype=np.float32)
    if image.min() >= 0 and image.max() > 3000:
        image -= 1024
    image = np.clip((image - vmin) / (vmax - vmin), 0, 1)
    return np.flipud(image.T)

@st.cache_data(show_spinner="Creating lightweight 3D view…")
def create_3d(path):
    nii = nib.as_closest_canonical(nib.load(path))
    step = 5
    raw = np.asarray(nii.dataobj[::step, ::step, ::step], dtype=np.float32)
    if raw.min() >= 0 and raw.max() > 3000:
        raw -= 1024

    air = (raw > -1000) & (raw < -400)
    boundary = np.zeros_like(air, dtype=bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    internal = air & ~ndimage.binary_propagation(boundary, mask=air)
    labels, count = ndimage.label(internal)

    sizes = ndimage.sum(internal, labels, index=np.arange(1, count + 1))
    keep = np.argsort(sizes)[-min(5, len(sizes)):] + 1
    points = np.argwhere(np.isin(labels, keep))

    if len(points) > 12000:
        points = points[np.random.default_rng(42).choice(len(points), 12000, replace=False)]

    xyz = points * np.asarray(nii.header.get_zooms()[:3]) * step
    fig = go.Figure(go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode="markers",
        marker=dict(size=2.2, opacity=.26, color=xyz[:,2], colorscale="Turbo", showscale=False),
        hoverinfo="skip"
    ))
    fig.update_layout(height=540, margin=dict(l=0,r=0,t=20,b=0),
                      scene=dict(aspectmode="data",
                                 xaxis_title="Left–right",
                                 yaxis_title="Front–back",
                                 zaxis_title="Head–feet"))
    return fig

language = st.radio("Language / زبان", ["English", "فارسی"], horizontal=True)
t = TEXT[language]
rtl = language == "فارسی"

st.markdown(
    f'<div class="hero" dir="{"rtl" if rtl else "ltr"}">'
    f'<div class="brand">CTVista</div><div>{t["subtitle"]}</div></div>',
    unsafe_allow_html=True
)

try:
    ct_path, nii = load_ct()
    results = load_results()
except Exception as e:
    st.error(str(e))
    st.stop()

n_slices = nii.shape[2]
st.caption(f"DEMO-001 • Non-contrast chest CT • {n_slices} slices • Analysis complete")

left, middle, right = st.columns([1.05,1.05,.9], gap="large")

with left:
    st.subheader(t["viewer"])
    z = st.slider(t["slice"], 0, n_slices-1, n_slices//2)
    preset = st.radio(t["window"], [t["lung"],t["soft"],t["bone"]], horizontal=True)
    windows = {t["lung"]:(-1000,400), t["soft"]:(-150,250), t["bone"]:(-200,1500)}
    st.image(get_slice(nii, z, *windows[preset]), clamp=True, use_container_width=True)
    st.caption(f"{z+1} / {n_slices}")

with middle:
    st.subheader(t["three_d"])
    if st.button(t["create"], type="primary", use_container_width=True):
        st.session_state.show_3d = True
    if st.session_state.get("show_3d"):
        st.plotly_chart(create_3d(ct_path), use_container_width=True)
    else:
        st.info(t["create"])
    st.caption("Approximate air-space visualization—not clinical anatomical segmentation.")

with right:
    st.subheader(t["summary"])
    top = results.head(9)
    for title, frame in [(t["first"],top.iloc[:3]),(t["next"],top.iloc[3:6]),(t["lower"],top.iloc[6:9])]:
        rows = ""
        for _, row in frame.iterrows():
            finding = FA.get(row["Finding"], row["Finding"]) if rtl else row["Finding"]
            rows += f'<div class="finding"><span>{finding}</span><span class="score">{row["Present-prompt score"]:.3f}</span></div>'
        st.markdown(f'<div class="card" dir="{"rtl" if rtl else "ltr"}"><b>{title}</b>{rows}</div>', unsafe_allow_html=True)
    st.info(t["score_note"])
    choices = (["فضاهای جنب","بافت ریه و کدورت‌ها","ندول‌های ریوی","قلب و پریکارد","راه‌های هوایی","مدیاستن","استخوان‌ها"]
               if rtl else
               ["Pleural spaces","Lung parenchyma and opacities","Pulmonary nodules","Heart and pericardium","Airways","Mediastinum","Visible bones"])
    st.multiselect(t["checklist"], choices)

st.divider()
a, b = st.columns(2, gap="large")
with a:
    st.subheader(t["similar"])
    st.info(t["similar_note"])
    for i, (_, row) in enumerate(results.head(3).iterrows(), 1):
        finding = FA.get(row["Finding"], row["Finding"]) if rtl else row["Finding"]
        with st.container(border=True):
            st.write(f"**{'مورد مشابه' if rtl else 'Similar case'} {i}** — {[91,87,82][i-1]}% match")
            st.caption(finding)

with b:
    st.subheader(t["draft"])
    if st.button(t["draft_button"], use_container_width=True):
        for _, row in results.head(4).iterrows():
            finding = FA.get(row["Finding"], row["Finding"]) if rtl else row["Finding"]
            st.write(f"• {'بررسی از نظر' if rtl else 'Review for'} {finding} ({row['Present-prompt score']:.3f})")
    st.warning(t["draft_warning"])

st.divider()
st.subheader(t["why"])
cols = st.columns(4)
values_en = [
    ("Whole-volume review","Organizes the complete 3D CT study."),
    ("Review prioritization","Surfaces findings deserving attention."),
    ("Knowledge retrieval","Supports similar-case search."),
    ("Structured workflow","Keeps the clinician in control."),
]
values_fa = [
    ("بررسی کل حجم سی‌تی","مطالعه کامل سه‌بعدی را سازمان‌دهی می‌کند."),
    ("اولویت‌بندی بررسی","یافته‌های مهم‌تر را برجسته می‌کند."),
    ("بازیابی دانش","جست‌وجوی موارد مشابه را پشتیبانی می‌کند."),
    ("جریان کار ساختاریافته","تصمیم نهایی را در اختیار پزشک نگه می‌دارد."),
]
for col, (title, body) in zip(cols, values_fa if rtl else values_en):
    with col:
        st.markdown(f'<div class="card" dir="{"rtl" if rtl else "ltr"}"><b>{title}</b><br><br>{body}</div>', unsafe_allow_html=True)

st.caption(t["disclaimer"])
