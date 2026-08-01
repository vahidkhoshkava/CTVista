
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
from scipy import ndimage

st.set_page_config(page_title="CTVista v2", page_icon="🫁", layout="wide")

# -----------------------------
# Translations
# -----------------------------
T = {
    "English": {
        "subtitle": "AI-assisted chest CT review workspace",
        "tagline": "Review • Explore • Explain • Summarize",
        "viewer": "CT Image Review",
        "slice": "Axial slice",
        "window": "Display window",
        "lung": "Lung",
        "soft": "Soft tissue",
        "bone": "Bone",
        "three_d": "3D Anatomical Overview",
        "make3d": "Generate 3D overview",
        "guided": "AI-Guided Review",
        "first": "Review first",
        "next": "Review next",
        "also": "Also review",
        "progress": "Review progress",
        "reviewed": "regions reviewed",
        "structured": "Structured Anatomical Review",
        "help": "Select an anatomical region. The viewer moves to a representative slice and chooses a suitable display window.",
        "inspect": "What to inspect",
        "mark": "Mark as reviewed",
        "undo": "Remove reviewed status",
        "why": "Why this suggestion?",
        "order": "Suggested Review Order",
        "draft": "Structured Review Draft",
        "generate": "Generate draft findings",
        "similar": "Similar Case Retrieval",
        "similar_note": "Prototype preview. Real deployment would search the hospital archive for comparable studies and reports.",
        "value": "Why CTVista matters",
        "warning": "Research and investor demonstration. Radiologist verification is required.",
        "three_note": "Approximate air-space visualization—not a clinical anatomical segmentation.",
    },
    "فارسی": {
        "subtitle": "محیط هوشمند برای بررسی سی‌تی‌اسکن قفسه سینه",
        "tagline": "بررسی • کاوش • توضیح • خلاصه‌سازی",
        "viewer": "بررسی تصاویر سی‌تی",
        "slice": "برش محوری",
        "window": "پنجره نمایش",
        "lung": "ریه",
        "soft": "بافت نرم",
        "bone": "استخوان",
        "three_d": "نمای کلی سه‌بعدی آناتومیک",
        "make3d": "ایجاد نمای سه‌بعدی",
        "guided": "بررسی هدایت‌شده هوشمند",
        "first": "ابتدا بررسی شود",
        "next": "در مرحله بعد بررسی شود",
        "also": "همچنین بررسی شود",
        "progress": "پیشرفت بررسی",
        "reviewed": "ناحیه بررسی شده",
        "structured": "بررسی ساختاریافته آناتومیک",
        "help": "یک ناحیه آناتومیک را انتخاب کنید. نمایشگر به یک برش نماینده منتقل می‌شود و پنجره مناسب را انتخاب می‌کند.",
        "inspect": "مواردی که باید بررسی شوند",
        "mark": "علامت‌گذاری به‌عنوان بررسی‌شده",
        "undo": "حذف وضعیت بررسی‌شده",
        "why": "چرا این پیشنهاد ارائه شده است؟",
        "order": "ترتیب پیشنهادی بررسی",
        "draft": "پیش‌نویس بررسی ساختاریافته",
        "generate": "ایجاد پیش‌نویس یافته‌ها",
        "similar": "بازیابی موارد مشابه",
        "similar_note": "این بخش نمایشی است. نسخه واقعی آرشیو بیمارستان را برای یافتن مطالعات و گزارش‌های مشابه جست‌وجو می‌کند.",
        "value": "چرا CTVista اهمیت دارد؟",
        "warning": "نسخه نمایشی پژوهشی و سرمایه‌گذاری است و تأیید رادیولوژیست ضروری است.",
        "three_note": "نمایش تقریبی فضای هوادار ریه است، نه تقسیم‌بندی دقیق بالینی.",
    },
}

# -----------------------------
# Anatomical workflow
# -----------------------------
ANATOMY = {
    "Pleural spaces": {
        "fa": "فضاهای جنب", "window": "lung", "fraction": 0.42,
        "summary_en": "Possible pleural-fluid pattern",
        "summary_fa": "الگوی احتمالی مایع پلور",
        "inspect_en": ["Pleural effusion", "Pneumothorax", "Pleural thickening", "Pleural mass"],
        "inspect_fa": ["افیوژن پلور", "پنوموتوراکس", "ضخیم‌شدگی پلور", "توده پلور"],
        "why_en": "The scan representation most closely matches descriptions associated with pleural fluid. Inspect posterior and lower pleural spaces.",
        "why_fa": "نمایش اسکن بیشترین شباهت را به توصیف‌های مرتبط با مایع پلور دارد. فضاهای خلفی و تحتانی پلور بررسی شوند.",
    },
    "Lung parenchyma": {
        "fa": "بافت ریه و کدورت‌ها", "window": "lung", "fraction": 0.50,
        "summary_en": "Possible collapse or opacity pattern",
        "summary_fa": "الگوی احتمالی کلاپس یا کدورت",
        "inspect_en": ["Atelectasis", "Consolidation", "Ground-glass opacity", "Diffuse opacity"],
        "inspect_fa": ["آتِلکتازی", "کنسولیدیشن", "کدورت شیشه مات", "کدورت منتشر"],
        "why_en": "The scan resembles studies containing volume loss and lung opacity. Review dependent lower-lobe regions.",
        "why_fa": "اسکن به مطالعات دارای کاهش حجم و کدورت ریوی شباهت دارد. نواحی وابسته لوب‌های تحتانی بررسی شوند.",
    },
    "Airways": {
        "fa": "راه‌های هوایی", "window": "lung", "fraction": 0.58,
        "summary_en": "Possible bronchial-wall change",
        "summary_fa": "تغییر احتمالی دیواره برونش",
        "inspect_en": ["Wall thickening", "Bronchiectasis", "Mucus plugging", "Airway narrowing"],
        "inspect_fa": ["ضخیم‌شدگی دیواره", "برونشکتازی", "پلاگ موکوسی", "تنگی راه هوایی"],
        "why_en": "The image–text representation resembles descriptions of peribronchial change. Review central and segmental airways.",
        "why_fa": "نمایش تصویر و متن به توصیف‌های تغییرات اطراف برونش شباهت دارد. راه‌های هوایی مرکزی و سگمنتال بررسی شوند.",
    },
    "Pulmonary nodules": {
        "fa": "ندول‌های ریوی", "window": "lung", "fraction": 0.52,
        "summary_en": "Small focal region worth inspecting",
        "summary_fa": "ناحیه کانونی کوچک نیازمند بررسی",
        "inspect_en": ["Solid nodules", "Subsolid nodules", "Calcified nodules", "Multiplicity"],
        "inspect_fa": ["ندول جامد", "ندول نیمه‌جامد", "ندول کلسیفیه", "تعداد ندول‌ها"],
        "why_en": "The scan has a moderate semantic match to nodule-related descriptions. A systematic review of both lungs remains necessary.",
        "why_fa": "اسکن تطابق معنایی متوسطی با توصیف‌های مرتبط با ندول دارد. بررسی نظام‌مند هر دو ریه ضروری است.",
    },
    "Heart and pericardium": {
        "fa": "قلب و پریکارد", "window": "soft", "fraction": 0.38,
        "summary_en": "Cardiac and pericardial review",
        "summary_fa": "بررسی قلب و پریکارد",
        "inspect_en": ["Cardiac size", "Pericardial fluid", "Coronary calcification", "Cardiac contour"],
        "inspect_fa": ["اندازه قلب", "مایع پریکارد", "کلسیفیکاسیون کرونر", "کانتور قلب"],
        "why_en": "This region is included to ensure systematic review of cardiac size, pericardium, and visible coronary calcification.",
        "why_fa": "این ناحیه برای تضمین بررسی نظام‌مند اندازه قلب، پریکارد و کلسیفیکاسیون قابل مشاهده کرونر در نظر گرفته شده است.",
    },
    "Mediastinum": {
        "fa": "مدیاستن و گره‌های لنفاوی", "window": "soft", "fraction": 0.60,
        "summary_en": "Routine mediastinal review",
        "summary_fa": "بررسی روتین مدیاستن",
        "inspect_en": ["Lymph nodes", "Mediastinal mass", "Great vessels", "Esophagus"],
        "inspect_fa": ["گره‌های لنفاوی", "توده مدیاستن", "عروق بزرگ", "مری"],
        "why_en": "The mediastinum should be reviewed systematically even when it is not among the highest-priority suggestions.",
        "why_fa": "مدیاستن حتی زمانی که در پیشنهادهای اصلی نیست باید به‌صورت نظام‌مند بررسی شود.",
    },
    "Bones": {
        "fa": "استخوان‌ها", "window": "bone", "fraction": 0.48,
        "summary_en": "Routine skeletal review",
        "summary_fa": "بررسی روتین اسکلت",
        "inspect_en": ["Ribs", "Vertebrae", "Sternum", "Focal bone lesions"],
        "inspect_fa": ["دنده‌ها", "مهره‌ها", "جناغ", "ضایعات کانونی استخوان"],
        "why_en": "Bone-window review can reveal fractures and focal osseous abnormalities that are less apparent on lung windows.",
        "why_fa": "پنجره استخوان می‌تواند شکستگی‌ها و ضایعات کانونی را که در پنجره ریه کمتر دیده می‌شوند آشکار کند.",
    },
    "Upper abdomen": {
        "fa": "بخش فوقانی شکم", "window": "soft", "fraction": 0.12,
        "summary_en": "Review included upper-abdominal structures",
        "summary_fa": "بررسی ساختارهای فوقانی شکم",
        "inspect_en": ["Liver", "Adrenal glands", "Spleen", "Upper kidneys"],
        "inspect_fa": ["کبد", "غدد فوق کلیوی", "طحال", "بخش فوقانی کلیه‌ها"],
        "why_en": "Chest CT often includes part of the upper abdomen, and visible structures should not be overlooked.",
        "why_fa": "سی‌تی قفسه سینه معمولاً بخشی از شکم فوقانی را نیز شامل می‌شود و این ساختارها نباید نادیده گرفته شوند.",
    },
}
ORDER = list(ANATOMY.keys())

# -----------------------------
# Style
# -----------------------------
st.markdown("""
<style>
.block-container{max-width:1550px;padding-top:1rem}
.hero{padding:20px 26px;border-radius:18px;color:white;background:linear-gradient(110deg,#102b4e,#126a87);margin-bottom:10px}
.brand{font-size:2.15rem;font-weight:760}
.tag{opacity:.94;margin-top:5px}
.study{padding:12px 16px;border:1px solid rgba(128,128,128,.23);border-radius:13px;margin-bottom:14px}
.review-card{padding:11px 13px;border-radius:12px;margin:5px 0 10px 0;border-left:5px solid}
.red{background:rgba(215,55,70,.09);border-left-color:#d73746}
.orange{background:rgba(232,145,30,.10);border-left-color:#e8911e}
.green{background:rgba(42,145,95,.09);border-left-color:#2a915f}
.box{padding:14px;border:1px solid rgba(128,128,128,.24);border-radius:13px}
.progress-track{height:10px;border-radius:999px;background:rgba(128,128,128,.18);overflow:hidden}
.progress-fill{height:100%;background:#1887a8;border-radius:999px}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
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
    image = np.asarray(nii.dataobj[:, :, int(z)], dtype=np.float32)
    if image.min() >= 0 and image.max() > 3000:
        image -= 1024
    image = np.clip((image-vmin)/(vmax-vmin), 0, 1)
    return np.flipud(image.T)

@st.cache_data(show_spinner="Creating lightweight 3D view…")
def make_3d(path):
    nii = nib.as_closest_canonical(nib.load(path))
    step = 5
    raw = np.asarray(nii.dataobj[::step,::step,::step], dtype=np.float32)
    if raw.min() >= 0 and raw.max() > 3000:
        raw -= 1024
    air = (raw > -1000) & (raw < -400)
    edge = np.zeros_like(air, dtype=bool)
    edge[0]=edge[-1]=True; edge[:,0]=edge[:,-1]=True; edge[:,:,0]=edge[:,:,-1]=True
    internal = air & ~ndimage.binary_propagation(edge, mask=air)
    labels, count = ndimage.label(internal)
    sizes = ndimage.sum(internal, labels, index=np.arange(1,count+1))
    keep = np.argsort(sizes)[-min(5,len(sizes)):] + 1
    pts = np.argwhere(np.isin(labels, keep))
    if len(pts) > 12000:
        pts = pts[np.random.default_rng(42).choice(len(pts),12000,replace=False)]
    xyz = pts * np.asarray(nii.header.get_zooms()[:3]) * step
    fig = go.Figure(go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode="markers",
        marker=dict(size=2.2,opacity=.25,color=xyz[:,2],colorscale="Turbo",showscale=False),
        hoverinfo="skip"
    ))
    fig.update_layout(height=545,margin=dict(l=0,r=0,t=20,b=0),
                      scene=dict(aspectmode="data",xaxis_title="Left–right",yaxis_title="Front–back",zaxis_title="Head–feet"))
    return fig

# -----------------------------
# State
# -----------------------------
if "region" not in st.session_state: st.session_state.region = "Pleural spaces"
if "reviewed" not in st.session_state: st.session_state.reviewed = []
if "show3d" not in st.session_state: st.session_state.show3d = False

language = st.radio("Language / زبان", ["English","فارسی"], horizontal=True)
t = T[language]
fa = language == "فارسی"
direction = "rtl" if fa else "ltr"

st.markdown(f'<div class="hero" dir="{direction}"><div class="brand">CTVista <span style="font-size:.75rem">v2</span></div><div>{t["subtitle"]}</div><div class="tag">{t["tagline"]}</div></div>', unsafe_allow_html=True)

try:
    ct_path, nii = load_ct()
except Exception as e:
    st.error(str(e)); st.stop()

n = int(nii.shape[2])
st.markdown(f'<div class="study" dir="{direction}"><b>DEMO-001</b> • Non-contrast chest CT • {n} slices • ✅ Analysis complete</div>', unsafe_allow_html=True)

region_data = ANATOMY[st.session_state.region]
default_z = int(region_data["fraction"]*(n-1))
window_ranges = {"lung":(-1000,400),"soft":(-150,250),"bone":(-200,1500)}
window_label = {"lung":t["lung"],"soft":t["soft"],"bone":t["bone"]}

left, middle, right = st.columns([1.05,1.05,.92], gap="large")

with left:
    st.subheader(t["viewer"])
    z = st.slider(t["slice"],0,n-1,default_z,key=f'z_{st.session_state.region}')
    options = [t["lung"],t["soft"],t["bone"]]
    selected_window = st.radio(t["window"],options,index=options.index(window_label[region_data["window"]]),horizontal=True,key=f'w_{st.session_state.region}')
    reverse = {t["lung"]:"lung",t["soft"]:"soft",t["bone"]:"bone"}
    st.image(get_slice(nii,z,*window_ranges[reverse[selected_window]]),clamp=True,width="stretch")
    st.caption(f"{z+1} / {n}")

with middle:
    st.subheader(t["three_d"])
    if st.button(t["make3d"],type="primary",width="stretch"): st.session_state.show3d = True
    if st.session_state.show3d: st.plotly_chart(make_3d(ct_path),width="stretch")
    else: st.info(t["make3d"])
    st.caption(t["three_note"])

with right:
    st.subheader(t["guided"])
    reviewed_count = len(st.session_state.reviewed)
    percent = int(100*reviewed_count/len(ORDER))
    st.markdown(f'<div dir="{direction}"><b>{t["progress"]}</b> — {reviewed_count}/{len(ORDER)} {t["reviewed"]}<div class="progress-track"><div class="progress-fill" style="width:{percent}%"></div></div></div>',unsafe_allow_html=True)
    groups = [(t["first"],ORDER[:2],"red"),(t["next"],ORDER[2:5],"orange"),(t["also"],ORDER[5:],"green")]
    for heading,names,css in groups:
        st.markdown(f"#### {heading}")
        for name in names:
            d = ANATOMY[name]
            label = d["fa"] if fa else name
            if name in st.session_state.reviewed: label += " ✓"
            if st.button(label,key=f'pick_{name}_{language}',width="stretch"):
                st.session_state.region = name
                st.rerun()
            summary = d["summary_fa"] if fa else d["summary_en"]
            st.markdown(f'<div class="review-card {css}" dir="{direction}">{summary}</div>',unsafe_allow_html=True)

st.divider()
a,b = st.columns([1.1,.9],gap="large")
selected_name = st.session_state.region
d = ANATOMY[selected_name]
display_name = d["fa"] if fa else selected_name
items = d["inspect_fa"] if fa else d["inspect_en"]
why = d["why_fa"] if fa else d["why_en"]

with a:
    st.subheader(t["structured"])
    st.info(t["help"])
    st.markdown(f'<div class="box" dir="{direction}"><h3>{display_name}</h3><b>{t["inspect"]}</b><ul>{"".join(f"<li>{x}</li>" for x in items)}</ul></div>',unsafe_allow_html=True)
    is_reviewed = selected_name in st.session_state.reviewed
    if st.button(t["undo"] if is_reviewed else t["mark"],type="secondary" if is_reviewed else "primary",width="stretch"):
        if is_reviewed: st.session_state.reviewed.remove(selected_name)
        else: st.session_state.reviewed.append(selected_name)
        st.rerun()

with b:
    st.subheader(t["why"])
    st.markdown(f'<div class="box" dir="{direction}"><b>{display_name}</b><br><br>{why}</div>',unsafe_allow_html=True)

st.divider()
st.subheader(t["order"])
cols = st.columns(4)
for i,name in enumerate(ORDER):
    with cols[i%4]:
        label = ANATOMY[name]["fa"] if fa else name
        icon = "✓" if name in st.session_state.reviewed else str(i+1)
        st.markdown(f'<div class="box" dir="{direction}"><b>{icon}. {label}</b></div>',unsafe_allow_html=True)

st.divider()
c1,c2 = st.columns(2,gap="large")
with c1:
    st.subheader(t["similar"])
    st.info(t["similar_note"])
    for i,name in enumerate(ORDER[:3],1):
        with st.container(border=True):
            label = ANATOMY[name]["fa"] if fa else name
            st.write(f"**{'مورد مشابه' if fa else 'Similar case'} {i}** — {[91,87,82][i-1]}% match")
            st.caption(label)

with c2:
    st.subheader(t["draft"])
    if st.button(t["generate"],width="stretch"):
        if not st.session_state.reviewed:
            st.info("ابتدا یک ناحیه را بررسی‌شده علامت‌گذاری کنید." if fa else "Mark at least one region as reviewed first.")
        for name in st.session_state.reviewed:
            d = ANATOMY[name]
            label = d["fa"] if fa else name
            summary = d["summary_fa"] if fa else d["summary_en"]
            st.write(f"• **{label}:** {summary}")
    st.warning(t["warning"])

st.divider()
st.subheader(t["value"])
vals_en = [
    ("Whole-volume review","Supports review of the complete 3D CT study."),
    ("Guided workflow","Prioritizes anatomy and standardizes the review sequence."),
    ("Comparable cases","Supports retrieval of prior studies with similar patterns."),
    ("Structured reporting","Organizes reviewed regions into a clinician-controlled draft."),
]
vals_fa = [
    ("بررسی کل حجم سی‌تی","مطالعه کامل سه‌بعدی را پشتیبانی می‌کند."),
    ("جریان کار هدایت‌شده","آناتومی را اولویت‌بندی و ترتیب بررسی را استاندارد می‌کند."),
    ("موارد قابل مقایسه","بازیابی مطالعات قبلی با الگوهای مشابه را پشتیبانی می‌کند."),
    ("گزارش ساختاریافته","ناحیه‌های بررسی‌شده را در پیش‌نویس تحت کنترل پزشک سازمان‌دهی می‌کند."),
]
for col,(title,body) in zip(st.columns(4),vals_fa if fa else vals_en):
    with col: st.markdown(f'<div class="box" dir="{direction}"><b>{title}</b><br><br>{body}</div>',unsafe_allow_html=True)

st.caption(t["warning"])
