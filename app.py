import os

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
from scipy import ndimage

st.set_page_config(page_title="CTVista v3", page_icon="🫁", layout="wide")

TEXT = {
    "English": {
        "subtitle": "AI-assisted chest CT review workspace",
        "tagline": "Review • Explain • Compare • Summarize",
        "viewer": "CT Image Review",
        "slice": "Slice",
        "window": "Display window",
        "lung": "Lung",
        "soft": "Soft tissue",
        "bone": "Bone",
        "guided": "AI-Guided Review",
        "progress": "Review progress",
        "region": "Region",
        "note": "AI note",
        "status": "Status",
        "open": "Open",
        "reviewed": "Reviewed",
        "selected": "Selected Region",
        "inspect": "What to inspect",
        "why": "Why this suggestion?",
        "mark": "Mark as reviewed",
        "undo": "Remove reviewed status",
        "draft": "Structured Review Draft",
        "generate": "Generate draft findings",
        "draft_empty": "Review at least one region first.",
        "reference": "Reference Cases",
        "reference_note": "Prototype preview. A clinical version would retrieve comparable prior studies and reports from the hospital archive.",
        "three_d": "Optional 3D Anatomical Overview",
        "make3d": "Generate lightweight 3D overview",
        "three_note": "Approximate air-space visualization—not clinical segmentation.",
        "value": "Product Value",
        "warning": "Research and investor demonstration. Radiologist verification is required.",
    },
    "فارسی": {
        "subtitle": "محیط هوشمند برای بررسی سی‌تی‌اسکن قفسه سینه",
        "tagline": "بررسی • توضیح • مقایسه • خلاصه‌سازی",
        "viewer": "بررسی تصاویر سی‌تی",
        "slice": "برش",
        "window": "پنجره نمایش",
        "lung": "ریه",
        "soft": "بافت نرم",
        "bone": "استخوان",
        "guided": "بررسی هدایت‌شده هوشمند",
        "progress": "پیشرفت بررسی",
        "region": "ناحیه",
        "note": "یادداشت هوشمند",
        "status": "وضعیت",
        "open": "باز کردن",
        "reviewed": "بررسی شد",
        "selected": "ناحیه انتخاب‌شده",
        "inspect": "مواردی که باید بررسی شوند",
        "why": "چرا این پیشنهاد ارائه شده است؟",
        "mark": "علامت‌گذاری به‌عنوان بررسی‌شده",
        "undo": "حذف وضعیت بررسی‌شده",
        "draft": "پیش‌نویس بررسی ساختاریافته",
        "generate": "ایجاد پیش‌نویس یافته‌ها",
        "draft_empty": "ابتدا حداقل یک ناحیه را بررسی کنید.",
        "reference": "موارد مرجع",
        "reference_note": "این بخش نمایشی است. نسخه بالینی مطالعات و گزارش‌های مشابه قبلی را از آرشیو بیمارستان بازیابی می‌کند.",
        "three_d": "نمای کلی سه‌بعدی اختیاری",
        "make3d": "ایجاد نمای سه‌بعدی سبک",
        "three_note": "نمایش تقریبی فضای هوادار ریه است، نه تقسیم‌بندی بالینی.",
        "value": "ارزش محصول",
        "warning": "نسخه نمایشی پژوهشی و سرمایه‌گذاری است و تأیید رادیولوژیست ضروری است.",
    },
}

ANATOMY = {
    "Pleural spaces": dict(fa="فضاهای جنب", priority="high", window="lung", fraction=.42,
        note_en="Possible pleural-fluid pattern", note_fa="الگوی احتمالی مایع پلور",
        inspect_en=["Pleural effusion", "Pneumothorax", "Pleural thickening", "Pleural mass"],
        inspect_fa=["افیوژن پلور", "پنوموتوراکس", "ضخیم‌شدگی پلور", "توده پلور"],
        why_en="The scan representation is most similar to descriptions associated with pleural fluid. Inspect posterior and lower pleural spaces.",
        why_fa="نمایش اسکن بیشترین شباهت را به توصیف‌های مرتبط با مایع پلور دارد. فضاهای خلفی و تحتانی پلور بررسی شوند."),
    "Lung parenchyma": dict(fa="بافت ریه و کدورت‌ها", priority="high", window="lung", fraction=.50,
        note_en="Possible collapse or opacity pattern", note_fa="الگوی احتمالی کلاپس یا کدورت",
        inspect_en=["Atelectasis", "Consolidation", "Ground-glass opacity", "Diffuse opacity"],
        inspect_fa=["آتِلکتازی", "کنسولیدیشن", "کدورت شیشه مات", "کدورت منتشر"],
        why_en="The scan resembles studies containing volume loss and lung opacity. Review dependent lower-lobe regions.",
        why_fa="اسکن به مطالعات دارای کاهش حجم و کدورت ریوی شباهت دارد. نواحی وابسته لوب‌های تحتانی بررسی شوند."),
    "Airways": dict(fa="راه‌های هوایی", priority="medium", window="lung", fraction=.58,
        note_en="Possible bronchial-wall change", note_fa="تغییر احتمالی دیواره برونش",
        inspect_en=["Wall thickening", "Bronchiectasis", "Mucus plugging", "Airway narrowing"],
        inspect_fa=["ضخیم‌شدگی دیواره", "برونشکتازی", "پلاگ موکوسی", "تنگی راه هوایی"],
        why_en="The image–text representation resembles descriptions of peribronchial change. Review central and segmental airways.",
        why_fa="نمایش تصویر و متن به توصیف‌های تغییرات اطراف برونش شباهت دارد. راه‌های هوایی مرکزی و سگمنتال بررسی شوند."),
    "Pulmonary nodules": dict(fa="ندول‌های ریوی", priority="medium", window="lung", fraction=.52,
        note_en="Small focal region worth inspecting", note_fa="ناحیه کانونی کوچک نیازمند بررسی",
        inspect_en=["Solid nodules", "Subsolid nodules", "Calcified nodules", "Multiplicity"],
        inspect_fa=["ندول جامد", "ندول نیمه‌جامد", "ندول کلسیفیه", "تعداد ندول‌ها"],
        why_en="The scan has a moderate semantic match to nodule-related descriptions. A systematic review of both lungs remains necessary.",
        why_fa="اسکن تطابق معنایی متوسطی با توصیف‌های مرتبط با ندول دارد. بررسی نظام‌مند هر دو ریه ضروری است."),
    "Heart and pericardium": dict(fa="قلب و پریکارد", priority="medium", window="soft", fraction=.38,
        note_en="Cardiac and pericardial review", note_fa="بررسی قلب و پریکارد",
        inspect_en=["Cardiac size", "Pericardial fluid", "Coronary calcification", "Cardiac contour"],
        inspect_fa=["اندازه قلب", "مایع پریکارد", "کلسیفیکاسیون کرونر", "کانتور قلب"],
        why_en="This region ensures systematic review of cardiac size, pericardium, and visible coronary calcification.",
        why_fa="این ناحیه برای تضمین بررسی نظام‌مند اندازه قلب، پریکارد و کلسیفیکاسیون قابل مشاهده کرونر در نظر گرفته شده است."),
    "Mediastinum": dict(fa="مدیاستن و گره‌های لنفاوی", priority="routine", window="soft", fraction=.60,
        note_en="Routine mediastinal review", note_fa="بررسی روتین مدیاستن",
        inspect_en=["Lymph nodes", "Mediastinal mass", "Great vessels", "Esophagus"],
        inspect_fa=["گره‌های لنفاوی", "توده مدیاستن", "عروق بزرگ", "مری"],
        why_en="The mediastinum should be reviewed systematically even when it is not among the highest-priority suggestions.",
        why_fa="مدیاستن حتی زمانی که در پیشنهادهای اصلی نیست باید به‌صورت نظام‌مند بررسی شود."),
    "Bones": dict(fa="استخوان‌ها", priority="routine", window="bone", fraction=.48,
        note_en="Routine skeletal review", note_fa="بررسی روتین اسکلت",
        inspect_en=["Ribs", "Vertebrae", "Sternum", "Focal bone lesions"],
        inspect_fa=["دنده‌ها", "مهره‌ها", "جناغ", "ضایعات کانونی استخوان"],
        why_en="Bone-window review can reveal fractures and focal osseous abnormalities that are less apparent on lung windows.",
        why_fa="پنجره استخوان می‌تواند شکستگی‌ها و ضایعات کانونی را که در پنجره ریه کمتر دیده می‌شوند آشکار کند."),
    "Upper abdomen": dict(fa="بخش فوقانی شکم", priority="routine", window="soft", fraction=.12,
        note_en="Review included upper-abdominal structures", note_fa="بررسی ساختارهای فوقانی شکم",
        inspect_en=["Liver", "Adrenal glands", "Spleen", "Upper kidneys"],
        inspect_fa=["کبد", "غدد فوق کلیوی", "طحال", "بخش فوقانی کلیه‌ها"],
        why_en="Chest CT often includes part of the upper abdomen, and visible structures should not be overlooked.",
        why_fa="سی‌تی قفسه سینه معمولاً بخشی از شکم فوقانی را نیز شامل می‌شود و این ساختارها نباید نادیده گرفته شوند."),
}
ORDER = list(ANATOMY)

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:.9rem;padding-bottom:2rem}
.hero{padding:18px 24px;border-radius:18px;color:white;background:linear-gradient(110deg,#102b4e,#126a87);margin-bottom:10px}
.brand{font-size:2.05rem;font-weight:760}.tagline{opacity:.94;margin-top:4px}
.study,.box{padding:12px 14px;border:1px solid rgba(128,128,128,.23);border-radius:13px}
.study{margin-bottom:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%}
.high{background:#d73746}.medium{background:#e8911e}.routine{background:#2a915f}
.table{border:1px solid rgba(128,128,128,.23);border-radius:13px;overflow:hidden}
.row{display:grid;grid-template-columns:28px 1.1fr 1.7fr 86px;gap:8px;align-items:center;padding:9px 10px;border-bottom:1px solid rgba(128,128,128,.15)}
.row:last-child{border-bottom:none}.head{font-weight:700;background:rgba(128,128,128,.08)}
.chip{display:inline-block;padding:3px 8px;border-radius:999px;background:rgba(128,128,128,.12);font-size:.82rem;text-align:center}
.track{height:10px;border-radius:999px;background:rgba(128,128,128,.18);overflow:hidden}.fill{height:100%;background:#1887a8;border-radius:999px}
@media(max-width:768px){.row{grid-template-columns:22px 1fr}.row .note,.row .status{grid-column:2}div[data-testid="stSlider"] [data-testid="stThumbValue"]{display:none!important}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Downloading demonstration CT…")
def load_ct():
    token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
    if not token:
        raise RuntimeError("HF_TOKEN is missing in Streamlit Secrets.")
    path = hf_hub_download(repo_id="ibrahimhamamci/CT-RATE", repo_type="dataset",
        filename="dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz", token=token)
    return path, nib.as_closest_canonical(nib.load(path))

def get_slice(nii,z,vmin,vmax):
    image=np.asarray(nii.dataobj[:,:,int(z)],dtype=np.float32)
    if image.min()>=0 and image.max()>3000:image-=1024
    return np.flipud(np.clip((image-vmin)/(vmax-vmin),0,1).T)

@st.cache_data(show_spinner="Creating lightweight 3D view…")
def create_3d(path):
    nii=nib.as_closest_canonical(nib.load(path)); step=5
    raw=np.asarray(nii.dataobj[::step,::step,::step],dtype=np.float32)
    if raw.min()>=0 and raw.max()>3000:raw-=1024
    air=(raw>-1000)&(raw<-400); edge=np.zeros_like(air,dtype=bool)
    edge[0]=edge[-1]=True;edge[:,0]=edge[:,-1]=True;edge[:,:,0]=edge[:,:,-1]=True
    internal=air & ~ndimage.binary_propagation(edge,mask=air)
    labels,count=ndimage.label(internal)
    sizes=ndimage.sum(internal,labels,index=np.arange(1,count+1));keep=np.argsort(sizes)[-min(5,len(sizes)):]+1
    pts=np.argwhere(np.isin(labels,keep))
    if len(pts)>10000:pts=pts[np.random.default_rng(42).choice(len(pts),10000,replace=False)]
    xyz=pts*np.asarray(nii.header.get_zooms()[:3])*step
    fig=go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode="markers",
        marker=dict(size=2.1,opacity=.24,color=xyz[:,2],colorscale="Turbo",showscale=False),hoverinfo="skip"))
    fig.update_layout(height=510,margin=dict(l=0,r=0,t=15,b=0),scene=dict(aspectmode="data"))
    return fig

if "region" not in st.session_state:st.session_state.region="Pleural spaces"
if "reviewed" not in st.session_state:st.session_state.reviewed=[]
if "show3d" not in st.session_state:st.session_state.show3d=False

language=st.radio("Language / زبان",["English","فارسی"],horizontal=True)
t=TEXT[language];fa=language=="فارسی";direction="rtl" if fa else "ltr"
st.markdown(f'<div class="hero" dir="{direction}"><div class="brand">CTVista <span style="font-size:.72rem">v3</span></div><div>{t["subtitle"]}</div><div class="tagline">{t["tagline"]}</div></div>',unsafe_allow_html=True)
try:ct_path,nii=load_ct()
except Exception as e:st.error(str(e));st.stop()
n=int(nii.shape[2])
st.markdown(f'<div class="study" dir="{direction}"><b>DEMO-001</b> • Non-contrast chest CT • {n} slices • ✅ Analysis complete</div>',unsafe_allow_html=True)

selected=ANATOMY[st.session_state.region];default_z=int(selected["fraction"]*(n-1))
windows={"lung":(-1000,400),"soft":(-150,250),"bone":(-200,1500)}
window_labels={"lung":t["lung"],"soft":t["soft"],"bone":t["bone"]}

left,right=st.columns([1.0,1.2],gap="large")
with left:
    st.subheader(t["viewer"])
    st.markdown(f'**{t["slice"]} {default_z+1} / {n}**')
    z=st.slider(t["slice"],0,n-1,default_z,label_visibility="collapsed",key=f'z_{st.session_state.region}')
    opts=[t["lung"],t["soft"],t["bone"]]
    chosen=st.radio(t["window"],opts,index=opts.index(window_labels[selected["window"]]),horizontal=True,key=f'w_{st.session_state.region}')
    rev={t["lung"]:"lung",t["soft"]:"soft",t["bone"]:"bone"}
    st.image(get_slice(nii,z,*windows[rev[chosen]]),clamp=True,width="stretch")
    st.caption(f'{t["slice"]} {z+1} / {n}')

with right:
    st.subheader(t["guided"])
    done=len(st.session_state.reviewed);pct=int(100*done/len(ORDER))
    st.markdown(f'<div dir="{direction}"><b>{t["progress"]}</b> — {done}/{len(ORDER)} {t["reviewed"]}<div class="track"><div class="fill" style="width:{pct}%"></div></div></div>',unsafe_allow_html=True)
    rows=''
    for name in ORDER:
        d=ANATOMY[name];label=d["fa"] if fa else name;note=d["note_fa"] if fa else d["note_en"]
        status=t["reviewed"] if name in st.session_state.reviewed else t["open"]
        rows+=f'<div class="row"><div><span class="dot {d["priority"]}"></span></div><div><b>{label}</b></div><div class="note">{note}</div><div class="status"><span class="chip">{status}</span></div></div>'
    st.markdown(f'<div class="table" dir="{direction}"><div class="row head"><div></div><div>{t["region"]}</div><div class="note">{t["note"]}</div><div class="status">{t["status"]}</div></div>{rows}</div>',unsafe_allow_html=True)
    choices=[ANATOMY[x]["fa"] if fa else x for x in ORDER]
    current=ANATOMY[st.session_state.region]["fa"] if fa else st.session_state.region
    picked=st.selectbox(t["selected"],choices,index=choices.index(current))
    lookup={(ANATOMY[x]["fa"] if fa else x):x for x in ORDER}
    if lookup[picked]!=st.session_state.region:st.session_state.region=lookup[picked];st.rerun()

st.divider();d=ANATOMY[st.session_state.region];label=d["fa"] if fa else st.session_state.region
items=d["inspect_fa"] if fa else d["inspect_en"];why=d["why_fa"] if fa else d["why_en"]
a,b,c=st.columns(3,gap="large")
with a:
    st.subheader(t["selected"]);st.markdown(f'<div class="box" dir="{direction}"><h3>{label}</h3><b>{t["inspect"]}</b><ul>{"".join(f"<li>{x}</li>" for x in items)}</ul></div>',unsafe_allow_html=True)
    checked=st.session_state.region in st.session_state.reviewed
    if st.button(t["undo"] if checked else t["mark"],type="secondary" if checked else "primary",width="stretch"):
        if checked:st.session_state.reviewed.remove(st.session_state.region)
        else:st.session_state.reviewed.append(st.session_state.region)
        st.rerun()
with b:
    st.subheader(t["why"]);st.markdown(f'<div class="box" dir="{direction}"><b>{label}</b><br><br>{why}</div>',unsafe_allow_html=True)
with c:
    st.subheader(t["draft"])
    if st.button(t["generate"],width="stretch"):
        if not st.session_state.reviewed:st.info(t["draft_empty"])
        for name in st.session_state.reviewed:
            x=ANATOMY[name];nm=x["fa"] if fa else name;note=x["note_fa"] if fa else x["note_en"]
            st.write(f'• **{nm}:** {note}')
    st.warning(t["warning"])

st.divider();st.subheader(t["reference"]);st.info(t["reference_note"])
for i,(col,name) in enumerate(zip(st.columns(3),ORDER[:3])):
    with col:
        x=ANATOMY[name];nm=x["fa"] if fa else name;note=x["note_fa"] if fa else x["note_en"]
        with st.container(border=True):st.markdown(f'**{"مورد مرجع" if fa else "Reference case"} {i+1}** — {[91,87,82][i]}% match');st.caption(nm);st.write(note)

st.divider();st.subheader(t["value"])
vals_en=[("Guided review","Directs attention to important anatomical regions."),("Structured workflow","Tracks completion and reduces missed steps."),("Reference cases","Connects the study to comparable prior cases."),("Draft support","Turns reviewed regions into a structured starting point.")]
vals_fa=[("بررسی هدایت‌شده","توجه را به ناحیه‌های مهم آناتومیک هدایت می‌کند."),("جریان کار ساختاریافته","مراحل تکمیل‌شده را ثبت می‌کند."),("موارد مرجع","مطالعه را به موارد مشابه قبلی متصل می‌کند."),("پشتیبانی از پیش‌نویس","ناحیه‌های بررسی‌شده را به پیش‌نویس تبدیل می‌کند.")]
for col,(title,body) in zip(st.columns(4),vals_fa if fa else vals_en):
    with col:st.markdown(f'<div class="box" dir="{direction}"><b>{title}</b><br><br>{body}</div>',unsafe_allow_html=True)

st.divider()
with st.expander(t["three_d"]):
    if st.button(t["make3d"],type="primary",width="stretch"):st.session_state.show3d=True
    if st.session_state.show3d:st.plotly_chart(create_3d(ct_path),width="stretch")
    st.caption(t["three_note"])
st.caption(t["warning"])
