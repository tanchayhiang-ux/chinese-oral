import streamlit as st
import json
import random
import datetime
import io
import base64
import os
import time
import pandas as pd
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="华文口试练习 | SG Chinese Oral",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load question bank ────────────────────────────────────────────────────────
from question_bank import QUESTIONS
from scoring import score_response, get_feedback, get_badge

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 50%, #c0392b 100%);
    color: white; padding: 20px 30px; border-radius: 12px;
    text-align: center; margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(192,57,43,0.4);
}
.main-header h1 { font-size: 2rem; margin: 0; }
.main-header p  { font-size: 1rem; margin: 4px 0 0; opacity: 0.9; }

.score-card {
    background: white; border-radius: 12px; padding: 16px;
    text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-top: 4px solid #c0392b;
}
.score-number { font-size: 2.5rem; font-weight: 700; color: #c0392b; }
.score-label  { font-size: 0.85rem; color: #666; margin-top: 4px; }

.question-box {
    background: #fff9f9; border-left: 5px solid #e74c3c;
    padding: 16px 20px; border-radius: 8px; margin: 12px 0;
    font-size: 1.1rem;
}
.feedback-good {
    background: #eafaf1; border-left: 5px solid #27ae60;
    padding: 14px 18px; border-radius: 8px; margin: 10px 0;
}
.feedback-improve {
    background: #fef9e7; border-left: 5px solid #f39c12;
    padding: 14px 18px; border-radius: 8px; margin: 10px 0;
}
.badge { font-size: 2rem; margin: 4px; display: inline-block; }
.leaderboard-row { padding: 8px 12px; border-radius: 8px; margin: 4px 0; }
.gold   { background: #fff8dc; border-left: 4px solid #ffd700; }
.silver { background: #f5f5f5; border-left: 4px solid #c0c0c0; }
.bronze { background: #fdf0e8; border-left: 4px solid #cd7f32; }

.vocab-chip {
    display: inline-block; background: #eaf2ff; color: #2471a3;
    border-radius: 20px; padding: 3px 10px; margin: 3px;
    font-size: 0.85rem; border: 1px solid #aed6f1;
}
.rubric-table th { background: #c0392b; color: white; padding: 8px 12px; }
.rubric-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.rubric-table tr:nth-child(even) { background: #fdf2f2; }

.stButton > button {
    border-radius: 8px; font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "home",
        "student_name": "",
        "class_name": "",
        "current_q": None,
        "scores": [],          # list of dicts per attempt
        "session_scores": [],  # current session
        "followup_idx": 0,
        "answered": False,
        "transcript": "",
        "ai_feedback": None,
        "leaderboard": [],
        "teacher_mode": False,
        "recording": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
def nav(page):
    st.session_state.page = page
    st.session_state.answered = False
    st.session_state.ai_feedback = None
    st.session_state.transcript = ""
    st.session_state.followup_idx = 0

def pick_question(level=None, topic=None):
    pool = QUESTIONS
    if level:
        pool = [q for q in pool if q["level"] == level]
    if topic:
        pool = [q for q in pool if q["topic"] == topic]
    if not pool:
        pool = QUESTIONS
    q = random.choice(pool)
    st.session_state.current_q = q
    st.session_state.answered = False
    st.session_state.ai_feedback = None
    st.session_state.transcript = ""
    st.session_state.followup_idx = 0

def add_to_leaderboard(name, cls, total, badges):
    entry = {
        "name": name, "class": cls, "total": total,
        "badges": badges,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.leaderboard.append(entry)
    st.session_state.leaderboard.sort(key=lambda x: x["total"], reverse=True)
    st.session_state.leaderboard = st.session_state.leaderboard[:50]

def scores_to_df():
    if not st.session_state.scores:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.scores)

# ── Audio recorder component ──────────────────────────────────────────────────
def audio_recorder_ui():
    """
    Uses streamlit-audio-recorder if available, else shows file uploader fallback.
    Returns transcript text or None.
    """
    transcript = None

    st.markdown("#### 🎤 录音 / Record Your Answer")

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            from streamlit_mic_recorder import mic_recorder
            audio = mic_recorder(
                start_prompt="🔴 开始录音 Start",
                stop_prompt="⏹ 停止录音 Stop",
                just_once=True,
                key="mic",
            )
            if audio and audio.get("bytes"):
                st.audio(audio["bytes"], format="audio/wav")
                # Attempt Whisper transcription
                transcript = transcribe_audio(audio["bytes"])
        except ImportError:
            st.info("💡 Tip: Install `streamlit-mic-recorder` for in-browser recording.\n\nFor now, type your answer or upload an audio file below.")

    with col2:
        uploaded = st.file_uploader("或上传录音 Upload audio", type=["wav","mp3","m4a","ogg"], key="upload_audio")
        if uploaded:
            st.audio(uploaded)
            transcript = transcribe_audio(uploaded.read())

    # Manual text input fallback
    manual = st.text_area(
        "✍️ 或直接输入答案 (Or type your answer in Chinese):",
        height=100, key="manual_input",
        placeholder="在这里用华文输入你的答案…"
    )
    if manual.strip():
        transcript = manual.strip()

    return transcript

def transcribe_audio(audio_bytes):
    """Try OpenAI Whisper API; fall back to placeholder."""
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        result = client.audio.transcriptions.create(
            model="whisper-1", file=buf, language="zh"
        )
        return result.text
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class="main-header">
        <h1>🎤 华文口试练习系统</h1>
        <p>Singapore Primary School Chinese Oral Practice · P4–P6</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 👋 欢迎 Welcome!")
        st.markdown("""
        练习华文口试，提高说话能力！  
        *Practice Chinese oral examination and improve your speaking skills!*

        **Features 功能:**
        - 🖼️ Picture Discussion 看图说话
        - 🎙️ Speech Recording 录音评分
        - 🤖 AI Examiner Follow-up Questions
        - 📊 MOE-style Scoring Rubric
        - 🏆 Class Leaderboard 班级排行榜
        - 📥 Export Results to Excel
        """)

        st.markdown("---")
        st.markdown("#### 请输入你的名字 Enter Your Name")
        name = st.text_input("学生姓名 Student Name", value=st.session_state.student_name, placeholder="e.g. 陈小明 Tan Xiao Ming")
        cls  = st.text_input("班级 Class", value=st.session_state.class_name, placeholder="e.g. 4A")

        if name.strip():
            st.session_state.student_name = name.strip()
            st.session_state.class_name   = cls.strip()

        if st.button("🚀 开始练习 Start Practice", type="primary", use_container_width=True, disabled=not name.strip()):
            nav("practice")
            st.rerun()

    with col2:
        st.markdown("#### 📋 考试评分标准 Scoring Rubric")
        st.markdown("""
        <table class="rubric-table" style="width:100%;border-collapse:collapse;">
        <tr><th>评分项目</th><th>满分</th></tr>
        <tr><td>📝 内容 Content</td><td style="text-align:center"><b>10</b></td></tr>
        <tr><td>🗣️ 表达 Expression</td><td style="text-align:center"><b>10</b></td></tr>
        <tr><td>📚 词汇 Vocabulary</td><td style="text-align:center"><b>10</b></td></tr>
        <tr><td style="background:#fdf2f2"><b>✅ 总分 Total</b></td><td style="text-align:center;background:#fdf2f2"><b>30</b></td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("#### 🏅 徽章 Badges")
        badges_info = [
            ("🥇", "金牌 Gold", "27–30 分"),
            ("🥈", "银牌 Silver", "21–26 分"),
            ("🥉", "铜牌 Bronze", "15–20 分"),
            ("⭐", "加油 Keep Going", "< 15 分"),
            ("🔥", "连胜 Streak", "3 correct in a row"),
            ("📚", "词汇王 Vocab King", "Perfect vocab score"),
        ]
        for icon, name_b, desc in badges_info:
            st.markdown(f"{icon} **{name_b}** — {desc}")

        st.markdown("---")
        if st.button("👩‍🏫 教师模式 Teacher Dashboard", use_container_width=True):
            nav("teacher")
            st.rerun()
        if st.button("🏆 排行榜 Leaderboard", use_container_width=True):
            nav("leaderboard")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRACTICE
# ══════════════════════════════════════════════════════════════════════════════
def page_practice():
    st.markdown("""
    <div class="main-header">
        <h1>🎤 口试练习 Oral Practice</h1>
        <p>Listen, think, and speak in Chinese!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.student_name or 'Student'}")
        st.markdown(f"班级: {st.session_state.class_name or '—'}")
        st.markdown("---")

        level = st.selectbox("年级 Level", ["All", "P4", "P5", "P6"])
        topic = st.selectbox("题目类型 Topic", [
            "All", "家庭 Family", "学校 School", "社区 Community",
            "环境 Environment", "节日 Festivals", "健康 Health",
            "科技 Technology", "动物 Animals", "食物 Food", "运动 Sports"
        ])

        lvl_filter = None if level == "All" else level
        top_filter = None if topic == "All" else topic.split()[0]  # use Chinese part as key

        if st.button("🎲 随机题目 Random Question", type="primary", use_container_width=True):
            pick_question(lvl_filter, top_filter)
            st.rerun()

        st.markdown("---")
        # Session stats
        sess = st.session_state.session_scores
        if sess:
            avg = sum(s["total"] for s in sess) / len(sess)
            st.metric("本次平均分 Avg Score", f"{avg:.1f}/30")
            st.metric("已答题数 Questions Done", len(sess))

        st.markdown("---")
        if st.button("🏠 主页 Home", use_container_width=True):
            nav("home"); st.rerun()
        if st.button("🏆 排行榜", use_container_width=True):
            nav("leaderboard"); st.rerun()

    # Pick initial question
    if st.session_state.current_q is None:
        pick_question(lvl_filter if 'lvl_filter' in dir() else None)

    q = st.session_state.current_q
    if q is None:
        st.warning("题库加载中… Loading questions…")
        return

    # ── Question display ──────────────────────────────────────────────────────
    st.markdown(f"### 📋 题目 Question — {q['topic']} | {q['level']}")

    col_img, col_q = st.columns([1, 1])

    with col_img:
        # Show picture via URL (Unsplash curated, topic-matched)
        if q.get("image_url"):
            st.image(q["image_url"], use_container_width=True, caption=q.get("image_caption", ""))
        else:
            # Fallback emoji illustration
            st.markdown(f"""
            <div style="background:#fdf2f2;border-radius:12px;padding:40px;
                        text-align:center;font-size:5rem;border:2px dashed #e74c3c;">
                {q.get('emoji','🖼️')}
            </div>
            """, unsafe_allow_html=True)

    with col_q:
        st.markdown(f"""
        <div class="question-box">
            <b>主题：{q['topic']}</b><br><br>
            {q['question_zh']}<br>
            <small style="color:#888">{q['question_en']}</small>
        </div>
        """, unsafe_allow_html=True)

        # Vocabulary hints
        st.markdown("**📚 参考词汇 Vocabulary Hints:**")
        vocab_html = " ".join(f'<span class="vocab-chip">{v}</span>' for v in q.get("vocab", []))
        st.markdown(vocab_html, unsafe_allow_html=True)

        # Follow-up question
        followups = q.get("followup_questions", [])
        if followups and st.session_state.followup_idx < len(followups):
            fi = st.session_state.followup_idx
            st.markdown(f"""
            <div class="feedback-improve">
                🤖 <b>AI考官追问:</b> {followups[fi]}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Recording / Input ─────────────────────────────────────────────────────
    if not st.session_state.answered:
        transcript = audio_recorder_ui()

        col_sub, col_skip = st.columns([3, 1])
        with col_sub:
            if st.button("✅ 提交答案 Submit Answer", type="primary", use_container_width=True,
                         disabled=not bool(transcript and transcript.strip())):
                st.session_state.transcript = transcript
                # Score the response
                result = score_response(transcript, q)
                st.session_state.ai_feedback = result
                st.session_state.answered = True

                # Save to history
                record = {
                    "student": st.session_state.student_name,
                    "class": st.session_state.class_name,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "topic": q["topic"],
                    "level": q["level"],
                    "question": q["question_zh"],
                    "answer": transcript,
                    "content": result["content"],
                    "expression": result["expression"],
                    "vocabulary": result["vocabulary"],
                    "total": result["total"],
                    "badge": result["badge"],
                }
                st.session_state.scores.append(record)
                st.session_state.session_scores.append(record)

                # Add to leaderboard
                add_to_leaderboard(
                    st.session_state.student_name,
                    st.session_state.class_name,
                    result["total"],
                    result["badge"],
                )
                st.rerun()

        with col_skip:
            if st.button("⏭️ 跳过 Skip", use_container_width=True):
                pick_question()
                st.rerun()

    # ── Feedback panel ────────────────────────────────────────────────────────
    if st.session_state.answered and st.session_state.ai_feedback:
        res = st.session_state.ai_feedback
        show_feedback(res, q)

        colA, colB = st.columns(2)
        with colA:
            if st.button("➡️ 下一题 Next Question", type="primary", use_container_width=True):
                pick_question(lvl_filter if 'lvl_filter' in dir() else None, top_filter if 'top_filter' in dir() else None)
                st.rerun()
        with colB:
            if followups and st.session_state.followup_idx < len(followups) - 1:
                if st.button("🤖 追问 Follow-up Question", use_container_width=True):
                    st.session_state.followup_idx += 1
                    st.session_state.answered = False
                    st.session_state.ai_feedback = None
                    st.session_state.transcript = ""
                    st.rerun()


def show_feedback(res, q):
    st.markdown("---")
    st.markdown("## 📊 评分结果 Your Score")

    # Score cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="score-card">
            <div class="score-number">{res['content']}</div>
            <div class="score-label">📝 内容 Content<br>/10</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="score-card">
            <div class="score-number">{res['expression']}</div>
            <div class="score-label">🗣️ 表达 Expression<br>/10</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="score-card">
            <div class="score-number">{res['vocabulary']}</div>
            <div class="score-label">📚 词汇 Vocabulary<br>/10</div></div>""", unsafe_allow_html=True)
    with c4:
        color = "#27ae60" if res['total'] >= 21 else "#e67e22" if res['total'] >= 15 else "#c0392b"
        st.markdown(f"""<div class="score-card" style="border-color:{color}">
            <div class="score-number" style="color:{color}">{res['total']}</div>
            <div class="score-label">✅ 总分 Total<br>/30</div></div>""", unsafe_allow_html=True)

    st.markdown(f"### {res['badge']} 评级 Grade: **{res['grade']}**")

    # Progress bars
    st.progress(res['content'] / 10)
    st.progress(res['expression'] / 10)
    st.progress(res['vocabulary'] / 10)

    col_fb, col_model = st.columns(2)

    with col_fb:
        st.markdown("#### ✅ 做得好 What You Did Well")
        for point in res.get("strengths", []):
            st.markdown(f"""<div class="feedback-good">✔️ {point}</div>""", unsafe_allow_html=True)

        st.markdown("#### 💡 改进建议 How to Improve")
        for point in res.get("improvements", []):
            st.markdown(f"""<div class="feedback-improve">💡 {point}</div>""", unsafe_allow_html=True)

    with col_model:
        st.markdown("#### 📖 参考答案 Model Answer")
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:14px;
                    border-left:4px solid #3498db;font-size:0.95rem;">
            {q.get('model_answer','—')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📚 重要词汇 Key Vocabulary")
        for vw in q.get("vocab_with_pinyin", []):
            st.markdown(f"- **{vw['word']}** ({vw['pinyin']}) — {vw['meaning']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_leaderboard():
    st.markdown("""
    <div class="main-header">
        <h1>🏆 班级排行榜 Class Leaderboard</h1>
        <p>Top students this session</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠 回主页 Home"):
        nav("home"); st.rerun()

    lb = st.session_state.leaderboard
    if not lb:
        st.info("暂无记录。完成练习后分数将显示在这里！\nNo records yet. Complete a practice to appear here!")
        return

    medals = {0: ("gold", "🥇"), 1: ("silver", "🥈"), 2: ("bronze", "🥉")}

    for i, entry in enumerate(lb[:20]):
        css_cls, medal = medals.get(i, ("", ""))
        rank_icon = medal if medal else f"#{i+1}"
        st.markdown(f"""
        <div class="leaderboard-row {css_cls}">
            <span style="font-size:1.3rem">{rank_icon}</span>
            &nbsp;&nbsp;<b>{entry['name']}</b>
            &nbsp;<span style="color:#888;font-size:0.85rem">{entry['class']}</span>
            &nbsp;&nbsp;
            <span style="float:right;font-size:1.2rem;font-weight:700;color:#c0392b">
                {entry['total']}/30 {entry['badges']}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TEACHER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_teacher():
    st.markdown("""
    <div class="main-header">
        <h1>👩‍🏫 教师控制台 Teacher Dashboard</h1>
        <p>Review student performance and export data</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠 回主页 Home"):
        nav("home"); st.rerun()

    df = scores_to_df()

    if df.empty:
        st.info("暂无学生数据。请先完成练习！\nNo student data yet.")
        return

    # Summary stats
    st.markdown("### 📊 总体统计 Overall Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总答题数", len(df))
    c2.metric("平均总分", f"{df['total'].mean():.1f}/30")
    c3.metric("最高分", f"{df['total'].max()}/30")
    c4.metric("最低分", f"{df['total'].min()}/30")

    # Charts
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 分数分布 Score Distribution")
        st.bar_chart(df["total"].value_counts().sort_index())

    with col_r:
        st.markdown("#### 各题型平均分 Avg by Topic")
        topic_avg = df.groupby("topic")["total"].mean().reset_index()
        topic_avg.columns = ["topic", "avg_score"]
        st.bar_chart(topic_avg.set_index("topic"))

    # Detailed table
    st.markdown("### 📋 详细记录 Detailed Records")
    show_cols = ["student", "class", "date", "topic", "level", "content", "expression", "vocabulary", "total", "badge"]
    available = [c for c in show_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True)

    # Export
    st.markdown("### 📥 导出 Export")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 下载 CSV Download CSV",
            data=csv.encode("utf-8-sig"),
            file_name=f"chinese_oral_scores_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df[available].to_excel(writer, index=False, sheet_name="Scores")
                # Add summary sheet
                summary = pd.DataFrame({
                    "Metric": ["Total Questions", "Average Score", "Highest Score", "Lowest Score"],
                    "Value": [len(df), round(df['total'].mean(),1), df['total'].max(), df['total'].min()]
                })
                summary.to_excel(writer, index=False, sheet_name="Summary")
            buf.seek(0)
            st.download_button(
                "📊 下载 Excel Download Excel",
                data=buf,
                file_name=f"chinese_oral_scores_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Excel export requires openpyxl: pip install openpyxl\n{e}")

    # Per-student breakdown
    if "student" in df.columns:
        st.markdown("### 👤 学生个人报告 Individual Student Report")
        students = df["student"].unique().tolist()
        sel_student = st.selectbox("选择学生 Select Student", students)
        sdf = df[df["student"] == sel_student]
        st.markdown(f"**{sel_student}** — {len(sdf)} 次练习 | 平均 {sdf['total'].mean():.1f}/30")
        st.dataframe(sdf[available], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page
if page == "home":
    page_home()
elif page == "practice":
    page_practice()
elif page == "leaderboard":
    page_leaderboard()
elif page == "teacher":
    page_teacher()
else:
    page_home()
