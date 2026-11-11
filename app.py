"""
Multi-Agent Research Assistant - Streamlit UI

Put this file in the Multi_Agents/ folder (next to pipeline.py) and run:
    streamlit run app.py

It runs the same four steps as pipeline.py (search -> reader -> writer -> critic)
but shows live progress, timings and tidy results in the browser.
"""

import time
import traceback
from datetime import datetime

import streamlit as st

from agents import critic_chain, writer_chain, reader_agent, search_agent

# ----------------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }

/* Hero */
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    color: #fff;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; color: #fff; }
.hero p  { margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem; color: #fff; }

/* Pipeline stepper */
.stepper { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin: .6rem 0 1.2rem 0; }
.step {
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 14px;
    padding: .85rem 1rem;
    background: rgba(128,128,128,.06);
    transition: all .2s ease;
}
.step .icon  { font-size: 1.4rem; }
.step .name  { font-weight: 600; margin-top: .15rem; }
.step .desc  { font-size: .82rem; opacity: .7; }
.step .badge { font-size: .75rem; margin-top: .45rem; display: inline-block;
               padding: .1rem .55rem; border-radius: 999px; background: rgba(128,128,128,.18); }
.step.running { border-color: #7c3aed; box-shadow: 0 0 0 3px rgba(124,58,237,.18);
                animation: pulse 1.4s ease-in-out infinite; }
.step.running .badge { background: rgba(124,58,237,.2); color: #7c3aed; }
.step.done    { border-color: rgba(34,197,94,.7); background: rgba(34,197,94,.08); }
.step.done .badge { background: rgba(34,197,94,.2); color: #16a34a; }
.step.error   { border-color: rgba(239,68,68,.8); background: rgba(239,68,68,.08); }
.step.error .badge { background: rgba(239,68,68,.2); color: #dc2626; }
@keyframes pulse { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-2px); } }

/* Stat cards */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: .7rem 1rem;
    background: rgba(128,128,128,.05);
}

/* Buttons */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px;
}

@media (max-width: 800px) { .stepper { grid-template-columns: repeat(2, 1fr); } }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
STEPS = [
    ("search", "🔍", "Search", "Finds recent, reliable sources"),
    ("reader", "📖", "Reader", "Scrapes the best URL"),
    ("writer", "✍️", "Writer", "Drafts the report"),
    ("critic", "🧐", "Critic", "Reviews the report"),
]

EXAMPLES = [
    "Latest advances in small language models",
    "How is AI changing healthcare diagnostics?",
    "State of quantum computing in 2026",
    "Future of renewable energy storage",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_text(content) -> str:
    """Normalise model output (str, message object, or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if hasattr(content, "content"):
        return to_text(content.content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def stepper_html(states: dict, timings: dict) -> str:
    labels = {"idle": "Waiting", "running": "Working...", "done": "Done", "error": "Failed"}
    cards = []
    for key, icon, name, desc in STEPS:
        state = states.get(key, "idle")
        badge = labels[state]
        if state == "done" and key in timings:
            badge = f"Done · {timings[key]:.1f}s"
        cards.append(
            f'<div class="step {state}">'
            f'<div class="icon">{icon}</div>'
            f'<div class="name">{name}</div>'
            f'<div class="desc">{desc}</div>'
            f'<span class="badge">{badge}</span>'
            f"</div>"
        )
    return f'<div class="stepper">{"".join(cards)}</div>'


def fresh_states() -> dict:
    return {key: "idle" for key, *_ in STEPS}


def set_topic_and_run(topic: str):
    st.session_state.topic_input = topic
    st.session_state.run_now = True


def load_past_run(index: int):
    run = st.session_state.history[index]
    st.session_state.topic_input = run["topic"]
    st.session_state.result = run
    st.session_state.states = {key: "done" for key, *_ in STEPS}


def build_full_markdown(run: dict) -> str:
    return (
        f"# Research Report: {run['topic']}\n\n"
        f"*Generated {run['when']}*\n\n"
        f"{run['report']}\n\n---\n\n"
        f"## Critic Feedback\n\n{run['feedback']}\n\n---\n\n"
        f"## Sources (search results)\n\n{run['search_result']}\n"
    )


def run_pipeline(topic: str, reader_chars: int, states: dict, timings: dict, draw, feed) -> dict:
    """Same logic as pipeline.run_research_pipeline, with live UI updates.

    `draw` redraws the stepper; `feed` is a container where each finished
    step's output is shown as it arrives.
    """
    state = {}

    def start(key):
        states[key] = "running"
        draw()
        return time.time()

    def finish(key, t0):
        timings[key] = time.time() - t0
        states[key] = "done"
        draw()

    def show(title, text):
        if feed is not None:
            with feed.expander(title, expanded=False):
                st.markdown(text)

    current = None
    try:
        # Step 1: Search agent
        current = "search"
        t0 = start(current)
        result = search_agent().invoke(
            {"messages": [("user", f"Find recent, reliable and information about: {topic}")]}
        )
        state["search_result"] = to_text(result["messages"][-1].content)
        finish(current, t0)
        show("🔍 Search results", state["search_result"])

        # Step 2: Reader agent
        current = "reader"
        t0 = start(current)
        result = reader_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based On the following search result about {topic}. "
                        f"Pick The Most Relevant URL and Scrap it for deeper content.\n\n"
                        f"Search results: \n {state['search_result'][:reader_chars]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(result["messages"][-1].content)
        finish(current, t0)
        show("📖 Scraped content", state["scraped_content"])

        # Step 3: Writer chain
        current = "writer"
        t0 = start(current)
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_result']}\n\n"
            f"DETAILED SCRAPED RESULTS : \n {state['scraped_content']}"
        )
        state["report"] = to_text(
            writer_chain.invoke({"topic": topic, "research": research_combined})
        )
        finish(current, t0)
        show("✍️ Draft report", state["report"])

        # Step 4: Critic chain
        current = "critic"
        t0 = start(current)
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        finish(current, t0)

        return state

    except Exception:
        if current:
            states[current] = "error"
            draw()
        raise


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("topic_input", "")
st.session_state.setdefault("run_now", False)
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("states", fresh_states())


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    reader_chars = st.slider(
        "Search text passed to Reader",
        min_value=400,
        max_value=3000,
        value=800,
        step=100,
        help="How many characters of the search results the reader agent sees. "
        "800 matches pipeline.py. Raise it if the reader picks weak links.",
    )
    show_steps = st.toggle(
        "Show intermediate outputs while running",
        value=True,
        help="Display each agent's output as soon as it finishes.",
    )

    st.divider()
    st.markdown("### 🕘 Past runs")
    if st.session_state.history:
        for i in range(len(st.session_state.history) - 1, -1, -1):
            run = st.session_state.history[i]
            label = run["topic"] if len(run["topic"]) <= 34 else run["topic"][:32] + "…"
            st.button(
                label,
                key=f"hist_{i}",
                use_container_width=True,
                on_click=load_past_run,
                args=(i,),
                help=f"{run['when']} · {run['total']:.0f}s",
            )
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.session_state.states = fresh_states()
            st.rerun()
    else:
        st.caption("Your runs from this session will show up here.")


# ----------------------------------------------------------------------------
# Hero + input
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🔎 Multi-Agent Research Assistant</h1>
  <p>Four AI agents search, read, write and critique, so you get a reviewed report on any topic.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("research_form", border=False):
    col_in, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_in:
        st.text_input(
            "What do you want to research?",
            key="topic_input",
            placeholder="e.g. Latest advances in small language models",
        )
    with col_btn:
        submitted = st.form_submit_button("Run research", type="primary", use_container_width=True)

st.caption("Or try an example:")
ex_cols = st.columns(len(EXAMPLES))
for col, example in zip(ex_cols, EXAMPLES):
    col.button(
        example,
        key=f"ex_{example}",
        use_container_width=True,
        on_click=set_topic_and_run,
        args=(example,),
    )

# ----------------------------------------------------------------------------
# Pipeline progress (stepper) + run
# ----------------------------------------------------------------------------
stepper_ph = st.empty()
live_ph = st.empty()

current_timings = (st.session_state.result or {}).get("timings", {})
stepper_ph.markdown(stepper_html(st.session_state.states, current_timings), unsafe_allow_html=True)

run_topic = None
if submitted:
    run_topic = st.session_state.topic_input.strip()
    if not run_topic:
        st.warning("Please enter a research topic first.")
        run_topic = None
elif st.session_state.run_now:
    run_topic = st.session_state.topic_input.strip() or None
st.session_state.run_now = False

if run_topic:
    states = fresh_states()
    timings = {}
    st.session_state.result = None
    st.session_state.states = states

    def draw():
        stepper_ph.markdown(stepper_html(states, timings), unsafe_allow_html=True)

    draw()
    started = time.time()

    with live_ph.container():
        feed = st.container() if show_steps else None
        with st.spinner("Agents are working. This can take a minute or two..."):
            try:
                out = run_pipeline(run_topic, reader_chars, states, timings, draw, feed)
                run = {
                    "topic": run_topic,
                    "when": datetime.now().strftime("%d %b %Y, %H:%M"),
                    "total": time.time() - started,
                    "timings": dict(timings),
                    **out,
                }
                st.session_state.result = run
                st.session_state.history.append(run)
            except Exception as e:
                st.error(f"The pipeline stopped with an error: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    st.session_state.states = states
    live_ph.empty()  # results are shown below, so clear the live feed


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
run = st.session_state.result
if run:
    st.markdown(f"## 📄 {run['topic']}")

    words = len(run["report"].split())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total time", f"{run['total']:.0f}s")
    m2.metric("Report length", f"{words} words")
    m3.metric("Reading time", f"~{max(1, round(words / 200))} min")
    m4.metric("Generated", run["when"].split(",")[1].strip())

    tab_report, tab_feedback, tab_sources, tab_raw = st.tabs(
        ["📄 Report", "🧐 Critic feedback", "🌐 Sources", "📋 Copy markdown"]
    )

    with tab_report:
        left, right = st.columns([3, 1])
        with right:
            st.download_button(
                "⬇️ Report (.md)",
                data=run["report"],
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Report + critique (.md)",
                data=build_full_markdown(run),
                file_name="research_report_full.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with left:
            st.markdown(run["report"])

    with tab_feedback:
        st.info("The critic agent reviewed the report above and left this feedback.", icon="🧐")
        st.markdown(run["feedback"])

    with tab_sources:
        with st.expander("🔍 Search agent output", expanded=True):
            st.markdown(run["search_result"])
        with st.expander("📖 Reader agent output (scraped page)", expanded=False):
            st.markdown(run["scraped_content"])

    with tab_raw:
        st.caption("Use the copy icon at the top right of the box.")
        st.code(run["report"], language="markdown")

elif not run_topic:
    st.info("Enter a topic above (or pick an example) to start a research run.", icon="👆")