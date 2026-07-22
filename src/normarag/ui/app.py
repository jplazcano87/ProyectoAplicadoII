"""Streamlit demo UI for NormaRAG (presentation layer only).

Consumes the existing API contract unchanged: answer, citations, abstained,
config, latency_ms, cost_usd, sources[], sub_questions, rejected_citations.
No backend / pipeline / retrieval logic lives here.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

CONFIGS = {
    "A": "A · baseline (LLM solo)",
    "B": "B · RAG naive (denso)",
    "C": "C · RAG mejorado (híbrido + rerank)",
    "D": "D · agéntico (sub-preguntas + self-check)",
}
CONFIG_DESC = {
    "A": "Solo el LLM, sin acceso al corpus. Baseline de referencia.",
    "B": "Recuperación densa top-k + generación con cita.",
    "C": "Búsqueda híbrida (densa + BM25) con fusión RRF, reranking Cohere y prompt de "
    "citación/abstención.",
    "D": "Agente que descompone la pregunta en sub-preguntas, recupera por cada una y "
    "**verifica cada cita contra el corpus** antes de responder.",
}

EXAMPLES = [
    (
        "Cierre de oficina",
        "¿Con cuánta anticipación debe un banco informar el cierre de una oficina?",
    ),
    ("Comité de Auditoría", "¿Es obligatorio que cada banco cuente con un Comité de Auditoría?"),
    ("Riesgo-país", 'Para efectos de la normativa, ¿qué se entiende por "riesgo-país"?'),
    (
        "Fuera de alcance ⚠️",
        "¿Qué sanción penal aplica por girar un cheque sin fondos según el Código Penal?",
    ),
]

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="NormaRAG", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
      .nr-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
        color: #fff; padding: 20px 26px; border-radius: 12px; margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(31,78,121,.18);
      }
      .nr-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: .3px; }
      .nr-header p  { margin: 4px 0 0; opacity: .92; font-size: .98rem; }
      .nr-answer {
        background: #f7fafc; border: 1px solid #e2e8f0; border-left: 4px solid #2e6da4;
        padding: 16px 20px; border-radius: 10px; font-size: 1.06rem; line-height: 1.6;
        color: #1f2933;
      }
      .nr-abstain {
        background: #fff6e8; border-left: 5px solid #e8912d; padding: 16px 18px;
        border-radius: 10px; color: #7a4b12; font-size: 1.02rem;
      }
      .nr-chip {
        display: inline-block; padding: 5px 12px; margin: 3px 5px 3px 0;
        border-radius: 16px; font-size: .86rem; font-weight: 600; line-height: 1.4;
      }
      .nr-chip-ok    { background: #e6f4ec; color: #1a7f5a; border: 1px solid #b7e0c9; }
      .nr-chip-plain { background: #eef1f5; color: #3a4757; border: 1px solid #d6dde6; }
      .nr-chip-bad   { background: #fdecec; color: #b3261e; border: 1px solid #f3c2be;
                       text-decoration: line-through; }
      .nr-sub { color: #2e6da4; font-weight: 600; }
      .stButton>button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nr-header">
      <h1>⚖️ NormaRAG</h1>
      <p>Asistente de consulta sobre la normativa bancaria chilena (RAN de la CMF) ·
         responde con <b>cita verificable</b> (capítulo + numeral) y se abstiene si no hay evidencia.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers de presentación
# --------------------------------------------------------------------------- #
def _source_index(sources: list[dict]) -> dict:
    """Mapa (capítulo, numeral) -> fragmento recuperado, para evidenciar citas."""
    idx = {}
    for s in sources:
        idx[(str(s.get("chapter_code")), str(s.get("numeral")))] = s
    return idx


def render_citations(citations: list[dict], sources: list[dict]) -> None:
    if not citations:
        return
    idx = _source_index(sources)
    chips = []
    for c in citations:
        key = (str(c.get("chapter")), str(c.get("numeral")))
        verified = key in idx
        cls = "nr-chip-ok" if verified else "nr-chip-plain"
        mark = "✓" if verified else "•"
        chips.append(
            f'<span class="nr-chip {cls}">{mark} Cap. {c.get("chapter")} · '
            f'Numeral {c.get("numeral")}</span>'
        )
    st.markdown("###### 📌 Citas")
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    # Evidencia verificable: el fragmento del numeral que respalda cada cita.
    with_evidence = [c for c in citations if (str(c.get("chapter")), str(c.get("numeral"))) in idx]
    if with_evidence:
        st.caption("Evidencia (el fragmento que respalda cada cita):")
        for c in with_evidence:
            s = idx[(str(c.get("chapter")), str(c.get("numeral")))]
            titulo = f" · Título {s['title_section']}" if s.get("title_section") else ""
            with st.expander(
                f"✓ Cap. {c.get('chapter')}{titulo} · Numeral {c.get('numeral')}"
                f" — {s.get('chapter_title', '')}"
            ):
                st.write(s.get("text", ""))
                if s.get("source_url"):
                    st.markdown(f"[📄 PDF fuente]({s['source_url']})")


def render_rejected(rejected: list[dict]) -> None:
    if not rejected:
        return
    items = ", ".join(f"Cap. {c.get('chapter')} · {c.get('numeral')}" for c in rejected)
    st.markdown(
        f'<div class="nr-abstain">🛡️ <b>Self-check:</b> se rechazaron citas que no existen '
        f"en el corpus: {items}</div>",
        unsafe_allow_html=True,
    )


def render_sources_panel(sources: list[dict]) -> None:
    if not sources:
        return
    ranked = sorted(
        sources,
        key=lambda s: (
            s.get("rerank_score") if s.get("rerank_score") is not None else s.get("score") or 0
        ),
        reverse=True,
    )
    with st.expander(f"🔎 Fragmentos recuperados ({len(ranked)}) — cómo llegó a la respuesta"):
        for i, s in enumerate(ranked, start=1):
            titulo = f" · Título {s['title_section']}" if s.get("title_section") else ""
            head = (
                f"**[{i}] Cap. {s.get('chapter_code')}{titulo} · "
                f"Numeral {s.get('numeral')}** — {s.get('chapter_title', '')}"
            )
            score_bits = []
            if s.get("rerank_score") is not None:
                score_bits.append(f"rerank `{s['rerank_score']:.3f}`")
            if s.get("score") is not None:
                score_bits.append(f"retrieval `{s['score']:.3f}`")
            st.markdown(head + ("  ·  " + " · ".join(score_bits) if score_bits else ""))
            text = (s.get("text") or "")[:400]
            st.caption(text + ("…" if len(s.get("text") or "") > 400 else ""))
            st.divider()


def render_result(data: dict, config: str, compact: bool = False) -> None:
    """Renderiza una respuesta completa. compact=True para el modo comparar."""
    # Métricas de la consulta.
    m = st.columns(3)
    m[0].metric("Config", data.get("config", config))
    if data.get("latency_ms") is not None:
        m[1].metric("Latencia", f"{data['latency_ms'] / 1000:.2f} s")
    m[2].metric("Fragmentos", len(data.get("sources", [])))

    # Sub-preguntas del agente (config D).
    subq = data.get("sub_questions", [])
    if subq and config == "D":
        st.markdown("###### 🧭 Sub-preguntas del agente")
        for i, sq in enumerate(subq, start=1):
            st.markdown(f'<span class="nr-sub">{i}.</span> {sq}', unsafe_allow_html=True)

    # Respuesta o abstención.
    if data.get("abstained"):
        st.markdown(
            '<div class="nr-abstain">🚫 <b>Sin evidencia suficiente en la normativa.</b><br>'
            "El sistema se abstiene en lugar de inventar una respuesta. "
            "<i>(La abstención es un feature: evita alucinaciones.)</i></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("###### 💬 Respuesta")
        st.markdown(
            f'<div class="nr-answer">{data.get("answer", "")}</div>', unsafe_allow_html=True
        )

    render_rejected(data.get("rejected_citations", []))
    render_citations(data.get("citations", []), data.get("sources", []))
    if not compact:
        render_sources_panel(data.get("sources", []))


def query_api(question: str, config: str, top_k: int) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_URL}/query",
            json={"question": question, "config": config, "top_k": top_k},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(
            f"La API respondió {exc.response.status_code}. Revisa la configuración y el corpus."
        )
    except httpx.HTTPError:
        st.error(
            f"No se pudo conectar con la API en `{API_URL}`. "
            "¿Está levantada? (`make demo` o `docker compose up`)."
        )
    return None


# --------------------------------------------------------------------------- #
# Controles (barra lateral)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    config_label = st.radio(
        "Pipeline", list(CONFIGS.values()), index=2, label_visibility="collapsed"
    )
    config = next(k for k, v in CONFIGS.items() if v == config_label)
    st.caption(CONFIG_DESC[config])

    st.divider()
    compare = st.toggle("🔀 Modo comparar (2 pipelines)", value=False)
    config_b = None
    if compare:
        default_b = "B · RAG naive (denso)" if config != "B" else "A · baseline (LLM solo)"
        config_b_label = st.radio(
            "Segundo pipeline",
            [v for k, v in CONFIGS.items() if k != config],
            index=[v for k, v in CONFIGS.items() if k != config].index(default_b),
        )
        config_b = next(k for k, v in CONFIGS.items() if v == config_b_label)

    st.divider()
    top_k = st.slider("Fragmentos (top-k) · exploración", 1, 30, 6, disabled=(config == "A"))
    st.caption("Corpus: Recopilación Actualizada de Normas (RAN) de la CMF.")


# --------------------------------------------------------------------------- #
# Área principal
# --------------------------------------------------------------------------- #
def _use_example(text: str) -> None:
    st.session_state.q_input = text
    st.session_state.autorun = True


st.markdown("**Preguntas de ejemplo** (una está fuera del corpus, para ver la abstención):")
ex_cols = st.columns(len(EXAMPLES))
for col, (short, full) in zip(ex_cols, EXAMPLES):
    col.button(short, on_click=_use_example, args=(full,), use_container_width=True)

question = st.text_input(
    "Tu pregunta",
    key="q_input",
    placeholder="¿Qué plazo tiene un banco para informar el cierre de una oficina?",
)
go = st.button("Consultar", type="primary")
run = go or st.session_state.pop("autorun", False)

if run and question.strip():
    if compare and config_b:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"#### {CONFIGS[config].split(' · ')[0]} · {config}")
            with st.spinner("Consultando la normativa…"):
                d = query_api(question, config, top_k)
            if d:
                render_result(d, config, compact=True)
        with right:
            st.markdown(f"#### {CONFIGS[config_b].split(' · ')[0]} · {config_b}")
            with st.spinner("Consultando la normativa…"):
                d2 = query_api(question, config_b, top_k)
            if d2:
                render_result(d2, config_b, compact=True)
    else:
        with st.spinner("Consultando la normativa…"):
            data = query_api(question, config, top_k)
        if data:
            render_result(data, config)
elif run:
    st.info("Escribe una pregunta o elige un ejemplo.")
