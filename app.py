import streamlit as st
import json
from typing import Optional
from agno.tools import tool
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
from pydantic import BaseModel
import requests
import re
import unicodedata


# -------------------- Modelo de saída --------------------
class LegalOutput(BaseModel):
    analysis: str
    jurisprudence_links: Optional[list[str]]
    summary: Optional[str]

# -------------------- Agente --------------------
agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"]),
    tools=[GoogleSearchTools()],
    show_tool_calls=True,
    markdown=True,
    description=(
        """
        You are **Legal Counsel-IA (Brazil)**, acting as a legal consultant.

        Your role is to analyze facts under Brazilian law and **recommend the most practical course of action** in a clear and accessible way, without compromising legal accuracy.

        You prioritize **simplicity** and **decisiveness**: first, you explain the situation in plain, digestible terms; then, you recommend a specific course of action with a brief legal justification.
        """
    ),
    instructions=(
        """
# Objective  
Provide **short and actionable answers by default**, with the option to deepen the analysis if the user requests it.

# Response Style (always in two parts)  
1) **Practical Summary (up to 5 lines):** explain what matters in simple, clear language.  
2) **Recommendation (decisive direction):** state clearly what should be done and why.  
   - If there are significant risks, mention them in 1–2 lines.  
   - Offer alternative paths only when they are truly relevant.  

# Complexity Levels (auto-adjust based on user intent)  
- **Basic (default):** if the user does not request detail → keep it short, no extensive citations.  
- **Technical:** if the user asks to "detail", "justify", or "cite articles" → include the legal basis with statute name, article, paragraph/inciso, and brief commentary.  
- **Formal Opinion/Advanced:** if the user asks for a formal legal opinion → build full legal reasoning, including factual premises, systematic interpretation, and potential norm conflicts.  

# Legal Basis  
- Always respond based on **applicable Brazilian legislation**.  
- When citing legal provisions (at the Technical or Opinion level), include **law name + article/inciso/paragraph**, and if possible, **link to official sources** (e.g., Planalto, STF, STJ, CNJ).  
- **Jurisprudence:** **only include if the user asks**. If requested, search for recent and relevant case law (especially from higher courts) and provide **full links**.  

# Use of Tools  
- Use search capabilities to **validate up-to-date legal texts**, and when requested, to locate jurisprudence or scholarly sources.  
- Prioritize official sources, and secondarily use trusted legal repositories. Always provide full URLs when citing.  

# Conflicts or Divergences  
- If there are conflicting precedents or relevant legal gaps, **briefly explain the divergence** and still provide a **main recommendation** (including a brief reason for your preference).  

# Limits and Ethics  
- Never fabricate citations; if there is no clear legal basis, state what **must be investigated** and suggest next steps.  
- If essential facts are missing, ask **only the necessary, targeted questions** to proceed.  

# Output Format (when structured_outputs=True)  
- **summary:** use the Practical Summary.  
- **analysis:** include the brief justification (or full reasoning if the user requests).  
- **jurisprudence_links:** **populate only if the user explicitly requests case law**.

        """
    ),
    response_model=LegalOutput,
    structured_outputs=True,  # ativa modo estruturado
)
# ------------------- Tools -----------------------------------------
@ tool
def serch_site_by_url(url: str) -> str:
    """
    Tool purpose:
    -------------
    This function serves as an agent tool within an LLM pipeline.
    Its role is:
      - Accept a URL provided by a user prompt,
      - Fetch the page via the `requests` library,
      - Decode the content appropriately,
      - Convert HTML to Markdown,
      - Clean up encoding artifacts using regex,
      - Return a clean Markdown string for agent consumption.
    """

    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
    }

    resp = requests.get(url, headers=headers, timeout=20.0)
    resp.raise_for_status()  # Raise HTTPError on bad 4xx/5xx responses

    # Step 1: Determine encoding
    # requests uses Content-Type charset if present; otherwise
    # it may guess using chardet or charset_normalizer via resp.apparent_encoding
    encoding = resp.encoding or resp.apparent_encoding
    resp.encoding = encoding  # override to ensure correct decoding
    html = resp.text

    # Step 2: Convert HTML to Markdown using your converter
    md = convert_to_markdown(html)

    # Step 3: Normalize Unicode (e.g. combine accents properly)
    md = unicodedata.normalize('NFC', md)

    # Step 4: Cleanup control characters and placeholders
    md = md.replace('\r', ' ').replace('\xa0', ' ')
    md = re.sub(r'�+', ' ', md)               # remove garbled replacement characters
    md = re.sub(r'<!--.*?-->', '', md, flags=re.DOTALL)  # strip HTML comments
    md = re.sub(r'\n{3,}', '\n\n', md)        # collapse multiple blank lines
    md = re.sub(r'[ \t]{2,}', ' ', md)        # collapse multiple spaces/tabs

    return md

# -------------------- Inicialização de estado --------------------
def new_chat(default_title: str | None = None) -> str:
    """Cria uma nova conversa e retorna o chat_id."""
    next_id = str(len(st.session_state.chats) + 1)
    st.session_state.chats[next_id] = {
        "title": default_title or f"Conversa {next_id}",
        "messages": [],
    }
    st.session_state.active_chat_id = next_id
    return next_id

if "chats" not in st.session_state:
    # tente carregar de arquivo (opcional)
    try:
        with open("chats.json", "r", encoding="utf-8") as f:
            st.session_state.chats = json.load(f)
    except Exception:
        st.session_state.chats = {}

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# se não houver conversas, cria a primeira
if not st.session_state.chats:
    new_chat()

# -------------------- Sidebar: seleção/novo/renomear --------------------
chat_ids = list(st.session_state.chats.keys())

selected_id = st.sidebar.selectbox(
    "Selecione a conversa:",
    options=chat_ids,
    index=chat_ids.index(st.session_state.active_chat_id)
           if st.session_state.active_chat_id in chat_ids else 0,
    format_func=lambda cid: st.session_state.chats[cid]["title"],
)
st.session_state.active_chat_id = selected_id

if st.sidebar.button("➕ Nova conversa"):
    new_chat()
    st.rerun() # Para o UI refletir a nova conversa
    

# Renomear conversa
current_title = st.session_state.chats[selected_id]["title"]
new_title = st.sidebar.text_input("Título da conversa", value=current_title, key=f"title_{selected_id}")
if new_title != current_title:
    st.session_state.chats[selected_id]["title"] = new_title

# -------------------- Área principal: render e input --------------------
chat = st.session_state.chats[selected_id]
st.header(chat["title"])
history = chat["messages"]

# Render do histórico
for turn in history:
    st.chat_message("user").markdown(turn["user"])
    if "model" in turn:
        st.chat_message("assistant").markdown(turn["model"])
        links = turn.get("links") or []
        if links:
            with st.expander("Jurisprudências citadas"):
                for link in links:
                    st.write(f"- {link}")

# Input do usuário
if prompt := st.chat_input("Pergunta legal:"):
    # 1) imprime a mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) adiciona ao histórico
    history.append({"user": prompt})

    # 3) monta o contexto
    full_context = "\n".join(
        f"User: {t['user']}\n Assistant: {t.get('model','')}".strip()
        for t in history
    )

    # 4) cria um placeholder para a mensagem do assistente
    assistant_box = st.chat_message("assistant")

    # 5) indicador de processamento + chamada ao modelo
    with assistant_box:
        with st.spinner("Analisando legislação e buscando jurisprudência…"):
            run_response: RunResponse = agent.run(full_context, stream=False)
            resp = run_response.content  # instancia de LegalOutput

        # 6) imprime a resposta imediatamente
        st.markdown(resp.analysis)
        if resp.jurisprudence_links:
            st.write("### Jurisprudências citadas")
            for link in resp.jurisprudence_links:
                st.write(f"- {link}")

    # 7) atualiza o histórico (para aparecer também nos próximos reruns)
    history[-1]["model"] = resp.analysis
    if resp.jurisprudence_links:
        history[-1]["links"] = resp.jurisprudence_links

    # 8) persiste no session_state (e opcionalmente em arquivo)
    st.session_state.chats[selected_id]["messages"] = history
    with open("chats.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)

# Excluir conversa atual
if st.sidebar.button("🗑️ Excluir conversa atual"):
    # Remove a conversa selecionada
    del st.session_state.chats[selected_id]

    # Escolhe nova conversa ativa
    if st.session_state.chats:
        st.session_state.active_chat_id = next(iter(st.session_state.chats.keys()))
    else:
        new_chat()  # recria uma conversa vazia

    # Persiste e recarrega UI
    with open("chats.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)
    st.rerun()
