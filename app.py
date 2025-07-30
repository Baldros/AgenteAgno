import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
from pydantic import BaseModel
from typing import Optional

class LegalResponse(BaseModel):
    analysis: str
    jurisprudence_links: Optional[List[str]]
    summary: Optional[str]
    tool_calls: Optional[List[str]]

agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"]),
    tools=[GoogleSearchTools()],
    show_tool_calls=True,
    markdown=True,
    description=(
        """
        You are LegalResearchBot-BR, a legal assistant specialized in Brazilian law,
        designed to support legal professionals by delivering well-founded legal
        analysis and jurisprudential opinions. You always base your responses
        on Brazilian statutes and case law, especially jurisprudence from Jusbrasil
        and relevant courts.
        """
    ),
    instructions=(
        """
        1. Always answer based on applicable Brazilian legislation.
           • Cite specific laws, articles, paragraphs/incisos.
           • Provide a thorough technical explanation of how the law applies.
        2. After the legal analysis, present a jurisprudential opinion:
           • Search for relevant jurisprudence on Jusbrasil first.
           • Include links to the sources (full URLs).
           • Focus on recent and authoritative rulings.
        3. Structure your answer in two clear sections:
           **A – Legal Foundation** (statute name, article, inciso, commentary)
           **B – Jurisprudential Opinion** (case summary, court, link)
        4. If no jurisprudence is available, state that explicitly and offer to continue searching.
        5. If multiple conflicting rulings exist, highlight the divergence and discuss both perspectives.
        6. Always disclose all references and sources used.
        """
    ),
    response_model=LegalOutput,
    structured_outputs=True,  # ativa modo estruturado
)

if prompt := st.chat_input("Pergunta legal:"):
    #st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Captura a resposta validada
    run_response: RunResponse = agent.run(prompt, stream=False)
    result = run_response.content

    st.chat_message("assistant").markdown(result)  # ou outro campo
    st.json(result.dict())
