import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
from my_models import MyResponseModel  # Pydantic

agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"]),
    tools=[GoogleSearchTools()],
    show_tool_calls=True,
    markdown=True,
    description="LegalResearchBot‑BR",
    instructions=...,
    response_model=MyResponseModel,
)

if prompt := st.chat_input("Pergunta legal:"):
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Captura a resposta validada
    run_response: RunResponse = agent.run(prompt, stream=False)
    result = run_response.content

    st.chat_message("assistant").markdown(result.feedback)  # ou outro campo
    st.json(result.dict())
