import os
from dataclasses import dataclass
from dotenv import load_dotenv
import streamlit as st
import pandas as pd 
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from htmlTemplates import css,bot_template,user_template
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.llms import OpenAI
# from langchain.agents.agent_types import AgentType
# from langchain.chains import ConversationalRetrievalChain

load_dotenv()
st.set_page_config(page_title="Talk to your data",
                       page_icon=":robot_face:")
st.write(css, unsafe_allow_html=True)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = AzureOpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


@st.cache_resource
def get_llm():
    llm = AzureChatOpenAI(
            openai_api_version=os.getenv("openai_api_version"),
            azure_deployment="gpt-16k-model",)
    # print("Got the LLM!")
    return llm

def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = \
            [
                Message
                (
                    actor = ASSISTANT,
                    payload =
                    """
                        You can ask me anything about the data uploaded
                    """
                )
            ]
    # if "llm_chain" not in st.session_state:
    #     vectorstore = None
    #     st.session_state["llm_chain"] = get_llm_chain(vectorstore)

def user_input(prompt):
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    with st.spinner("Please wait.."):
        agent = create_pandas_dataframe_agent(
                get_llm(),
                df,
                agent_type="openai-tools",
                verbose=True
        )
        # print("agent created!",agent)
        try: 
            response = agent.invoke(prompt)
        except:
            print("There is an error in generating response, please try again!")
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response["output"]))
        st.chat_message(ASSISTANT).write(response["output"])

st.sidebar.title('TalktoData')
navigation = st.sidebar.radio("Talk to", ('CSV File', 'PDF File'))
# navigation = st.sidebar.radio("Talk to",('CSV File'))

if navigation == 'CSV File':
    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    @dataclass
    class Message:
        actor: str
        payload: str
    
    st.header('Talk to your CSV files here!')

    initialize_session_state()

    with st.sidebar:
        user_csv = st.file_uploader("Upload your data here",type="csv")
        if user_csv:
            df = pd.read_csv(user_csv)

    prompt= st.chat_input("Enter your query here")

    msg: Message
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)

    if user_csv is None and prompt is not None:
        st.error("Please upload your data first!")
    elif user_csv is not None and prompt is not None:
        user_input(prompt)

#---------------------------------------------------------------------------------------------------------------------#
elif navigation == 'PDF File':
    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    @dataclass
    class Message:
        actor: str
        payload: str
    
    st.header('Talk to your PDF files here!')

    initialize_session_state()

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True,type="pdf")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

    msg: Message
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)

    prompt= st.chat_input("Enter your query here")

#     if prompt:
#         st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
#         st.chat_message(USER).write(prompt)

#     with st.spinner("Please wait.."):
#         llm_chain = get_llm_chain_from_session()
#         response = llm_chain({"question": prompt})["text"]
#         st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
#         st.chat_message(ASSISTANT).write(response)






