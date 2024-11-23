import streamlit as st
import texts as texts
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import prompts as pr

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# load api key
# load_dotenv()

api_key = st.secrets["api_key"]

# page
st.set_page_config(
    page_title="PaperPal",
    page_icon="📄",
)

# sidebar
with st.sidebar:
    st.header('About')
    st.write(texts.about)
    st.divider()
    st.write("Start experimenting with the below papers or add your own 😊")
    st.info("""
        [1. Multimodal System for Precision Agriculture using IOT and ML](https://drive.google.com/uc?export=download&id=18NQv39XmPhPX5qP4Gii5UNpBhTuLKzPH)\n
        [2. Recognition of Nutrition Facts Labels from Mobile](https://drive.google.com/uc?export=download&id=16UzIt-clkju4v9Xl12M3OeU9k3JEqQ9J)\n
        [3. Evolution of Early SARS-CoV-2 and Cross-Coronavirus](https://drive.google.com/uc?export=download&id=1mAYkDgo6KN4kWFElzCGes78ZRpkv7wFe)
    """)
    st.write("Made with ❤ by [Hamza Khan](https://hamzakhan07.netlify.app/)")


# header
st.header(texts.header)
st.write(texts.subheader)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def start():
    # file upload
    pdf = st.file_uploader('')
    text = ''

    if pdf is not None:
        # check file name
        if 'filename' not in st.session_state:
            st.session_state.filename = pdf.name

        # check if file is new
        if st.session_state.filename != pdf.name:
            # clear session state variables
            if 'vectordb' in st.session_state:
                del st.session_state.vectordb
            if 'chain' in st.session_state:
                del st.session_state.chain
            if 'response' in st.session_state:
                del st.session_state.response
            if 'messages' in st.session_state:
                del st.session_state.messages
            # save new file name
            st.session_state.filename = pdf.name

        # store embeddings into current state
        if 'vectordb' not in st.session_state:
            pdf_object = PdfReader(pdf)

            if len(pdf_object.pages) > 200:
                st.warning('Cannot process more than 200 pages', icon="⚠️")
                return

            with st.spinner('Paper Processing...'):
                for page in pdf_object.pages:
                    text = text + page.extract_text()

                # divide text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                chunks = text_splitter.split_text(text=text)

                # create embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

                # store embeddings
                vectordb = FAISS.from_texts(chunks, embedding=embeddings)
                # store into current state
                st.session_state.vectordb = vectordb
                # store latest file name
                st.session_state.filename = pdf.name

            st.success('Paper processed')
            st.write('\n\n')

            # get insights and chat
            return vectordb
        else:
            vectordb = st.session_state.vectordb
            st.info('Paper processed')
            return vectordb


def get_insights(vectordb):
    if 'chain' not in st.session_state:
        prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """

        # question answer chain
        model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key) #models/text-bison-001
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        st.session_state.chain = chain

    if 'response' not in st.session_state:
        query = pr.prompt_insights
        docs = st.session_state.vectordb.similarity_search(query)

        # response
        response = st.session_state.chain(
            {"input_documents": docs, "question": query}
            , return_only_outputs=True)

        # insights
        print(response)
        st.info(response["output_text"])

        st.write('\n\n')
        st.session_state.response = response["output_text"]
    else:
        # insights
        st.info(st.session_state.response)

    return st.session_state.chain


def handle_chat(prompt):
    # chat
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # add loading
    with st.spinner('Loading...'):
        query = prompt
        try:
            docs = st.session_state.vectordb.similarity_search(query)

            response = st.session_state.chain(
            {"input_documents": docs, "question": query}
            , return_only_outputs=True)

            print('Response Chat: ', response['output_text'])
            st.session_state.chat_result = response['output_text']
        except:
            st.session_state.chat_result = "Sorry, I'm not able to assist you with that"

    response = st.session_state.chat_result
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    db = start()
    if db:
        chain = get_insights(db)
        if chain:
            prompt = st.chat_input("Talk with paper")
            if prompt:
                handle_chat(prompt)

