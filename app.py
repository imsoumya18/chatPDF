import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Heading
st.markdown("""
<style>
    #MainMenu
    {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
Made with ❤️ by [Soumya](https://github.com/imsoumya18)\n
Star ⭐ this repo on [GitHub](https://github.com/imsoumya18/chatPDF)
""", unsafe_allow_html=True)
st.title("Chat with PDF 💬")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Soumya](https://github.com/imsoumya18)')

load_dotenv()


def main():
    # upload a PDF file
    pdf = st.file_uploader("Choose a PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        store_name = pdf.name[:-4]
        embeddings = OpenAIEmbeddings()

        if not os.path.exists(f"vectorDB/{store_name}"):
            vectordb = FAISS.from_texts(chunks, embeddings)
            vectordb.save_local(f"vectorDB/{store_name}")
            st.write("Saved successfully")

        vectordb = FAISS.load_local(f"vectorDB/{store_name}", embeddings, allow_dangerous_deserialization=True)
        st.write("Loaded successfully")

        query = st.text_input("Ask questions about the PDF file: ")

        if query:
            docs = vectordb.similarity_search(query, k=3)

            llm = OpenAI(temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            resp = chain.run(input_documents=docs, question=query)
            st.write(resp)


if __name__ == '__main__':
    main()
