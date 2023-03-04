import streamlit as st
import openai
import pypdf
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

@st.cache_data
def create_embeddings(uploaded_file):
    pages = split_file(uploaded_file)
    filtered_pages = [page for page in pages if page.page_content != '']

    faiss_index = FAISS.from_documents(filtered_pages, OpenAIEmbeddings(openai_api_key=API_KEY))
    return(faiss_index)

def split_file(uploaded_file):
    pdf_reader = pypdf.PdfReader(uploaded_file)

    return [
          Document(
            page_content = page.extract_text(),
            metadata = {"page": i}
          )
          for i, page in enumerate(pdf_reader.pages)
    ]

system_prompts = {
    "Informative": "You are a text summariser for university students. You summarise in the active voice and aim only to capture the main topics. It's very important the summary is informative while remaining accurate. Give a brief, informative summary of the following text:",
    "Engaging": "You are a text summariser for advertising. Give a snappy, engaging summary of the following text:",
    "Creative": "You are a text summariser for a creative writing class. Give a creative, imaginative summary of the following text:"
}

st.set_page_config(
    page_title="Source Finder",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
    "About": "Made by Aeron Laffere."
    }
)

st.sidebar.image("logo.png", width=200)
API_KEY = st.sidebar.text_input("`OpenAI API Key:`", value="", type="password")
st.sidebar.write("You need an OpenAI API key to run this demo. You can get one [here](https://platform.openai.com/signup).")
st.sidebar.write("This application was developed by [@aeronjl](https://twitter.com/aeronjl) and is open source. You can find the source code [here](https://github.com/aeronlaffere/gpt-sourcefinder).")
st.sidebar.write("### Options")
style = st.sidebar.selectbox(label="Summarisation style", options=["Informative", "Engaging", "Creative"])
n_sources = st.sidebar.slider(label="Number of sources", min_value=1, max_value=5, value=3)
st.title("Quickly find sources for your essays")

st.write("""
## How does it work?
1. Upload a PDF or EPUB of any book or journal article and ask a question. 
2. The app will find the most relevant in-text sources and summarise them for you. Page numbers will be provided along with the original text. Note that page numbers correspond to the PDF and may not match the number on the page itself.
""")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded_file is not None:
    faiss_index = create_embeddings(uploaded_file)

if uploaded_file is not None:
    st.header("Ask questions")
    with st.form("question"):
        query = st.text_input("Enter your question:", "What is the nature of justice?")
        submitted = st.form_submit_button("Submit")

    if submitted:
        docs = faiss_index.similarity_search(query, k=n_sources)
        
        for doc in docs:
            with st.spinner("Locating sources..."):
                with st.container():
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        temperature=0.9,
                        messages=[
                            {"role": "system", "content":  system_prompts[style]},
                            {"role": "user", "content": doc.page_content},
                        ]
                    )
                    st.markdown("**Page " + str(doc.metadata["page"]) + "**")
                    st.markdown(response["choices"][0]["message"]["content"])
                    with st.expander("See text"):
                        st.markdown(doc.page_content)

        st.button("See more sources")