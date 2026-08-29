import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Cover Letter Generator",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #546E7A;
        margin-bottom: 1.8rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1565C0;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.65rem 1rem;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar – Configuration
# --------------------------------------------------
with st.sidebar:
    st.header("🤖  AI Resume Evaluator")

    # API Key input
    api_key = ''
    model_name="models/gemini-3.5-flash"
    temperature=0
    max_tokens=2048

    #
    # st.markdown("---")

    # model_name = st.selectbox(
    #     "Model",
    #     options=[
    #         "models/gemini-2.0-flash",
    #         "models/gemini-1.5-flash",
    #         "models/gemini-1.5-pro",
    #         "models/gemini-2.0-flash-lite"
    #     ],
    #     index=0,
    #     help="gemini-2.0-flash is fast and recommended"
    # )

    # temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    # max_tokens = st.slider("Max Output Tokens", 512, 4096, 2048, 256)
    #
    # st.markdown("---")
    st.markdown("### 📋 How to use")
    st.markdown("""
    1. Upload a PDF resume
    2. Enter your evaluation question
    3. Click **Evaluate Resume**
    """)

    st.markdown("---")
    st.caption("Built with Streamlit + LangChain + Gemini")
# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown('<p class="main-header">✉️ AI Cover Letter Generator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload your resume + paste a job description → get a highly tailored cover letter</p>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Input Sections
# --------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Resume (PDF)")
    uploaded_file = st.file_uploader(
        "Choose your resume PDF",
        type=["pdf"],
        help="Only PDF files are supported"
    )
    if uploaded_file is not None:
        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

with col2:
    st.subheader("2. Additional Instructions (Optional)")
    question = st.text_area(
        "Focus areas / special instructions",
        value="Highlight relevant technical skills, leadership experience, and alignment with the role.",
        height=140,
        help="Example: Highlight cloud architecture and leadership experience. Keep it under 400 words."
    )

st.subheader("3. Target Job Description")
jd_text = st.text_area(
    "Paste the full job description here",
    height=280,
    placeholder="Paste the complete job description, responsibilities, required skills, etc.",
    help="The more detailed the JD, the better the cover letter will be tailored."
)

# --------------------------------------------------
# Prompt Template (from your notebook)
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are an expert career advisor and professional resume writer.

Using the candidate's resume and the target job description provided below, draft a compelling, highly tailored cover letter.

Candidate Resume:
{context}

Target Job Description:
{job_description}

Additional Instructions / Focus Areas:
{question}

Cover Letter:
"""

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Save uploaded PDF temporarily and extract text with PyPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        context = "\n\n".join(doc.page_content for doc in documents)
        return context
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_llm(api_key: str, model_name: str, temperature: float, max_tokens: int):
    """Initialize the Gemini chat model."""
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=2048
    )
    return llm


# --------------------------------------------------
# Generate Button
# --------------------------------------------------
st.markdown("---")
generate_btn = st.button("🚀 Generate Cover Letter", type="primary", use_container_width=True)

if generate_btn:
    # Validation
    if not api_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    if uploaded_file is None:
        st.error("❌ Please upload a PDF resume first.")
        st.stop()

    if not jd_text.strip():
        st.error("❌ Please paste a Job Description.")
        st.stop()

    # Extract resume text
    with st.spinner("📄 Extracting text from resume..."):
        try:
            context = extract_text_from_pdf(uploaded_file)
            if not context.strip():
                st.error("Could not extract any text from the PDF. Please try another file.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            st.stop()

    # Optional: show extracted text
    with st.expander("📝 View extracted resume text", expanded=False):
        st.text_area("Extracted Content", context, height=220, disabled=True)

    # Generate cover letter
    with st.spinner("✍️ Writing a tailored cover letter with Gemini... Please wait."):
        try:
            llm = get_llm(api_key, model_name, temperature, max_tokens)

            prompt = PromptTemplate(
                input_variables=["context", "job_description", "question"],
                template=PROMPT_TEMPLATE
            )

            chain = prompt | llm | StrOutputParser()

            # Stream the response
            response_placeholder = st.empty()
            full_response = ""

            for chunk in chain.stream({
                "context": context,
                "job_description": jd_text,
                "question": question if question.strip() else "Write a professional and compelling cover letter."
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            st.success("✅ Cover letter generated successfully!")

            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=full_response,
                    file_name="cover_letter.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_dl2:
                # Also offer a simple .md version
                st.download_button(
                    label="📥 Download as Markdown",
                    data=full_response,
                    file_name="cover_letter.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            st.info("💡 Tips: Check that your API key is valid and has remaining quota. Also verify the model name.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Your resume is processed temporarily and is never stored permanently. Powered by Google Gemini.")
