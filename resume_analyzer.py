import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import google.generativeai as genai

# Page config
st.set_page_config(
    page_title="AI Resume Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #546E7A;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    .score-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - API Key & Settings
with st.sidebar:
    st.header("🤖  AI Resume Evaluator")

    # API Key input
    api_key = ''
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

# Main content
st.markdown('<p class="main-header">📄 AI Resume Evaluator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a resume PDF and get detailed AI-powered evaluation for Data Science / Tech roles</p>',
    unsafe_allow_html=True)

# Two columns for upload and question
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF resume",
        type=["pdf"],
        help="Only PDF files are supported"
    )

    if uploaded_file is not None:
        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

with col2:
    st.subheader("2. Evaluation Question")
    default_question = "Score this resume out of 100 for a Data Scientist role."
    question = st.text_area(
        "What would you like to evaluate?",
        value=default_question,
        height=120,
        help="Examples: Score for Data Scientist role, Suggest improvements for Product Manager, Check ATS friendliness, etc."
    )

# Prompt template (same structure as your notebook)
PROMPT_TEMPLATE = """
You are an advanced AI resume evaluation assistant. Evaluate the candidate's resume based on the provided context (Resume Content) and answer the user's specific request.

Evaluate the candidate and provide a score out of 100 based on this breakdown:

**Scoring Criteria (Total: 100 Points):**
1. **Skills & Technical Proficiency (40 Points):** Alignment with relevant hard/soft skills, tools, and technical domain depth.
2. **Experience & Achievements (30 Points):** Impact of previous roles, quantifiable accomplishments, and career progression.
3. **Education & Certifications (15 Points):** Relevant degrees, coursework, domain certifications, and continuous learning.
4. **Formatting, Clarity & Quality (15 Points):** Structural organization, readability, conciseness, and ATS friendliness.

---

**Response Format:**

### 1. Overall Score: [X]/100
* **Skills (40):** [Score]
* **Experience (30):** [Score]
* **Education (15):** [Score]
* **Formatting & Clarity (15):** [Score]

### 2. Key Strengths
* Highlight major accomplishments and strong skill matches.

### 3. Areas for Improvement
* Highlight missing skills, weak descriptions, or gaps.

### 4. Suggest what should be next career path

### 5. Answer to Specific Question / Feedback
[Provide detailed feedback based on the user's request]

---
**Resume Content:**
{context}

**User Query/Instructions:**
{question}

**Evaluation:**
"""


def extract_text_from_pdf(uploaded_file) -> str:
    """Save uploaded PDF to temp file and extract text using PyPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        context = "\n\n".join(doc.page_content for doc in documents)
        return context
    finally:
        # Clean up temp file
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


# Evaluate button
st.markdown("---")
evaluate_btn = st.button("🚀 Evaluate Resume", type="primary", use_container_width=True)

if evaluate_btn:
    # Validation
    if not api_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    if uploaded_file is None:
        st.error("❌ Please upload a PDF resume first.")
        st.stop()

    if not question.strip():
        st.error("❌ Please enter an evaluation question.")
        st.stop()

    # Processing
    with st.spinner("📄 Extracting text from PDF..."):
        try:
            context = extract_text_from_pdf(uploaded_file)
            if not context.strip():
                st.error("Could not extract any text from the PDF. Please try another file.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            st.stop()

    # Show extracted text in expander (useful for debugging)
    with st.expander("📝 View extracted resume text", expanded=False):
        st.text_area("Extracted Content", context, height=250, disabled=True)

    with st.spinner("🤖 Evaluating resume with Gemini... This may take a few seconds."):
        try:
            llm = get_llm(api_key, model_name, temperature, max_tokens)

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=PROMPT_TEMPLATE
            )

            chain = prompt | llm | StrOutputParser()

            # Stream the response
            response_placeholder = st.empty()
            full_response = ""

            for chunk in chain.stream({
                "context": context,
                "question": question
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Success message
            st.success("✅ Evaluation complete!")

            # Download button for the evaluation
            st.download_button(
                label="📥 Download Evaluation as TXT",
                data=full_response,
                file_name=f"resume_evaluation_{uploaded_file.name.replace('.pdf', '')}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
            st.info(
                "💡 Tips: Check that your API key is valid and has quota remaining. Also ensure the model name is correct.")

# Footer
st.markdown("---")
st.caption("This tool uses Google Gemini via LangChain. Your resume is processed temporarily and not stored.")