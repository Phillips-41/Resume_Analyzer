import streamlit as st
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

# Hard-code your API key here
GEMINI_API_KEY = ""

# LLM Configuration
MODEL_NAME = "models/gemini-3.5-flash"
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 2048


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📄 ATS Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare a resume against a Job Description using Gemini AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_llm():

    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    return model


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["job_description", "resume"],
    template="""
You are an ATS resume analyzer and technical recruiter.

Compare the candidate's resume with the Job Description.
Use only information present in the resume. Do not assume
skills or experience that are not mentioned.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

Provide the analysis in this format:

## 1. ATS Score
Give a score out of 100 and briefly explain why.

## 2. Matched Keywords
List the important JD keywords found in the resume.

## 3. Missing Keywords
List important JD keywords that are missing or weakly represented.

## 4. Skill Gap
Create a concise table:

| Requirement | Status | Gap |
|-------------|--------|-----|

Status: Strong / Partial / Missing.

## 5. Strengths
List the strongest areas matching the JD.

## 6. Improvements
Give practical suggestions for improving the resume,
including important keywords, skills, experience, and ATS formatting.

## 7. Bullet Improvements
Give 2-3 improved resume bullets using ONLY information
already present in the resume. Do not invent experience.

## 8. Final Recommendation
Give:
- Overall Fit: Excellent / Good / Moderate / Weak
- Estimated ATS Match: X%
- One short recommendation.

Keep the response concise and avoid unnecessary explanations.
"""
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("LLM Configuration")

    st.write(f"**Model:** `{MODEL_NAME}`")

    st.write(f"**Temperature:** `{TEMPERATURE}`")

    st.write(
        f"**Max Output Tokens:** `{MAX_OUTPUT_TOKENS}`"
    )

    st.divider()

    st.info(
        "The Gemini API key is configured by the application "
        "owner."
    )


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# JOB DESCRIPTION
# ------------------------------------------------------------

with col1:

    st.subheader("📋 Job Description")

    job_description = st.text_area(
        "Paste the Job Description",
        height=500,
        placeholder=(
            "Paste the complete job description here..."
        )
    )


# ------------------------------------------------------------
# RESUME
# ------------------------------------------------------------

with col2:

    st.subheader("📄 Candidate Resume")

    uploaded_resume = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        help="Upload the candidate's resume in PDF format."
    )

    if uploaded_resume:

        st.success(
            f"Uploaded: {uploaded_resume.name}"
        )

        st.caption(
            f"File size: "
            f"{uploaded_resume.size / 1024:.1f} KB"
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not job_description.strip():

        st.error(
            "Please enter the Job Description."
        )

        st.stop()


    if uploaded_resume is None:

        st.error(
            "Please upload a resume PDF."
        )

        st.stop()


    # --------------------------------------------------------
    # EXTRACT RESUME
    # --------------------------------------------------------

    with st.spinner("Reading resume PDF..."):

        try:

            resume_text = extract_resume_text(
                uploaded_resume
            )

        except Exception as e:

            st.error(
                f"Unable to read the PDF: {str(e)}"
            )

            st.stop()


    if not resume_text.strip():

        st.error(
            "No readable text was found in the PDF. "
            "The resume may be a scanned/image-only PDF."
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY EXTRACTED TEXT
    # --------------------------------------------------------

    with st.expander("🔍 View Extracted Resume Text"):

        st.text_area(
            "Resume Text",
            resume_text,
            height=400
        )


    # --------------------------------------------------------
    # CREATE LLM
    # --------------------------------------------------------

    try:

        model = get_llm()

    except Exception as e:

        st.error(
            f"Unable to initialize Gemini: {str(e)}"
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE CHAIN
    # --------------------------------------------------------

    chain = prompt | model


    # --------------------------------------------------------
    # RESULT SECTION
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 ATS Analysis")

    result_container = st.empty()

    full_response = ""

    try:

        for chunk in chain.stream(
                {
                    "job_description": job_description,
                    "resume": resume_text
                }
        ):

            content = chunk.content

            if isinstance(content, str):

                full_response += content

            elif isinstance(content, list):

                for item in content:

                    if isinstance(item, str):

                        full_response += item

                    elif isinstance(item, dict):

                        text = item.get("text")

                        if text:
                            full_response += text

            result_container.markdown(
                full_response
            )

        st.success("Analysis completed!")

        st.download_button(
            label="📥 Download Analysis",
            data=full_response,
            file_name="ATS_Resume_Analysis.md",
            mime="text/markdown",
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Error while analyzing resume: {str(e)}"
        )