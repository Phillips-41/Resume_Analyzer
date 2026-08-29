import streamlit as st

from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = ""

MODEL_NAME = "models/gemini-3.7-flash"

TEMPERATURE = 0

MAX_OUTPUT_TOKENS = 1500


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career Coach",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_model():

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )


model = get_model()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n\n"

    return text


# ============================================================
# HEADER
# ============================================================

st.title("💼 AI Career Coach")

st.write(
    "Upload your resume and ask questions about your career, "
    "skills, jobs, interviews, and resume."
)


# ============================================================
# SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "system_message" not in st.session_state:
    st.session_state.system_message = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"]
    )

    if uploaded_file:

        # Process only when a new file is uploaded
        if (
            st.session_state.resume_text is None
            or st.session_state.get("file_name") != uploaded_file.name
        ):

            with st.spinner("Reading resume..."):

                resume_text = extract_resume_text(
                    uploaded_file
                )

            st.session_state.resume_text = resume_text
            st.session_state.file_name = uploaded_file.name

            # Reset conversation for new resume
            st.session_state.chat_history = []

            st.session_state.system_message = SystemMessage(
                content=f"""
You are a professional career coach and mentor.

Help the candidate with:
- Career guidance
- Resume improvement
- Job search strategy
- Interview preparation
- Skill-gap analysis
- Career planning
- Learning roadmap
- Job role recommendations

Use the candidate's resume as the primary source
when answering questions.

Do not invent skills, experience, education, projects,
or achievements that are not present in the resume.

If something is not mentioned in the resume, clearly
say that it is not mentioned.

Keep answers practical, concise and actionable.

CANDIDATE RESUME:

{resume_text}
"""
            )

            st.success("Resume loaded!")

    if st.session_state.resume_text:

        st.divider()

        st.write(
            f"**Resume:** {st.session_state.file_name}"
        )

        if st.button("🗑️ Clear Resume"):

            st.session_state.resume_text = None
            st.session_state.chat_history = []
            st.session_state.system_message = None
            st.session_state.file_name = None

            st.rerun()


# ============================================================
# CHECK RESUME
# ============================================================

if not st.session_state.resume_text:

    st.info(
        "👈 Upload your resume from the sidebar to start "
        "your career coaching session."
    )

    st.stop()


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.subheader("💡 Try asking")

suggestions = [
    "What are my strongest skills?",
    "What career should I choose?",
    "What skills am I missing?",
    "Am I ready for a Java Developer role?",
    "How can I improve my resume?",
    "What should I learn next?"
]


cols = st.columns(3)

for i, question in enumerate(suggestions):

    with cols[i % 3]:

        if st.button(
            question,
            use_container_width=True
        ):

            st.session_state.pending_question = question


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.markdown(message.content)


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something about your career..."
)


# Handle suggested question
if "pending_question" in st.session_state:

    user_input = st.session_state.pending_question

    del st.session_state.pending_question


# ============================================================
# CHAT PROCESSING
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    user_message = HumanMessage(
        content=user_input
    )

    st.session_state.chat_history.append(
        user_message
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------------------------
    # Build messages
    # --------------------------------------------------------

    messages = [
        st.session_state.system_message
    ] + st.session_state.chat_history


    # --------------------------------------------------------
    # Stream AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        response_text = ""

        try:

            for chunk in model.stream(messages):

                content = chunk.content

                # Gemini can return string
                if isinstance(content, str):

                    response_text += content

                # Gemini can also return list
                elif isinstance(content, list):

                    for item in content:

                        if isinstance(item, str):

                            response_text += item

                        elif isinstance(item, dict):

                            text = item.get("text")

                            if text:
                                response_text += text

                response_placeholder.markdown(
                    response_text
                )


            # ------------------------------------------------
            # Save AI response
            # ------------------------------------------------

            st.session_state.chat_history.append(
                AIMessage(
                    content=response_text
                )
            )


        except Exception as e:

            st.error(
                f"Error while generating response: {str(e)}"
            )