# 📄 AI Resume Analyzer

An AI-powered resume analysis application built with **Python, Streamlit, LangChain, and Google Gemini**.

The application allows users to upload a resume in PDF format and use Gemini to evaluate the resume, calculate an overall score, identify strengths and improvement areas, analyze skill gaps, and provide career recommendations.

---

## 🚀 Features

- 📄 Upload resumes in PDF format
- 🤖 AI-powered resume evaluation using Google Gemini
- 📊 Resume scoring out of 100
- 🎯 Job-role-specific resume evaluation
- 🔍 Skills and keyword analysis
- 📈 Experience and achievement evaluation
- 🎓 Education and certification assessment
- 📝 Formatting, clarity, and ATS-friendliness analysis
- 💡 Personalized resume improvement suggestions
- 🚀 Career-path recommendations
- ✉️ AI-powered cover letter generation
- 💼 Interactive AI Career Coach
- ⚡ Streaming AI responses
- 📥 Download generated analysis and cover letters
- 🧠 Conversational career coaching using resume context

---

# 🏗️ Application Modules

The project contains three main AI capabilities.

### 1. ATS Resume Analyzer

The ATS Resume Analyzer compares a candidate's resume against a target job description.

It evaluates:

```text
Resume
   │
   ▼
PDF Text Extraction
   │
   ▼
Google Gemini
   │
   ├── ATS Score
   ├── Matched Keywords
   ├── Missing Keywords
   ├── Skill Gaps
   ├── Strengths
   ├── Improvements
   ├── Improved Resume Bullets
   └── Final Recommendation
