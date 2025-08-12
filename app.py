# streamlit_resume_optimizer.py
# AI Resume Optimiser and Generator (Streamlit)
# Uses: pdfplumber, spaCy, python-docx, openai

import streamlit as st
import pdfplumber
import spacy
import re
import os
import io
import difflib
from docx import Document
import openai

# -------------------- CONFIG --------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Load spaCy model with auto-download if missing
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# -------------------- UTILITIES --------------------
def extract_text_from_pdf(file_bytes):
    """Extract text from uploaded PDF file bytes using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n\n".join(text_chunks)


def extract_keywords_spacy(text, top_k=30):
    """Extract candidate keywords from text using spaCy noun chunks and entities."""
    doc = nlp(text)
    candidates = []
    for chunk in doc.noun_chunks:
        token_text = chunk.text.strip()
        if len(token_text) > 2:
            candidates.append(token_text.lower())
    for ent in doc.ents:
        if len(ent.text) > 2:
            candidates.append(ent.text.lower())
    # frequency score
    freq = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, v in items[:top_k]]


def compute_keyword_overlap(resume_text, jd_text):
    resume_kw = set(extract_keywords_spacy(resume_text, top_k=200))
    jd_kw = set(extract_keywords_spacy(jd_text, top_k=200))
    matched = resume_kw & jd_kw
    missing = jd_kw - resume_kw
    return {
        "resume_keywords_count": len(resume_kw),
        "jd_keywords_count": len(jd_kw),
        "matched_count": len(matched),
        "matched": sorted(list(matched))[:50],
        "missing": sorted(list(missing))[:50]
    }


def build_prompt_for_optimization(resume_text, jd_text, instructions=None):
    base = (
        "You are an expert resume writer and ATS optimisation assistant.\n"
        "Given the candidate resume and the target job description below, produce an optimised, ATS-friendly resume.\n"
        "- Keep content truthful; do not invent qualifications.\n"
        "- Emphasise, rephrase, and reorder existing experiences to match the job description keywords.\n"
        "- Insert missing keywords only if they can be naturally applied.\n"
        "- Output the resume in plain text with headings: Contact, Summary, Skills, Experience, Education, Projects.\n"
        "- Provide a brief change log after the resume: list added keywords and reworded sections.\n\n"
    )
    if instructions:
        base += "Extra instructions: " + instructions + "\n\n"
    base += "---BEGIN RESUME---\n" + resume_text + "\n---END RESUME---\n\n"
    base += "---BEGIN JOB DESCRIPTION---\n" + jd_text + "\n---END JOB DESCRIPTION---\n\n"
    base += "Produce only the optimised resume and then the change log.\n"
    return base


def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=1200, temperature=0.2):
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not set.")
    client = openai.OpenAI(api_key=openai.api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def generate_docx_from_text(text):
    doc = Document()
    for line in text.splitlines():
        if line.strip() == "":
            doc.add_paragraph("")
        elif re.match(r"^(Contact|Summary|Skills|Experience|Education|Projects|Certifications):", line):
            doc.add_heading(line.strip(), level=2)
        else:
            doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio


def generate_change_summary(old_text, new_text):
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="", n=0)
    return "\n".join(diff)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="AI Resume Optimiser", layout="wide")
st.title("AI Resume Optimiser and Generator")
st.write("Upload a candidate resume (PDF) and paste a job description. The app will analyse keywords, call an LLM to produce an optimised resume, and show a change log.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-3.5-turbo"], index=0)
    openai_key_input = st.text_input("OpenAI API Key", type="password")
    if openai_key_input:
        openai.api_key = openai_key_input

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Paste job description here", height=300)
    instructions = st.text_area("Optional: Extra optimisation instructions", height=120)
    optimize_btn = st.button("Optimise Resume")

with col2:
    st.subheader("Parsed Resume Text")
    parsed_text_area = st.empty()

resume_text = ""
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
    parsed_text_area.code(resume_text[:10000] + ("..." if len(resume_text) > 10000 else ""))

if optimize_btn:
    if not resume_text:
        st.error("Please upload a resume PDF first.")
    elif not jd_text.strip():
        st.error("Please paste the target job description.")
    else:
        with st.spinner("Analysing and optimising..."):
            try:
                overlap = compute_keyword_overlap(resume_text, jd_text)

                st.subheader("Keyword Analysis")
                st.write(f"Resume keywords: {overlap['resume_keywords_count']}")
                st.write(f"Job description keywords: {overlap['jd_keywords_count']}")
                st.write(f"Matched keywords: {overlap['matched_count']}")
                st.write("Top matched keywords:")
                st.write(overlap['matched'])
                st.write("Top missing keywords from JD:")
                st.write(overlap['missing'])

                prompt = build_prompt_for_optimization(resume_text, jd_text, instructions=instructions)
                llm_out = call_openai_chat(prompt, model=model_choice)

                st.subheader("Optimised Resume (LLM Output)")
                st.code(llm_out[:10000] + ("..." if len(llm_out) > 10000 else ""))

                docx_bytes = generate_docx_from_text(llm_out)
                st.download_button("Download Optimised Resume (DOCX)", data=docx_bytes.getvalue(), file_name="optimised_resume.docx")

                st.subheader("Change Summary (diff)")
                change_summary = generate_change_summary(resume_text, llm_out)
                st.text_area("Diff between original and optimised resume", value=change_summary, height=300)

            except Exception as e:
                st.error(f"Error during optimisation: {e}")

st.markdown("---")
st.write("Notes:\n- Install dependencies: `pip install -r requirements.txt`\n- Download spaCy model: `python -m spacy download en_core_web_md`\n- Default model is `gpt-4o-mini` for wide availability.\n")
