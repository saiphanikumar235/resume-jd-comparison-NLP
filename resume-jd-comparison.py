import os

import numpy as np
import pandas as pd
import PyPDF2, pdfplumber, nlp, re, docx2txt, streamlit as st, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher
from nltk.corpus import stopwords
from pathlib import Path
from pyresparser import ResumeParser


def compare_jd(resume_text, jd):
    # Req_Clear = ''.join(open("./req.txt", 'r', encoding="utf8").readlines()).replace("\n", "")
    jd_text = jd
    Match_Test = [resume_text, jd_text]
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(Match_Test)
    MatchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    MatchPercentage = round(MatchPercentage, 2)
    # print('Match Percentage is :' + str(MatchPercentage) + '% to Requirement')
    return MatchPercentage


def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return list(set(r.findall(string)))


def get_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return list(set([re.sub(r'\D', '', num) for num in phone_numbers]))


def get_education(path):
    print(os.listdir())
    education_new = ResumeParser(path).get_extracted_data()
    print(education_new)
    return education_new['degree']


def get_current_location(resume_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)
    location_entities = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            location_entities.append(ent.text)
    if location_entities:
        return location_entities[0]
    return None


def extract_name(resume_text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(resume_text)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern], on_match=None)
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text


def get_skills(resume_text):
    nlp = spacy.load('en_core_web_sm')
    nlp_text = nlp(resume_text)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skills = [r.lower().replace("\n", "") for r in open('./linkedin skill', 'r', encoding="utf8").readlines()]
    skillset = []
    for i in tokens:
        if i.lower() in skills:
            skillset.append(i)
    for i in nlp_text.noun_chunks:
        i = i.text.lower().strip()
        if i in skills:
            skillset.append(i)

    return [word.capitalize() for word in set([word.lower() for word in skillset])]


def extract_certifications(resume_text):
    pattern = r"(?i)(certification|certifications|certified)(.*?)\n"

    # Use regular expression to find certification matches
    certification_matches = re.findall(pattern, resume_text)

    certifications = []
    for match in certification_matches:
        certifications.append(match[1].strip())

    return certifications


def get_exp(resume_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "DATE" and "year" in ent.text.lower():
            years_of_experience = ent.text.split()[0]
            return years_of_experience
    return None


def get_details(resume_text, path):
    extracted_text = {"Name": extract_name(resume_text),
                      "E-Mail": get_email_addresses(resume_text),
                      "Phone Number": get_phone_numbers(resume_text),
                      'Skills': get_skills(resume_text),
                      'Experience': get_exp(resume_text),
                      'Education': get_education(path),
                      'Approx current location': get_current_location(resume_text),
                      'certifications': extract_certifications(resume_text)
                      }
    return extracted_text


def read_pdf(file):
    save_path = Path('./', file.name)
    with open(save_path, mode='wb') as w:
        w.write(file.getvalue())
    resume_data = open(f'./{file.name}', 'rb')
    Script = PyPDF2.PdfReader(resume_data)
    pages = len(Script.pages)
    Script = []
    with pdfplumber.open(resume_data) as pdf:
        for i in range(0, pages):
            page = pdf.pages[i]
            text = page.extract_text()
            Script.append(text)
    Script = ''.join(Script)
    resume_data = Script.replace("\n", " ")
    return resume_data


def read_docx(file):
    my_text = docx2txt.process(file)
    return my_text


st.title("Resume and JD comparison")
jd = st.text_input('please enter the job description below:')
uploaded_resumes = st.file_uploader(
    "Upload a resume (PDF or Docx)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
total_files = []
for uploaded_resume in uploaded_resumes:
    if uploaded_resume.type == "application/pdf":
        resume_text = read_pdf(uploaded_resume)
    else:
        resume_text = read_docx(uploaded_resume)
    # resume_text = TextCleaner(resume_text).clean_text()
    resume_details = get_details(resume_text, uploaded_resume)
    resume_details['Resume-Score'] = compare_jd(resume_text, jd)
    resume_details['file-name'] = uploaded_resume.name
    total_files.append(
        resume_details
    )
if len(total_files) != 0:
    res_df = st.table(pd.DataFrame(total_files))
    st.download_button(
        "Press to Download",
        pd.DataFrame(total_files).to_csv(),
        "file.csv",
        "text/csv",
        key='download-csv'
    )
