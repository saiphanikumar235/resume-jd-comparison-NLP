import os

import numpy as np
import nltk

nltk.download('stopwords')
import pandas as pd
import PyPDF2, pdfplumber, nlp, re, docx2txt, streamlit as st, nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from nltk.corpus import stopwords
from pathlib import Path
from pyresparser import ResumeParser
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

os.system("python -m spacy download en_core_web_sm")
os.system("python -m nltk.downloader words")
os.system("python -m nltk.downloader stopwords")
nltk.download('punkt')


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def compare_jd(resume_text, jd):
    if jd != '':
        resume_tokens = word_tokenize(resume_text.lower())
        job_desc_tokens = word_tokenize(jd.lower())
        model = Word2Vec([resume_tokens, job_desc_tokens], vector_size=100, window=5, min_count=1, sg=0)
        resume_vector = np.mean([model.wv[token] for token in resume_tokens], axis=0)
        job_desc_vector = np.mean([model.wv[token] for token in job_desc_tokens], axis=0)
        MatchPercentage = cosine_similarity(resume_vector, job_desc_vector) * 100
        # Req_Clear = ''.join(open("./req.txt", 'r', encoding="utf8").readlines()).replace("\n", "")
        # jd_text = jd
        # Match_Test = [resume_text.lower(), jd_text.lower()]
        # cv = TfidfVectorizer()
        # count_matrix = cv.fit_transform(Match_Test)
        # MatchPercentage = cosine_similarity(count_matrix[0], count_matrix[1])
        # MatchPercentage = round(MatchPercentage[0][0]*100, 2)
        # print('Match Percentage is :' + str(MatchPercentage) + '% to Requirement')
        return MatchPercentage
    return "No JD to Compare"


def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return ','.join(list(set(r.findall(string))))


def get_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return ','.join(list(set([re.sub(r'\D', '', num) for num in phone_numbers])))


def get_education(path, resume_text):
    education_new = ResumeParser(path).get_extracted_data()
    education_new = education_new['degree']
    # if education_new is None:
    #     education_new = []
    #     nlp = spacy.load('en_core_web_sm')
    #     STOPWORDS = set(stopwords.words('english'))
    #     EDUCATION = [
    #         'ME', 'M.E', 'M.E.',
    #         'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
    #         'PHD', 'phd', 'ph.d', 'Ph.D.', 'MBA', 'mba', 'graduate', 'post-graduate', '5 year integrated masters',
    #         'masters', 'bachelor', "bachelor's"
    #     ]
    #     nlp_text = nlp(resume_text)
    #     nlp_text = [sent.string.strip().strip("\n") for sent in nlp_text.sents]
    #     for index, text in enumerate(nlp_text):
    #         for tex in text.split():
    #             for tex in text.split():
    #                 tex = re.sub(r'[?|$|.|!|,]', r'', tex)
    #                 if tex.upper() in EDUCATION and tex not in STOPWORDS:
    #                     education_new.append(tex)
    return ','.join(education_new) if education_new is not None else None


def get_current_location(resume_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)
    location_entities = []
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
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
        print(span.text)
        if span.text.lower() in [r.lower().replace("\n", "") for r in
                                              open('./linkedin skill', 'r', encoding="utf8").readlines()]:
            return re.sub(r'\d', '', get_email_addresses(resume_text).split('@')[0])
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

    return ','.join([word.capitalize() for word in set([word.lower() for word in skillset])])


def extract_certifications(resume_text):
    pattern = r"(?i)(certifications|certification|certified)(.*?)\n"

    # Use regular expression to find certification matches
    certification_matches = re.findall(pattern, resume_text)

    certifications = []
    for match in certification_matches:
        certifications.append(match[1].strip())

    return ','.join(certifications) if len(certifications) != 0 else "No Certifications found"


def get_exp(resume_text):
    words_to_numbers = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'zero': '0'
    }
    pattern = re.compile(r'\b(' + '|'.join(words_to_numbers.keys()) + r')\b')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "DATE" and "year" in ent.text.lower():
            years_of_experience = ent.text
            for y in years_of_experience.split():
                if y.lower() in words_to_numbers.keys() or y.replace('+', '').isnumeric():
                    years = f"{y.replace('+', '')}+"
                    return re.sub(pattern, lambda x: words_to_numbers[x.group()], years)
    for ent in doc.ents:
        if ent.label_ == 'CARDINAL':
            years_of_experience = ent.text
            for y in years_of_experience.split():
                if y.lower() in words_to_numbers.keys() or y.isnumeric():
                    years = f'{y}+'
                    return re.sub(pattern, lambda x: words_to_numbers[x.group()], years)
    return None


def get_details(resume_text, path):
    extracted_text = {"Name": extract_name(resume_text),
                      "E-Mail": get_email_addresses(resume_text),
                      "Phone Number": get_phone_numbers(resume_text),
                      'Skills': get_skills(resume_text),
                      'Experience': get_exp(resume_text),
                      'Education': get_education(path, resume_text),
                      'Approx current location': None,  # get_current_location(resume_text),
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
    resume_details = get_details(resume_text, uploaded_resume)
    resume_details['Resume-Score'] = compare_jd(resume_text, jd)
    resume_details['file-name'] = uploaded_resume.name
    total_files.append(
        resume_details
    )
if len(total_files) != 0:
    res_df = st.table(pd.DataFrame(total_files))
    st.download_button(
        "Click to Download",
        pd.DataFrame(total_files).to_csv(),
        "file.csv",
        "text/csv",
        key='download-csv'
    )
