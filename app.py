
import pickle
import streamlit as st
import PyPDF2
import docx2txt
st.title("Anishs Resume Category Prediction System")
def extract_text_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
     pdf_reader = PyPDF2.PdfReader(uploaded_file)
     text = ""
     for page in pdf_reader.pages:
        text += page.extract_text() or ''
     return text
    elif  uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
     text = docx2txt.process(uploaded_file)
     return text
    elif uploaded_file.type == "text/plain":
     text = str(uploaded_file.read(), 'utf-8')
     return text
    else:
      st.error("Unsupported file type.")
      return None
def predict_category(new_category):
    with open('trained_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('label_encoder.pkl', 'rb') as encoder_file:
        loaded_label_encoder = pickle.load(encoder_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    new_resume_vectorized = loaded_vectorizer.transform([new_category])  #transforming new category using tfidf vectorizer
    prediction = loaded_model.predict(new_resume_vectorized)  #prediction using loaded model
    decoded_prediction = loaded_label_encoder.inverse_transform(prediction) #decoding prediction to get orginal categorical label
    return decoded_prediction[0]
uploaded_file = st.file_uploader("Upload Your Resume file", type=["pdf", "docx", "txt"])
if st.button("Predict Category"):
    if uploaded_file is not None:
        resume_text = extract_text_file(uploaded_file)
        if resume_text:
            predicted_value = predict_category(resume_text)
            st.success(f"Predicted category for the resume: {predicted_value}")
        else:
            st.error("Couldnot extract text from the uploaded file.")
    else:
      st.error("Please upload a resume file.")
