import streamlit as st
from fastai.vision.all import load_learner, PILImage

st.header("Image Classifier")
page_description = """Bu model rasm taniy oladi"""
st.markdown(page_description)

model_file_path = "/home/beksulton/Downloads/Vscode/rasmni_tanish.pkl" 
model = None


try:
    model = load_learner(model_file_path)
except Exception as e:
    st.error(f"Modelni yuklashda xato: {e}")



uploaded_file = st.file_uploader("Rasm yuklash", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Yuklangan rasm.", use_column_width=True)

    img = PILImage.create(uploaded_file)

    if model:
        pred, pred_idx, probs = model.predict(img)
        st.success(f"Modelning bashorati: {pred}, ehtimoli: {probs[pred_idx]:.4f}")

if uploaded_file is None:
    st.info("Iltimos, rasm yuklang.")
