import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import seaborn as sns

## Page configuration
st.set_page_config(
    page_title="EuRepoC's Cyber Attack Model",
    page_icon="🤖",
    layout="wide"
)

## Loading model
@st.cache()
def loading_model():
    hg_model_hub_name = "CamilleBorrett/mdeberta-v3-base-nli-multiling-cyber-eurepoc-2"
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    return model

def load_tokenizer():
    hg_model_hub_name = "CamilleBorrett/mdeberta-v3-base-nli-multiling-cyber-eurepoc-2"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    return tokenizer

model = loading_model()
tokenizer = load_tokenizer()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

im1, im2, im3 = st.columns([14,10,10])
im2.image("logo.svg", width=300)
st.markdown("<h1 style='text-align: center;'>Identify cyber attacks in text using machine-learning</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ About", expanded=True):
    st.write(
        """     
 <b>Does a text mention a cyber attack?</b> <br>
 This machine-learning model enables you to scan a large pool of <b>multilingual</b> documents to identify those that mention cyber attacks. It is a Natural Language Inference (NLI) model, using mDeBERTa-v3-base fine-tuned on data collected in the context of the EuRepoC project.  
 <br>
 You can test the model using the app below: <ol>
 <li>Copy a paragraph of text (between 1 and 3 sentences). Note that your paragraph can be in any language.</li>
 <li>Click on the 'Run' button. A table will appear on the right displaying the model's prediction and confidence level.</li>
 </ol>
	    """, unsafe_allow_html=True
    )

st.markdown("")
col3, col4 = st.columns(2)
col3.markdown("<h4>📄 Paste your paragraph here</h4>", unsafe_allow_html=True)
col4.markdown("<h4>🤖 Prediction</h4>", unsafe_allow_html=True)
col4.markdown("Does the text mention a cyber attack?")
with col3.form(key="my_form"):
    premise = st.text_area(
        "Paste your paragraph of between 1 and 3 sentences below (max 200 words). It can be in any language.",
        height=100,
        value="A threat group believed to be sponsored by the Chinese government has breached the networks of U.S. state governments, including through the exploitation of a zero-day vulnerability."
        )

    MAX_WORDS = 200

    import re

    res = len(re.findall(r"\w+", premise))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + " Only the first 200 words will be reviewed."
        )

        premise = premise[:MAX_WORDS]

    submit_button = st.form_submit_button(label="✨Run")

if not submit_button:
    st.stop()

hypothesis = "A cyber attack happened"

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
labels_converted = {"Yes": "", "No": ""}
labels_converted["Yes"] = prediction["entailment"]
labels_converted["No"] = prediction["neutral"] + prediction["contradiction"]

df = pd.DataFrame()
df["Answer"] = labels_converted.keys()
df["Confidence level (%)"] = labels_converted.values()
df["Confidence level (%)"] = round(df["Confidence level (%)"],1)

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Confidence level (%)",
    ],
)

format_dictionary = {
    "Confidence level (%)": "{:.1f}",
}

df = df.format(format_dictionary)

col4.table(df)
col4.markdown("⚠️ <i>Please note that the model is still work in progress. Further training is planned in the near future.</i>", unsafe_allow_html=True)