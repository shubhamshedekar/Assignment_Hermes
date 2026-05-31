import streamlit as st
import pandas as pd
import numpy as np
import fitz
import json
from PIL import Image
from paddleocr import PaddleOCR
from groq import Groq

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Document Validator", layout="wide")

# Load Groq client (USE SECRETS)
client = Groq(api_key=st.secrets["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"])

ocr = PaddleOCR(use_angle_cls=True, lang="en")


# -----------------------------
# PDF -> IMAGE
# -----------------------------
def convert_pdf_to_image(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=300)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# -----------------------------
# IMAGE -> TEXT
# -----------------------------
def img_to_text(img):
    img_np = np.array(img)
    result = ocr.ocr(img_np)

    all_text = []
    for block in result:
        for line in block:
            all_text.append(line[1][0])

    return " ".join(all_text)


# -----------------------------
# GROQ EXTRACTION
# -----------------------------
def key_extraction_prompt(text):
    prompt = f"""
    Extract structured data from OCR text.

    Return ONLY valid JSON with keys:
    Shipper, Consignee, Product_Description, Quantity,
    Gross_Weight, Net_Weight, Packages, Invoice_Value

    If missing -> "".

    OCR TEXT:
    {text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


# -----------------------------
# VALIDATION
# -----------------------------
def validation_report(invoice_json, packing_json):
    prompt = f"""
    Compare Invoice vs Packing List JSON.

    Return JSON:
    {{
      "overall_status": "",
      "matched_fields": 0,
      "mismatched_fields": 0,
      "field_validation": [
        {{
          "field": "",
          "invoice_value": "",
          "packing_list_value": "",
          "status": "",
          "reason": ""
        }}
      ],
      "summary": ""
    }}

    Invoice:
    {invoice_json}

    Packing:
    {packing_json}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📄 Trade Document Validator (Invoice vs Packing List)")
st.write("Upload two PDFs and get validation report")

invoice_file = st.file_uploader("Upload Invoice PDF", type=["pdf"])
packing_file = st.file_uploader("Upload Packing List PDF", type=["pdf"])

if st.button("Run Validation"):

    if not invoice_file or not packing_file:
        st.error("Please upload both files")
    else:
        with st.spinner("Processing PDFs..."):
            invoice_img = convert_pdf_to_image(invoice_file)
            packing_img = convert_pdf_to_image(packing_file)

        with st.spinner("Running OCR..."):
            invoice_text = img_to_text(invoice_img)
            packing_text = img_to_text(packing_img)

        with st.spinner("Extracting structured data..."):
            invoice_json = key_extraction_prompt(invoice_text)
            packing_json = key_extraction_prompt(packing_text)

        with st.expander("📦 Invoice JSON"):
            st.json(invoice_json)

        with st.expander("📦 Packing List JSON"):
            st.json(packing_json)

        with st.spinner("Validating documents..."):
            report = validation_report(invoice_json, packing_json)

        st.success("Validation Complete")

        # -----------------------------
        # OUTPUT UI
        # -----------------------------
        st.subheader("📊 Summary")
        st.write(report["summary"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", report["overall_status"])
        col2.metric("Matched", report["matched_fields"])
        col3.metric("Mismatched", report["mismatched_fields"])

        st.subheader("🔍 Field-wise Validation")

        for item in report["field_validation"]:
            if item["status"] == "MATCH":
                st.success(f"✔ {item['field']}")
            else:
                st.error(f"✖ {item['field']}")

            st.write("Invoice:", item["invoice_value"])
            st.write("Packing:", item["packing_list_value"])
            st.write("Reason:", item["reason"])
            st.divider()