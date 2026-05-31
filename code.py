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

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
ocr = PaddleOCR(use_angle_cls=True, lang="en")


# -----------------------------
# PDF -> IMAGE
# -----------------------------
def convert_pdf_to_image(pdf_file):
  try:
    doc = fitz.open(pdf_file)
    page = doc[0]  # First page

    pix = page.get_pixmap(dpi=300)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )
    return img
  except Exception as e:
    print(f"Error converting PDF to image: {e}")
    return None


# -----------------------------
# IMAGE -> TEXT
# -----------------------------
def img_to_text(img):
  try:
    img_np = np.array(img)
    result = ocr.ocr(img_np)

    all_text = []

    for block in result:
        for line in block:
            text = line[1][0]
            all_text.append(text)

    # Join into one string
    final_text = " ".join(all_text)

    return final_text
  except Exception as e:
    print(f"Error converting image to text: {e}")
    return None


# -----------------------------
# GROQ EXTRACTION
# -----------------------------
def key_extraction_prompt(OCR):
  try:
    key_extraction_prompt = f"""
    You are an expert document information extraction system.

    Extract the following fields from the document:

    - Shipper
    - Consignee
    - Product_Description
    - Quantity
    - Gross_Weight
    - Net_Weight
    - Packages
    - Invoice_Value

    Rules:
    1. Return ONLY valid JSON.
    2. Use exactly these keys:
      "Shipper"
      "Consignee"
      "Product_Description"
      "Quantity"
      "Gross_Weight"
      "Net_Weight"
      "Packages"
      "Invoice_Value"
    3. If a value is not found, return an empty string "".
    4. Do not add explanations, markdown, or extra text.
    5. Extract the most relevant value even if labels vary slightly.

    Document OCR Text:
    {OCR}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": key_extraction_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content

    jsn = json.loads(result)
    return jsn
  except Exception as e:
    print(f"Error extracting keys: {e}")
    return None


# -----------------------------
# VALIDATION
# -----------------------------
def validation_report(invoice_json, packing_list_json):
  try:
    prompt = f"""
    You are an expert Trade Document Validation Assistant.

    Your task is to compare the Invoice JSON and Packing List JSON and generate a structured validation report.

    Comparison Rules:

    1. Compare the following fields:
      - Shipper
      - Consignee
      - Product_Description
      - Quantity
      - Gross_Weight
      - Net_Weight
      - Packages
      - Invoice_Value

    2. Consider minor formatting differences as MATCH:
      Examples:
      - "Ltd" vs "Ltd."
      - "S/A" vs "S.A"
      - Extra spaces
      - Uppercase/lowercase differences
      - Units attached to numbers
        Example:
          "200.000 KGS" == "200.00"
          "233.770 KGS" == "233.77"
          "8 HDPE Drums" ~= "8"

    3. For Product Description:
      - If one description contains the other and clearly refers to the same product,
        mark as MATCH.
      - Example:
          "MF Active Pharmaceutical Ingredients - Vildagliptin"
          and
          "Vildagliptin"
          should be considered MATCH.

    4. For numeric fields:
      - Compare numeric values after removing units.
      - Allow insignificant decimal formatting differences.

    5. Empty values:
      - If one document contains a value and the other is empty,
        mark as MISMATCH.

    6. Determine:
      - field status (MATCH / MISMATCH)
      - overall_status (PASS / FAIL)

    Return ONLY valid JSON in the following format:

    {{
      "overall_status": "PASS or FAIL",
      "matched_fields": <number>,
      "mismatched_fields": <number>,
      "field_validation": [
        {{
          "field": "",
          "invoice_value": "",
          "packing_list_value": "",
          "status": "MATCH or MISMATCH",
          "reason": ""
        }}
      ],
      "summary": ""
    }}

    Invoice JSON:
    {invoice_json}

    Packing List JSON:
    {packing_list_json}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content

    final_report = json.loads(result)

    return final_report
  except Exception as e:
    print(f"Error generating validation report: {e}")
    return None
  
  
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
        # SUMMARY
        # -----------------------------
        st.subheader("📊 Summary")
        st.write(report["summary"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", report["overall_status"])
        col2.metric("Matched", report["matched_fields"])
        col3.metric("Mismatched", report["mismatched_fields"])

        # -----------------------------
        # FIELD VALIDATION (COLOR UI)
        # -----------------------------
        st.subheader("🔍 Field-wise Validation")

        for item in report["field_validation"]:
            if item["status"] == "MATCH":
                st.success(f"🟢 ✔ {item['field']}")
            else:
                st.error(f"🔴 ✖ {item['field']}")

            st.write("Invoice:", item["invoice_value"])
            st.write("Packing:", item["packing_list_value"])
            st.write("Reason:", item["reason"])
            st.divider()

        # -----------------------------
        # DOWNLOAD JSON REPORT
        # -----------------------------
        st.subheader("⬇ Download Report")

        json_data = json.dumps(report, indent=4)

        st.download_button(
            label="Download JSON Report",
            data=json_data,
            file_name="validation_report.json",
            mime="application/json"
        )
