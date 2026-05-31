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
# ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Add this instead:
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


# -----------------------------
# PDF -> IMAGE
# -----------------------------
def convert_pdf_to_image(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        page = doc[0]

        pix = page.get_pixmap(dpi=300)

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )
        return img
    except Exception as e:
        st.error(f"PDF conversion error: {e}")
        return None


# -----------------------------
# IMAGE -> TEXT
# -----------------------------
# def img_to_text(img):
#     try:
#         img_np = np.array(img)
#         result = ocr.ocr(img_np)

#         all_text = []
#         for block in result:
#             for line in block:
#                 all_text.append(line[1][0])

#         return " ".join(all_text)

#     except Exception as e:
#         st.error(f"OCR error: {e}")
#         return None

def img_to_text(img):
    try:
        ocr = load_ocr()  # gets cached instance
        img_np = np.array(img)
        result = ocr.ocr(img_np)

        all_text = []
        for block in result:
            if block:  # guard against None blocks
                for line in block:
                    all_text.append(line[1][0])

        return " ".join(all_text)

    except Exception as e:
        st.error(f"OCR error: {e}")
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
# VALIDATION (UPDATED LOGIC)
# -----------------------------

def validation_report(invoice_json, packing_list_json):
    try:
        prompt = f"""
        You are an expert Trade Document Validation Assistant.
        
        Your task is to compare the Invoice JSON and Packing List JSON and generate a structured validation report.
        
        ========================
        IMPORTANT RULE (STRICT)
        ========================
        
        You MUST:
        - Identify each field as MATCH or MISMATCH
        - DO NOT return counts
        - DO NOT summarize numbers
        
        Instead:
        - "matched_fields" = list of field names that MATCH
        - "mismatched_fields" = list of field names that MISMATCH
        
        Each field must appear in exactly one list.
        
        No duplicates allowed.
        
        ========================
        COMPARISON RULES (FIELD-WISE STRICT)
        ========================
        
        1. Shipper
        - Ignore punctuation differences (Ltd vs Ltd.)
        - Ignore case differences
        - Must match company name meaningfully
        - If core name differs → MISMATCH
        
        2. Consignee
        - Ignore punctuation (S/A vs S.A)
        - Ignore spacing and case differences
        - Must refer to same organization
        - If country/company root differs → MISMATCH
        
        3. Product_Description
        - MATCH if one contains the other AND refers to same product
        - Ignore prefixes like "MF Active Pharmaceutical Ingredients"
        - Focus on core product name
        
        4. Quantity
        - Remove units
        - Compare numeric values only
        - Allow minor decimal differences
        
        5. Gross_Weight
        - Compare numeric values only after removing units
        
        6. Net_Weight
        - Same rule as Gross_Weight
        
        7. Packages
        - Extract number only (e.g., "8 HDPE Drums" → 8)
        - Ignore text differences
        
        8. Invoice_Value
        - Remove formatting differences (commas, decimals if equal numerically)
        - Empty vs non-empty → MISMATCH
        
        ========================
        OUTPUT RULE (VERY IMPORTANT)
        ========================
        
        Return ONLY valid JSON.
        
        Rules:
        - First evaluate all fields
        - Then place field names into correct arrays
        - DO NOT output counts
        - DO NOT output numbers
        - DO NOT include explanations outside JSON structure
        
        ========================
        OUTPUT FORMAT
        ========================
        
        {{
          "overall_status": "PASS or FAIL",
          "matched_fields": [],
          "mismatched_fields": [],
          "field_validation": [
            {{
              "field": "",
              "invoice_value": "",
              "packing_list_value": "",
              "status": "MATCH or MISMATCH",
              "reason": ""
            }}
          ],
          "summary": "Write One line summary about missmatch and match"
        }}
        
        ========================
        INPUT
        ========================
        
        Invoice JSON:
        {invoice_json}
        
        Packing List JSON:
        {packing_list_json}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
# STREAMLIT UI (UNCHANGED)
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

        st.expander("📦 Invoice JSON").write(invoice_json)
        st.expander("📦 Packing JSON").write(packing_json)

        with st.spinner("Validating documents..."):
            report = validation_report(invoice_json, packing_json)

        st.success("Validation Complete")

        if not report:
            st.error("No report generated")
            st.stop()

        # -----------------------------
        # SAFE LIST HANDLING
        # -----------------------------
        matched_fields = report.get("matched_fields", [])
        mismatched_fields = report.get("mismatched_fields", [])

        # -----------------------------
        # SUMMARY
        # -----------------------------
        st.subheader("📊 Summary")
        st.write(report.get("summary", ""))

        # -----------------------------
        # METRICS (UPDATED)
        # -----------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", report.get("overall_status", "N/A"))
        col2.metric("Matched Fields", len(matched_fields))
        col3.metric("Mismatched Fields", len(mismatched_fields))

        # -----------------------------
        # LIST VIEW
        # -----------------------------
        st.subheader("✅ Matched Fields")
        st.write(matched_fields)

        st.subheader("❌ Mismatched Fields")
        st.write(mismatched_fields)

        # -----------------------------
        # FIELD VALIDATION
        # -----------------------------
        st.subheader("🔍 Field-wise Validation")

        for item in report.get("field_validation", []):

            if item.get("status") == "MATCH":
                st.success(f"🟢 {item.get('field')}")
            else:
                st.error(f"🔴 {item.get('field')}")

            st.write("Invoice:", item.get("invoice_value"))
            st.write("Packing:", item.get("packing_list_value"))
            st.write("Reason:", item.get("reason"))
            st.divider()

        # -----------------------------
        # DOWNLOAD REPORT
        # -----------------------------
        st.subheader("⬇ Download Report")

        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="validation_report.json",
            mime="application/json"
        )
