import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm import safe_llm_invoke
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import re
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\srivi\\Downloads\\Tesseract-OCR\\tesseract.exe"

def process_transaction_image(image):
    img = image
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.LANCZOS)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = img.filter(ImageFilter.SHARPEN)

    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, lang="eng", config=config)

    # Fix currency symbols
    text = re.sub(r'(?<!\d)2(?=\d+\.\d)', '₹', text)
    text = re.sub(r'\bRs\.?\s*', '₹', text)
    text = re.sub(r'\bINR\s*', '₹', text)

    # Remove status bar noise
    text = re.sub(r'.*(KB/s|H\+|H\+\+|WiFi|signal).*\n?', '', text)

    # Remove garbage lines
    lines = text.splitlines()
    clean_lines = [line for line in lines if len(re.sub(r'[^a-zA-Z0-9₹$€£@.:\-]', '', line)) >= 3]

    return '\n'.join(clean_lines).strip()


def transaction_validator(image):

    text = process_transaction_image(image)

    SYSTEM_PROMPT = """You are a bank transaction receipt analyzer.

You will receive OCR-extracted text from a transaction screenshot.

REQUIRED fields (at least one of transaction_id, upi_id, or payment_id MUST be present, along with date and amount):
- date           : Any date found in the text (DD-MM-YYYY or YYYY-MM-DD)
- amount         : Amount with currency symbol (e.g. ₹2.0, $10.00)
- transaction_id : Unique transaction/reference ID (if present)
- upi_id         : UPI ID e.g. name@paytm (if present)
- payment_id     : Any other payment reference ID (if present)

OPTIONAL fields (extract only if clearly present):
- status         : Success | Failed | Pending
- account_number : Last 4 digits only
- payee_name     : Receiver or sender name
- timestamp      : Full timestamp in DD-MM-YYYY HH:MM:SS

VALIDATION RULE:
- If date AND amount AND at least one of (transaction_id, upi_id, payment_id) are found → valid
- Otherwise → invalid

If valid:
{{
  "valid": true,
  "fields": {{
    "amount": "₹2.0",
    "date": "23-05-2021",
    "transaction_id": "114302729857",
    "upi_id": "xxx977@paytm",
    "status": "Failed",
    "payee_name": "Raunak"
  }},
  "summary": "A payment of ₹2.0 to Raunak (xxx977@paytm) on 23-05-2021 failed due to timeout. Transaction ID: 114302729857."
}}

If invalid:
{{
  "valid": false,
  "error": "Missing required fields: <list the missing ones>"
  "summary": "This transaction could not be processed as the image does not contain a valid transaction screenshot. Missing required fields: <list the missing ones>"
}}

Return ONLY a valid JSON object. No explanation, no extra text. Do not include null fields.
"""

    prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Analyze this transaction OCR text:\n\n{text}")
    ])

    chain = prompt | safe_llm_invoke | StrOutputParser()

    result = chain.invoke({"text": text})

    try:
        
        clean = re.sub(r'```json|```', '', result).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "error": "LLM returned invalid JSON",
            "raw": result
        }