import logging
import zipfile
import xml.etree.ElementTree as ET
import re

logger = logging.getLogger(__name__)

def parse_docx_to_text(docx_path: str) -> str:
    try:
        with zipfile.ZipFile(docx_path) as z:
            data = z.read("word/document.xml")
    except Exception as e:
        logger.error(f"Failed to read .docx file {docx_path}: {e}")
        return ""
    root = ET.fromstring(data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = root.find("w:body", ns)
    if body is None:
        logger.error(f"No body found in .docx file {docx_path}")
        return ""
    paragraphs = []
    for p in body.findall("w:p", ns):
        parts = []
        for t in p.findall(".//w:t", ns):
            if t.text:
                parts.append(t.text)
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    joined = "\n".join(paragraphs)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined