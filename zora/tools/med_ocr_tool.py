"""
Med-OCR Tool — Tiered Medical Entity Extraction from PDFs
==========================================================

Tier 1 (primary):
    AWS Amazon Textract Medical (ComprehendMedical).
    Extracts structured entities: MEDICAL_CONDITION, MEDICATION,
    ANATOMY, TEST_TREATMENT_PROCEDURE, PROTECTED_HEALTH_INFORMATION,
    TIME_EXPRESSION.
    Maps entities to standard ontologies: ICD-10-CM, RxNorm, SNOMED.

Tier 2 (fallback):
    Groq LLaMA 3.3 70B — LLM-based extraction from raw PDF text
    when AWS credentials are not configured.

Tier 3 (last resort):
    Raw pdfplumber text chunks (existing behaviour).

Usage:
    result = med_ocr_tool(pdf_path="path/to/report.pdf", run_id="abc123")

Output:
    MedOCRResult with structured entity lists, diagnoses, medications,
    lab values, and clinical context.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import pdfplumber
from pydantic import BaseModel, Field


# ── Pydantic output schemas ───────────────────────────────────────────────────

class MedicalEntity(BaseModel):
    """A single extracted medical entity with optional ontology codes."""
    text: str                           # raw text span (e.g. "Type 2 Diabetes")
    category: str                       # MEDICAL_CONDITION | MEDICATION | ANATOMY |
                                        # TEST_TREATMENT_PROCEDURE | LAB_VALUE | PHI | OTHER
    confidence: float = 0.0            # 0.0-1.0
    icd10_code: Optional[str] = None   # e.g. "E11.9"
    rxnorm_code: Optional[str] = None  # medication code
    snomed_code: Optional[str] = None  # SNOMED CT concept
    normalized_name: Optional[str] = None   # canonical name from ontology
    attributes: list[dict] = Field(default_factory=list)  # e.g. [{Name: "NEGATION", Value: "TRUE"}]
    source_page: Optional[int] = None  # page number in PDF (1-indexed)


class MedOCRResult(BaseModel):
    """Full structured output from the Med-OCR pipeline for one PDF document."""
    source_file: str
    extraction_method: str             # "textract_medical" | "llm_groq" | "raw_text"
    entity_count: int = 0
    entities: list[MedicalEntity] = Field(default_factory=list)

    # Convenience groupings (populated by helper below)
    diagnoses: list[str] = Field(default_factory=list)     # MEDICAL_CONDITION texts
    medications: list[str] = Field(default_factory=list)   # MEDICATION texts
    anatomy: list[str] = Field(default_factory=list)       # ANATOMY texts
    procedures: list[str] = Field(default_factory=list)    # TEST_TREATMENT_PROCEDURE texts
    lab_values: list[str] = Field(default_factory=list)    # LAB_VALUE mentions
    phi_detected: bool = False                             # any PHI found?
    raw_text_chunks: list[str] = Field(default_factory=list)  # fallback text


# ── Internal helpers ─────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: str) -> tuple[str, list[str]]:
    """Extract full text and per-page chunks from a PDF using pdfplumber."""
    pages: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text.strip())
    except Exception as e:
        raise RuntimeError(f"pdfplumber failed on {pdf_path}: {e}") from e
    return "\n\n".join(pages), pages


def _populate_groups(result: MedOCRResult) -> MedOCRResult:
    """Fill the convenience group lists from entities list."""
    for ent in result.entities:
        cat = ent.category
        if cat == "MEDICAL_CONDITION":
            result.diagnoses.append(ent.text)
        elif cat == "MEDICATION":
            result.medications.append(ent.text)
        elif cat == "ANATOMY":
            result.anatomy.append(ent.text)
        elif cat in ("TEST_TREATMENT_PROCEDURE", "PROCEDURE"):
            result.procedures.append(ent.text)
        elif cat == "LAB_VALUE":
            result.lab_values.append(ent.text)
        elif cat == "PROTECTED_HEALTH_INFORMATION":
            result.phi_detected = True
    return result


# ── Tier 1: AWS Comprehend Medical (Textract Medical) ────────────────────────

def _tier1_textract_medical(full_text: str, source_file: str) -> MedOCRResult:
    """
    Call AWS Comprehend Medical detect_entities_v2.
    Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION env vars.
    Maps to ICD-10-CM via infer_icd10_cm and to RxNorm via infer_rx_norm.
    Raises RuntimeError if boto3 isn't installed or creds are missing.
    """
    import boto3  # type: ignore  # optional dependency

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client(
        "comprehendmedical",
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    entities: list[MedicalEntity] = []

    # AWS limits text to 20,000 UTF-8 bytes per request — chunk if needed
    MAX_BYTES = 19_000
    text_bytes = full_text.encode("utf-8")
    chunks = [
        text_bytes[i : i + MAX_BYTES].decode("utf-8", errors="ignore")
        for i in range(0, len(text_bytes), MAX_BYTES)
    ]

    for chunk in chunks:
        # ── Base entity detection ─────────────────────────────────────────
        resp = client.detect_entities_v2(Text=chunk)
        for ent in resp.get("Entities", []):
            entity = MedicalEntity(
                text=ent["Text"],
                category=ent["Category"],
                confidence=round(ent.get("Score", 0.0), 4),
                attributes=[
                    {"Name": a["Type"], "Value": a["Text"], "Score": round(a.get("Score", 0.0), 4)}
                    for a in ent.get("Attributes", [])
                ],
            )
            entities.append(entity)

    # ── ICD-10 mapping (diagnoses only) ──────────────────────────────────
    try:
        icd_resp = client.infer_icd10_cm(Text=full_text[:19_000])
        for icd_ent in icd_resp.get("Entities", []):
            matched_text = icd_ent["Text"].lower()
            codes = icd_ent.get("ICD10CMConcepts", [])
            best_code = codes[0]["Code"] if codes else None
            best_name = codes[0]["Description"] if codes else None
            # Attach code to matching entity
            for ent in entities:
                if ent.text.lower() == matched_text:
                    ent.icd10_code = best_code
                    ent.normalized_name = best_name
                    break
    except Exception:
        pass  # ICD mapping is best-effort

    # ── RxNorm mapping (medications only) ────────────────────────────────
    try:
        rx_resp = client.infer_rx_norm(Text=full_text[:19_000])
        for rx_ent in rx_resp.get("Entities", []):
            matched_text = rx_ent["Text"].lower()
            codes = rx_ent.get("RxNormConcepts", [])
            best_rx = codes[0]["Code"] if codes else None
            best_name = codes[0]["Description"] if codes else None
            for ent in entities:
                if ent.text.lower() == matched_text:
                    ent.rxnorm_code = best_rx
                    ent.normalized_name = best_name
                    break
    except Exception:
        pass  # RxNorm mapping is best-effort

    result = MedOCRResult(
        source_file=source_file,
        extraction_method="textract_medical",
        entity_count=len(entities),
        entities=entities,
    )
    return _populate_groups(result)


# ── Tier 2: LLM-based extraction (Groq LLaMA 3.3 70B) ───────────────────────
#   Cheaper and no AWS dependency. Returns the same MedOCRResult schema.

_LLM_EXTRACT_SYSTEM = """You are a medical NLP system. Extract all medical entities from clinical text.
Return ONLY a valid JSON object with this exact schema:
{
  "entities": [
    {
      "text": "<exact span from input>",
      "category": "<MEDICAL_CONDITION | MEDICATION | ANATOMY | TEST_TREATMENT_PROCEDURE | LAB_VALUE | PHI | OTHER>",
      "confidence": <0.0-1.0>,
      "icd10_code": "<ICD-10-CM code if known, else null>",
      "rxnorm_code": "<RxNorm code if medication and known, else null>",
      "normalized_name": "<canonical medical name from ICD-10 or RxNorm if known, else null>"
    }
  ]
}
Rules:
- Include ALL diagnoses, medications, lab values, procedures, anatomical locations, and PHI.
- PHI = patient name, DOB, MRN, address, phone, SSN.
- For common conditions assign correct ICD-10-CM: "Type 2 Diabetes" → "E11.9", "Hypertension" → "I10", etc.
- For medications assign correct RxNorm: "Metformin 500mg" → rxnorm_code "860975", etc.
- confidence = your certainty (0.0-1.0) that this entity exists in the text.
- Return valid JSON only. No prose. No markdown."""

def _tier2_llm_extraction(full_text: str, source_file: str) -> MedOCRResult:
    """
    LLM-based medical entity extraction using Groq LLaMA 3.3 70B.
    Chunks long documents (>6000 chars) and merges results.
    """
    from groq import Groq  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — cannot use LLM fallback.")

    client = Groq(api_key=api_key)
    CHUNK_SIZE = 5_500    # chars per chunk (well within LLaMA context)
    text_chunks = [full_text[i : i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]

    all_entities: list[MedicalEntity] = []

    for i, chunk in enumerate(text_chunks[:6]):   # max 6 chunks ≈ 33k chars
        prompt = f"Extract all medical entities from the following clinical text:\n\n{chunk}"
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": _LLM_EXTRACT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or "{}"

            # Strip markdown fences if present
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```")

            data = json.loads(raw)
            for ent_dict in data.get("entities", []):
                try:
                    all_entities.append(
                        MedicalEntity(
                            text=ent_dict.get("text", ""),
                            category=ent_dict.get("category", "OTHER"),
                            confidence=float(ent_dict.get("confidence", 0.7)),
                            icd10_code=ent_dict.get("icd10_code"),
                            rxnorm_code=ent_dict.get("rxnorm_code"),
                            normalized_name=ent_dict.get("normalized_name"),
                            source_page=i + 1,
                        )
                    )
                except Exception:
                    continue
        except json.JSONDecodeError:
            # If LLM didn't return valid JSON, skip this chunk gracefully
            continue
        except Exception as e:
            raise RuntimeError(f"LLM extraction call failed: {e}") from e

    result = MedOCRResult(
        source_file=source_file,
        extraction_method="llm_groq",
        entity_count=len(all_entities),
        entities=all_entities,
    )
    return _populate_groups(result)


# ── Tier 3: Raw text fallback ─────────────────────────────────────────────────

def _tier3_raw_text(full_text: str, raw_pages: list[str], source_file: str) -> MedOCRResult:
    """
    Last-resort: return raw pdfplumber text chunks.
    No entity extraction — preserves existing behaviour.
    """
    chunks = [p.strip() for p in raw_pages if p.strip()]
    return MedOCRResult(
        source_file=source_file,
        extraction_method="raw_text",
        entity_count=0,
        raw_text_chunks=chunks,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def med_ocr_tool(pdf_path: str, run_id: str) -> MedOCRResult:
    """
    Tiered Med-OCR pipeline.

    Priority:
      1. AWS Comprehend Medical (if AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY set)
      2. Groq LLaMA 3.3 70B   (if GROQ_API_KEY set)
      3. Raw pdfplumber text   (always available as last resort)

    Args:
        pdf_path: Absolute path to the PDF file.
        run_id:   Pipeline run identifier (for logging).

    Returns:
        MedOCRResult with structured entities + convenience groupings.
    """
    # Step 0: Extract raw text (needed for all tiers)
    full_text, raw_pages = _extract_pdf_text(pdf_path)

    source_file = os.path.basename(pdf_path)

    if not full_text.strip():
        # Completely empty PDF — return empty result
        return MedOCRResult(
            source_file=source_file,
            extraction_method="raw_text",
            entity_count=0,
        )

    # ── Tier 1: AWS Comprehend Medical ────────────────────────────────────
    aws_ready = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    if aws_ready:
        try:
            return _tier1_textract_medical(full_text, source_file)
        except ImportError:
            pass   # boto3 not installed — fall through
        except Exception as e:
            print(f"[med_ocr] Textract Medical failed ({e}) — falling back to LLM.")

    # ── Tier 2: Groq LLM extraction ───────────────────────────────────────
    groq_ready = bool(os.environ.get("GROQ_API_KEY"))
    if groq_ready:
        try:
            return _tier2_llm_extraction(full_text, source_file)
        except Exception as e:
            print(f"[med_ocr] LLM extraction failed ({e}) — falling back to raw text.")

    # ── Tier 3: Raw pdfplumber text ───────────────────────────────────────
    return _tier3_raw_text(full_text, raw_pages, source_file)
