#!/usr/bin/env python3
"""
Antibody data extraction pipeline using DSPy — fully standalone.

Extracts structured antibody information from PDF papers using:
  1. Docling for PDF -> Markdown conversion
  2. DSPy + LLM for structured data extraction
  3. DSPy judge for evaluating extraction quality

Output fields per antibody:
  Name, VH, VL,
  HIC Retention Time (Min),
  Phenyl Sepharose Retention Time (Min),
  Butyl Sepharose Retention Time (Min),
  Octyl Sepharose Retention Time (Min),
  SMAC Retention Time (Min),
  CIC Retention Time (Min)

All HIC variants are extracted:
  "hydrophobic interaction chromatography", HIC, "hydrophobic interaction",
  "Phenyl Sepharose", "Butyl Sepharose", "Octyl Sepharose",
  "hydrophobic chromatography"

Usage:
    # Single PDF
    python extract_antibodies.py paper.pdf

    # Directory of PDFs, custom output and model
    python extract_antibodies.py papers/ --output results/ --model gpt-4o

    # Separate judge model, no GPU
    python extract_antibodies.py paper.pdf \\
        --model gemini-2.0-flash --judge-model claude-sonnet-4.6 --no-gpu

    # Skip judge (faster)
    python extract_antibodies.py paper.pdf --no-judge

Available models (defined in LLM_REGISTRY below):
    gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro,
    claude-sonnet-4.6, gpt-4o, gpt-4o-mini, gpt-4.1,
    mistral-small, mistral-medium, mistral-large, deepseek-v3.2

Dependencies (install via the repo's pyproject.toml or pip):
    dspy>=2.6, docling>=2.31, docling-core, pydantic>=2, python-dotenv
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import logging
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional: load .env from the repo root two levels up (if present)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=True)
except ImportError:
    pass  # rely on environment variables being set manually

# ---------------------------------------------------------------------------
# Silence noisy loggers
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
for _logger_name in ("pydantic", "LiteLLM", "litellm", "httpx", "docling"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("antibody_extraction")


# ===========================================================================
# SECTION 1 — LLM REGISTRY
# Inlined from src/llm_synthesis/utils/llms.py
# ===========================================================================


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for one LLM provider / model."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    extra_kwargs: dict | None = None


@dataclass(frozen=True)
class LLMRegistry:
    configs: dict[str, LLMConfig]


LLM_REGISTRY = LLMRegistry(
    configs={
        "gemini-2.0-flash": LLMConfig(model="gemini/gemini-2.0-flash"),
        "gemini-2.5-flash": LLMConfig(model="gemini/gemini-2.5-flash"),
        "gemini-2.5-flash-lite": LLMConfig(
            model="gemini/gemini-2.5-flash-lite",
            extra_kwargs={"thinking": {"type": "enabled"}},
        ),
        "gemini-2.5-pro": LLMConfig(model="gemini/gemini-2.5-pro"),
        "gemini-3.0-pro": LLMConfig(model="gemini/gemini-3-pro-preview"),
        "gemini-3.0-flash": LLMConfig(model="gemini/gemini-3-flash-preview"),
        "gemini-3.0-flash-lite": LLMConfig(
            model="gemini/gemini-3-flash-lite",
            extra_kwargs={"thinking": {"type": "enabled"}},
        ),
        "claude-sonnet-4.6": LLMConfig(model="anthropic/claude-sonnet-4-6"),
        "gemini-3-flash": LLMConfig(
            model="gemini/gemini-3-flash-preview",
            extra_kwargs={"reasoning_effort": "disable"},
        ),
        "qwen3.5-35b-a3b": LLMConfig(
            model="openrouter/qwen/qwen3.5-35b-a3b",
            api_key=os.getenv("OPENROUTER_QWEN_API_KEY"),
            api_base="https://openrouter.ai/api/v1",
            extra_kwargs={"enable_thinking": False},
        ),
        "kimi-k2.5": LLMConfig(
            model="openrouter/moonshotai/kimi-k2.5",
            api_key=os.getenv("OPENROUTER_KIMI_API_KEY"),
            api_base="https://openrouter.ai/api/v1",
            extra_kwargs={"enable_thinking": False},
        ),
        "qwen3.5-397b-a17b": LLMConfig(
            model="openrouter/qwen/qwen3.5-397b-a17b",
            api_key=os.getenv("OPENROUTER_QWEN_API_KEY"),
            api_base="https://openrouter.ai/api/v1",
            extra_kwargs={"enable_thinking": False},
        ),
        "deepseek-v3.2": LLMConfig(
            model="openrouter/deepseek/deepseek-v3.2",
            api_key=os.getenv("OPENROUTER_DEEPSEEK_API_KEY"),
            api_base="https://openrouter.ai/api/v1",
        ),
        "gpt-4o": LLMConfig(model="openai/gpt-4o"),
        "gpt-4o-mini": LLMConfig(model="openai/gpt-4o-mini"),
        "gpt-o4-mini": LLMConfig(model="openai/o4-mini-2025-04-16"),
        "gpt-o3-mini": LLMConfig(model="openai/o3-mini-2025-01-31"),
        "gpt-4.1": LLMConfig(model="openai/gpt-4.1-2025-04-14"),
        "mistral-small": LLMConfig(
            model="openai/mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            api_base="https://api.mistral.ai/v1/",
        ),
        "mistral-medium": LLMConfig(
            model="openai/mistral-medium-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            api_base="https://api.mistral.ai/v1/",
        ),
        "mistral-large": LLMConfig(
            model="openai/mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            api_base="https://api.mistral.ai/v1/",
        ),
    }
)


def _extract_cost_from_dspy_response(response: Any) -> float | None:
    """Extract USD cost from a DSPy LM history entry (if available)."""
    try:
        if hasattr(dspy.settings, "lm") and hasattr(dspy.settings.lm, "history"):
            history = dspy.settings.lm.history
            if history:
                last = history[-1]
                if isinstance(last, dict) and "cost" in last:
                    cost = last["cost"]
                    if cost is not None:
                        return float(cost)
    except (AttributeError, TypeError, ValueError):
        pass
    return None


class SystemPrefixedLM(dspy.LM):
    """
    Wraps dspy.LM to inject a fixed system prompt on every call and
    accumulate cumulative cost (USD) via LiteLLM's usage data.

    Inlined from src/llm_synthesis/utils/llms.py.
    """

    def __init__(self, system_prompt: str, model: str, **kwargs: Any):
        super().__init__(model, **kwargs)
        self._system_prompt = system_prompt
        self._cumulative_cost_usd: float = 0.0

    def get_cost(self) -> float:
        return self._cumulative_cost_usd

    def reset_cost(self) -> float:
        prev, self._cumulative_cost_usd = self._cumulative_cost_usd, 0.0
        return prev

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **override_kwargs: Any,
    ):
        if messages is None:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt or ""})
        elif self._system_prompt:
            messages = [{"role": "system", "content": self._system_prompt}, *messages]

        response = super().__call__(messages=messages, **override_kwargs)

        try:
            cost = _extract_cost_from_dspy_response(response)
            if cost is not None:
                self._cumulative_cost_usd += cost
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Cost accumulation failed: %r", exc)

        return response


# ===========================================================================
# SECTION 2 — DSPy HELPERS
# Inlined from src/llm_synthesis/utils/dspy_utils.py
# ===========================================================================


def get_llm_from_name(
    llm_name: str,
    model_kwargs: dict | None = None,
    system_prompt: str | None = None,
) -> SystemPrefixedLM:
    """
    Instantiate a SystemPrefixedLM from a registry key.

    Args:
        llm_name: Key in LLM_REGISTRY (e.g. "gemini-2.0-flash").
        model_kwargs: Extra kwargs forwarded to the LM (temperature, etc.).
        system_prompt: Fixed system prompt injected on every call.

    Returns:
        A SystemPrefixedLM instance with cost tracking.
    """
    model_kwargs = dict(model_kwargs) if model_kwargs else {}
    try:
        cfg: LLMConfig = LLM_REGISTRY.configs[llm_name]
    except KeyError:
        available = sorted(LLM_REGISTRY.configs)
        raise ValueError(
            f"LLM {llm_name!r} not in registry. Available: {available}"
        )

    if cfg.api_key:
        model_kwargs["api_key"] = cfg.api_key
        model_kwargs["api_base"] = cfg.api_base

    if cfg.extra_kwargs:
        model_kwargs.update(cfg.extra_kwargs)

    return SystemPrefixedLM(system_prompt or "", cfg.model, **model_kwargs)


# ===========================================================================
# SECTION 3 — PDF EXTRACTION
# Inlined from src/llm_synthesis/transformers/pdf_extraction/docling_pdf_extractor.py
# ===========================================================================


class DoclingPDFExtractor:
    """
    Converts a PDF (raw bytes) to Markdown using Docling.

    Inlined from src/llm_synthesis/transformers/pdf_extraction/docling_pdf_extractor.py.
    """

    def __init__(
        self,
        pipeline: str = "standard",
        table_mode: str = "accurate",
        add_page_images: bool = False,
        use_gpu: bool = True,
        scale: float = 2.0,
    ):
        self.pipeline = pipeline
        self.table_mode = table_mode
        self.add_page_images = add_page_images
        self.use_gpu = use_gpu
        self.scale = scale

    def forward(self, pdf_bytes: bytes) -> str:
        """Return the PDF content as Markdown with embedded figures."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.io import DocumentStream

        opts = PdfPipelineOptions(
            pipeline=self.pipeline,
            table_mode=self.table_mode,
            generate_picture_images=True,
            generate_page_images=self.add_page_images,
            images_scale=self.scale,
            ocr=True,
            batch_size=4 if self.use_gpu else 1,
        )
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = conv.convert(
            DocumentStream(name="pdf", stream=io.BytesIO(pdf_bytes))
        )
        return result.document.export_to_markdown(image_mode="embedded")


# ===========================================================================
# SECTION 4 — DATA MODELS
# ===========================================================================


class AntibodyEntry(BaseModel):
    """Structured record for one antibody extracted from a paper."""

    name: str = Field(
        ...,
        description=(
            "Antibody name, ID, or label as reported in the paper "
            "(e.g. 'mAb-A', 'IgG1-001', 'Ab3', 'Antibody 3')."
        ),
    )
    vh: str | None = Field(
        None,
        description=(
            "VH (heavy-chain variable domain) amino-acid sequence, clone ID, "
            "or gene designation. Null if not reported."
        ),
    )
    vl: str | None = Field(
        None,
        description=(
            "VL (light-chain variable domain) amino-acid sequence, clone ID, "
            "or gene designation. Null if not reported."
        ),
    )

    # ---- HIC / hydrophobic interaction chromatography ----------------------
    hic_retention_time_min: float | None = Field(
        None,
        description=(
            "Generic HIC (Hydrophobic Interaction Chromatography) retention "
            "time in minutes when no specific resin is named. "
            "Null if not reported."
        ),
    )
    phenyl_sepharose_retention_time_min: float | None = Field(
        None,
        description=(
            "Retention time on a Phenyl Sepharose column (any grade: HP, FF, "
            "6 Fast Flow, etc.) in minutes. Null if not reported."
        ),
    )
    butyl_sepharose_retention_time_min: float | None = Field(
        None,
        description=(
            "Retention time on a Butyl Sepharose column (any grade) in "
            "minutes. Null if not reported."
        ),
    )
    octyl_sepharose_retention_time_min: float | None = Field(
        None,
        description=(
            "Retention time on an Octyl Sepharose column (any grade) in "
            "minutes. Null if not reported."
        ),
    )
    hic_resin: str | None = Field(
        None,
        description=(
            "Name of the HIC stationary phase / resin used, exactly as stated "
            "in the paper (e.g. 'Phenyl Sepharose 6 FF', 'Butyl-650S', "
            "'Toyopearl Phenyl-650M'). Captures resin types not covered by "
            "the dedicated columns above. Null if not specified."
        ),
    )

    # ---- Other chromatography assays ---------------------------------------
    smac_retention_time_min: float | None = Field(
        None,
        description=(
            "SMAC (Self-interaction Chromatography or similar self-association "
            "assay) retention time in minutes. Null if not reported."
        ),
    )
    cic_retention_time_min: float | None = Field(
        None,
        description=(
            "CIC (Charge Interaction / Cation-exchange interaction "
            "Chromatography) retention time in minutes. Null if not reported."
        ),
    )


class AntibodyTable(BaseModel):
    """All antibody records extracted from a single paper."""

    antibodies: list[AntibodyEntry] = Field(
        default_factory=list,
        description=(
            "All antibody records found in the paper. "
            "Empty list if no antibody data is present."
        ),
    )


# ===========================================================================
# SECTION 5 — JUDGE DATA MODELS
# ===========================================================================


class AntibodyExtractionScores(BaseModel):
    """Per-criterion quality scores (scale 1 = very poor … 5 = excellent)."""

    completeness_score: float = Field(
        ...,
        description=(
            "Score 1-5: how completely all antibody records and their fields "
            "were captured relative to what the paper actually reports."
        ),
        ge=1.0,
        le=5.0,
    )
    completeness_reasoning: str = Field(..., description="Reasoning for completeness score.")

    accuracy_score: float = Field(
        ...,
        description=(
            "Score 1-5: numeric and textual accuracy of the extracted values "
            "compared to the source text."
        ),
        ge=1.0,
        le=5.0,
    )
    accuracy_reasoning: str = Field(..., description="Reasoning for accuracy score.")

    source_grounding_score: float = Field(
        ...,
        description=(
            "Score 1-5: every extracted value can be directly traced to the "
            "source text (no hallucinated values)."
        ),
        ge=1.0,
        le=5.0,
    )
    source_grounding_reasoning: str = Field(
        ..., description="Reasoning for source grounding score."
    )

    format_compliance_score: float = Field(
        ...,
        description=(
            "Score 1-5: adherence to the AntibodyEntry schema — correct types, "
            "null for absent values, retention times in minutes, "
            "HIC resin name captured in the right field."
        ),
        ge=1.0,
        le=5.0,
    )
    format_compliance_reasoning: str = Field(
        ..., description="Reasoning for format compliance score."
    )

    overall_score: float = Field(
        default=0.0,
        description="Mean of all four criterion scores (auto-computed).",
        ge=0.0,
        le=5.0,
    )
    overall_reasoning: str = Field(
        default="",
        description="High-level summary of extraction quality.",
    )


class AntibodyExtractionEvaluation(BaseModel):
    """Complete quality evaluation for one paper's antibody extraction."""

    reasoning: str = Field(
        ...,
        description=(
            "Overall analysis comparing the extracted antibody table against "
            "the source paper text."
        ),
    )
    scores: AntibodyExtractionScores
    confidence_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Judge's confidence in this evaluation.",
    )
    missing_antibodies: list[str] = Field(
        default_factory=list,
        description=(
            "Names / IDs of antibodies clearly present in the paper but "
            "absent from the extracted table."
        ),
    )
    incorrect_values: list[str] = Field(
        default_factory=list,
        description="Specific field values that differ from what the paper states.",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions for improving the extraction.",
    )


# ===========================================================================
# SECTION 6 — DSPy SIGNATURES
# ===========================================================================


class AntibodyExtractionSignature(dspy.Signature):
    """
    You are a biomedical data extraction expert specialising in antibody
    biophysical characterisation.

    Extract ALL antibody records from the paper into a structured table.

    ── HIC / hydrophobic chromatography fields ─────────────────────────────
    Look for ALL of the following synonyms and resin names:
      • "hydrophobic interaction chromatography", "HIC",
        "hydrophobic interaction", "hydrophobic chromatography"
      • "Phenyl Sepharose" (any grade: HP, FF, 6 Fast Flow …)
      • "Butyl Sepharose" (any grade)
      • "Octyl Sepharose" (any grade)
      • Any other hydrophobic stationary phase (Toyopearl Phenyl, Butyl-650S …)

    Fill the most specific field available:
      - Phenyl Sepharose  → phenyl_sepharose_retention_time_min
      - Butyl Sepharose   → butyl_sepharose_retention_time_min
      - Octyl Sepharose   → octyl_sepharose_retention_time_min
      - Unspecified HIC   → hic_retention_time_min
      - Other named resin → hic_resin (text) + hic_retention_time_min (value)

    ── Other fields ─────────────────────────────────────────────────────────
      • SMAC (Self-interaction / self-association chromatography)
        → smac_retention_time_min
      • CIC (Charge interaction / cation-exchange interaction chromatography)
        → cic_retention_time_min
      • VH / VL domain sequences or gene designations
        → vh / vl

    ── Rules ────────────────────────────────────────────────────────────────
      1. Search tables, figure captions, methods, and supplementary sections.
      2. Convert all retention times to MINUTES (60 s = 1 min).
      3. Set null — NOT 0 — when a value is genuinely absent from the paper.
      4. Do NOT hallucinate values not present in the text.
      5. List every antibody that has at least one chromatography measurement.
    """

    paper_text: str = dspy.InputField(
        description=(
            "Full text of the scientific paper in Markdown format, "
            "including tables and figure captions."
        )
    )
    antibody_table: AntibodyTable = dspy.OutputField(
        description=(
            "Structured table of all antibody records found in the paper, "
            "with every available chromatography retention time and VH/VL data."
        )
    )


class AntibodyJudgeSignature(dspy.Signature):
    """
    You are an expert evaluator for antibody biophysical data extraction.

    Assess how accurately the extracted antibody table captures the
    information in the source paper.

    ── Evaluation principles ────────────────────────────────────────────────
      • Null fields are CORRECT when the paper does not report the value.
        Do NOT penalise missing values that are genuinely absent.
      • Penalise hallucinated values (values not traceable to the paper).
      • Penalise missed antibody records that are clearly present.
      • Penalise wrong resin assignment (e.g. Butyl placed in Phenyl field).
      • Verify all retention times are in minutes; flag unit errors.
      • Populate ALL score and reasoning fields — an incomplete response is invalid.
    """

    source_text: str = dspy.InputField(
        description="Original paper text (Markdown) used for extraction."
    )
    extracted_table_json: str = dspy.InputField(
        description="JSON of the extracted AntibodyTable."
    )
    evaluation: AntibodyExtractionEvaluation = dspy.OutputField(
        description=(
            "Comprehensive evaluation of extraction quality. "
            "Scores 1 (very poor) to 5 (excellent). "
            "REQUIRED: reasoning, confidence_level, all four *_score + "
            "*_reasoning pairs in scores, and overall_reasoning."
        )
    )


# ===========================================================================
# SECTION 7 — EXTRACTOR MODULE
# ===========================================================================


class AntibodyExtractor(dspy.Module):
    """
    DSPy module that extracts structured antibody data from paper text.

    Uses ChainOfThought for step-by-step reasoning.
    Retries at escalating temperatures on failure.
    """

    def __init__(
        self,
        lm: dspy.LM,
        retry_temperatures: list[float] | None = None,
    ):
        super().__init__()
        self.lm = lm
        self.retry_temperatures = retry_temperatures or [0.0, 0.2, 0.5]
        self.predict = dspy.ChainOfThought(AntibodyExtractionSignature)

    def forward(self, paper_text: str) -> AntibodyTable:
        """Extract antibody data with temperature-escalation retry."""
        last_exc: Exception | None = None

        for t_idx, temp in enumerate(self.retry_temperatures):
            lm = copy.copy(self.lm)
            lm.kwargs = {**self.lm.kwargs, "temperature": temp}
            try:
                with dspy.settings.context(lm=lm, adapter=dspy.adapters.JSONAdapter()):
                    result = self.predict(paper_text=paper_text)
                    table: AntibodyTable = result.antibody_table
                    logger.info(
                        "Extracted %d antibody record(s) (temp=%.1f)",
                        len(table.antibodies),
                        temp,
                    )
                    return table
            except Exception as exc:
                last_exc = exc
                if t_idx < len(self.retry_temperatures) - 1:
                    logger.warning(
                        "Extraction failed at temp=%.1f: %r — retrying", temp, exc
                    )
                else:
                    logger.warning("All temperatures exhausted: %r", exc)

        logger.error("Extraction failed permanently: %r", last_exc)
        return AntibodyTable(antibodies=[])


# ===========================================================================
# SECTION 8 — JUDGE MODULE
# ===========================================================================


class AntibodyJudge(dspy.Module):
    """
    DSPy module that evaluates antibody extraction quality against the source
    paper text.

    Scores completeness, accuracy, source grounding, and format compliance
    on a 1–5 scale. Retries at escalating temperatures on failure.
    """

    def __init__(
        self,
        lm: dspy.LM,
        retry_temperatures: list[float] | None = None,
    ):
        super().__init__()
        self.lm = lm
        self.retry_temperatures = retry_temperatures or [0.0, 0.3, 0.5]
        self.predict = dspy.Predict(AntibodyJudgeSignature)

    def forward(
        self,
        source_text: str,
        extracted_table: AntibodyTable,
    ) -> AntibodyExtractionEvaluation:
        """Evaluate extraction quality with temperature-escalation retry."""
        extracted_json = extracted_table.model_dump_json(indent=2)
        last_exc: Exception | None = None

        for t_idx, temp in enumerate(self.retry_temperatures):
            lm = copy.copy(self.lm)
            lm.kwargs = {**self.lm.kwargs, "temperature": temp}
            try:
                with dspy.settings.context(lm=lm, adapter=dspy.adapters.JSONAdapter()):
                    result = self.predict(
                        source_text=source_text,
                        extracted_table_json=extracted_json,
                    )
                    evaluation: AntibodyExtractionEvaluation = result.evaluation
                    evaluation = _post_process_evaluation(evaluation)
                    logger.info(
                        "Judge overall score: %.1f/5.0 (temp=%.1f)",
                        evaluation.scores.overall_score,
                        temp,
                    )
                    return evaluation
            except Exception as exc:
                last_exc = exc
                if t_idx < len(self.retry_temperatures) - 1:
                    logger.warning(
                        "Judge failed at temp=%.1f: %r — retrying", temp, exc
                    )
                else:
                    logger.warning("Judge exhausted all temperatures: %r", exc)

        logger.error("Judge failed permanently: %r", last_exc)
        return _fallback_evaluation()


def _post_process_evaluation(ev: AntibodyExtractionEvaluation) -> AntibodyExtractionEvaluation:
    """Clamp scores to [1, 5] and compute the overall mean."""
    s = ev.scores
    criterion_fields = [
        "completeness_score",
        "accuracy_score",
        "source_grounding_score",
        "format_compliance_score",
    ]
    for f in criterion_fields:
        setattr(s, f, max(1.0, min(5.0, getattr(s, f))))
    s.overall_score = round(
        sum(getattr(s, f) for f in criterion_fields) / len(criterion_fields), 1
    )
    if not s.overall_reasoning:
        s.overall_reasoning = ev.reasoning
    return ev


def _fallback_evaluation() -> AntibodyExtractionEvaluation:
    """Return a minimal failing evaluation when the judge cannot complete."""
    return AntibodyExtractionEvaluation(
        reasoning="Judge could not complete evaluation.",
        scores=AntibodyExtractionScores(
            completeness_score=1.0,
            completeness_reasoning="Evaluation failed.",
            accuracy_score=1.0,
            accuracy_reasoning="Evaluation failed.",
            source_grounding_score=1.0,
            source_grounding_reasoning="Evaluation failed.",
            format_compliance_score=1.0,
            format_compliance_reasoning="Evaluation failed.",
        ),
        confidence_level="low",
    )


# ===========================================================================
# SECTION 9 — PDF EXTRACTION FUNCTION
# ===========================================================================


def extract_markdown_from_pdf(pdf_path: Path, use_gpu: bool = True) -> str:
    """Convert a PDF file to Markdown text via Docling."""
    extractor = DoclingPDFExtractor(use_gpu=use_gpu)
    logger.info("Extracting text from %s ...", pdf_path.name)
    t0 = time.time()
    with open(pdf_path, "rb") as fh:
        markdown = extractor.forward(fh.read())
    logger.info("Extracted %d characters in %.1fs", len(markdown), time.time() - t0)
    return markdown


# ===========================================================================
# SECTION 10 — PIPELINE
# ===========================================================================


def build_lm(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 16_000,
) -> SystemPrefixedLM:
    """Instantiate an LM from the registry."""
    return get_llm_from_name(
        model_name,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )


def process_paper(
    pdf_path: Path,
    extractor: AntibodyExtractor,
    judge: AntibodyJudge | None,
    use_gpu: bool,
) -> dict:
    """Run extraction (and optionally judgement) for one PDF."""
    paper_id = pdf_path.stem
    t0 = time.time()

    # 1. PDF → Markdown
    paper_text = extract_markdown_from_pdf(pdf_path, use_gpu=use_gpu)

    # 2. Extract antibody data
    logger.info("Running antibody extraction for %s ...", paper_id)
    table = extractor.forward(paper_text)

    # 3. Optional quality judgement
    evaluation: AntibodyExtractionEvaluation | None = None
    if judge is not None:
        logger.info("Running extraction judge for %s ...", paper_id)
        evaluation = judge.forward(source_text=paper_text, extracted_table=table)

    elapsed = round(time.time() - t0, 1)
    logger.info(
        "Finished %s: %d antibody record(s) in %.1fs",
        paper_id,
        len(table.antibodies),
        elapsed,
    )

    return {
        "paper_id": paper_id,
        "paper_path": str(pdf_path),
        "num_antibodies": len(table.antibodies),
        "antibodies": [ab.model_dump() for ab in table.antibodies],
        "evaluation": evaluation.model_dump() if evaluation else None,
        "processing_time_seconds": elapsed,
    }


# ===========================================================================
# SECTION 11 — OUTPUT HELPERS
# ===========================================================================

_CSV_FIELDS = [
    "paper_id",
    "name",
    "vh",
    "vl",
    "hic_retention_time_min",
    "phenyl_sepharose_retention_time_min",
    "butyl_sepharose_retention_time_min",
    "octyl_sepharose_retention_time_min",
    "hic_resin",
    "smac_retention_time_min",
    "cic_retention_time_min",
    "judge_overall_score",
    "judge_confidence",
]


def _ab_to_row(paper_id: str, ab: dict, judge_score: Any, judge_conf: Any) -> dict:
    def _val(v: Any) -> str:
        return "" if v is None else str(v)

    return {
        "paper_id": paper_id,
        "name": _val(ab.get("name")),
        "vh": _val(ab.get("vh")),
        "vl": _val(ab.get("vl")),
        "hic_retention_time_min": _val(ab.get("hic_retention_time_min")),
        "phenyl_sepharose_retention_time_min": _val(
            ab.get("phenyl_sepharose_retention_time_min")
        ),
        "butyl_sepharose_retention_time_min": _val(
            ab.get("butyl_sepharose_retention_time_min")
        ),
        "octyl_sepharose_retention_time_min": _val(
            ab.get("octyl_sepharose_retention_time_min")
        ),
        "hic_resin": _val(ab.get("hic_resin")),
        "smac_retention_time_min": _val(ab.get("smac_retention_time_min")),
        "cic_retention_time_min": _val(ab.get("cic_retention_time_min")),
        "judge_overall_score": _val(judge_score),
        "judge_confidence": _val(judge_conf),
    }


def save_results(all_results: list[dict], output_dir: Path) -> None:
    """Save per-paper JSON files and a combined flat CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-paper JSON
    for result in all_results:
        json_path = output_dir / f"{result['paper_id']}_antibodies.json"
        with open(json_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("Saved JSON: %s", json_path)

    # Combined CSV (one row per antibody per paper)
    csv_path = output_dir / "antibodies_combined.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in all_results:
            paper_id = result["paper_id"]
            ev = result.get("evaluation") or {}
            scores = ev.get("scores", {})
            overall_score = scores.get("overall_score", "")
            confidence = ev.get("confidence_level", "")

            if not result["antibodies"]:
                # Paper had no antibody data — still emit one placeholder row
                writer.writerow(
                    {f: "" for f in _CSV_FIELDS}
                    | {"paper_id": paper_id,
                       "judge_overall_score": overall_score,
                       "judge_confidence": confidence}
                )
            else:
                for ab in result["antibodies"]:
                    writer.writerow(
                        _ab_to_row(paper_id, ab, overall_score, confidence)
                    )

    logger.info("Saved combined CSV: %s", csv_path)


# ===========================================================================
# SECTION 12 — CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract antibody chromatography data from PDFs using DSPy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_antibodies.py paper.pdf
  python extract_antibodies.py papers/ --output results/ --model gpt-4o
  python extract_antibodies.py paper.pdf --judge-model claude-sonnet-4.6
  python extract_antibodies.py paper.pdf --no-judge --no-gpu --max 3
        """,
    )
    parser.add_argument(
        "input",
        help="PDF file or directory of PDF files to process.",
    )
    parser.add_argument(
        "--output",
        default="./antibody_results",
        help="Output directory for JSON + CSV (default: ./antibody_results).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Extraction LLM key from LLM_REGISTRY (default: gemini-2.0-flash).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge LLM key (default: same as --model). "
            "Use a different model for independent verification."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable extraction quality judge (faster).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Run Docling PDF extraction on CPU only.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Process at most N PDFs (useful for quick testing).",
    )
    return parser.parse_args()


def collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            logger.error("No PDF files found in %s", input_path)
        return pdfs
    logger.error("Input path does not exist: %s", input_path)
    return []


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    use_gpu = not args.no_gpu

    pdf_files = collect_pdfs(input_path)
    if not pdf_files:
        sys.exit(1)
    if args.max and len(pdf_files) > args.max:
        logger.info("Limiting to first %d PDFs (--max)", args.max)
        pdf_files = pdf_files[: args.max]

    logger.info("Processing %d PDF(s) with model=%r", len(pdf_files), args.model)

    extraction_lm = build_lm(args.model)
    judge_model_name = args.judge_model or args.model
    judge_lm = build_lm(judge_model_name, temperature=0.1)

    extractor = AntibodyExtractor(lm=extraction_lm)
    judge: AntibodyJudge | None = None if args.no_judge else AntibodyJudge(lm=judge_lm)

    if judge is None:
        logger.info("Quality judge disabled (--no-judge).")
    else:
        logger.info("Quality judge enabled (model=%r).", judge_model_name)

    all_results: list[dict] = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info("=== Paper %d/%d: %s ===", i, len(pdf_files), pdf_path.name)
        try:
            result = process_paper(pdf_path, extractor, judge, use_gpu)
            all_results.append(result)
        except Exception as exc:
            logger.error("FAILED %s: %r", pdf_path.name, exc)
            traceback.print_exc()
            all_results.append(
                {
                    "paper_id": pdf_path.stem,
                    "paper_path": str(pdf_path),
                    "error": str(exc),
                    "antibodies": [],
                    "evaluation": None,
                }
            )

    save_results(all_results, output_dir)

    # ---- Summary -----------------------------------------------------------
    total_ab = sum(r.get("num_antibodies", 0) for r in all_results)
    failed = sum(1 for r in all_results if "error" in r)
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Papers processed : {len(pdf_files)}")
    print(f"  Failed           : {failed}")
    print(f"  Total antibodies : {total_ab}")
    print(f"  Output directory : {output_dir}")
    if not args.no_judge:
        scores = [
            r["evaluation"]["scores"]["overall_score"]
            for r in all_results
            if r.get("evaluation") and r["evaluation"].get("scores")
        ]
        if scores:
            print(f"  Avg judge score  : {sum(scores)/len(scores):.1f}/5.0")
    print()
    for r in all_results:
        if "error" in r:
            print(f"  [FAIL] {r['paper_id']}: {r['error']}")
        else:
            score_str = ""
            if r.get("evaluation") and r["evaluation"].get("scores"):
                score_str = f"  judge={r['evaluation']['scores']['overall_score']:.1f}/5"
            print(
                f"  [OK]   {r['paper_id']}: {r['num_antibodies']} antibody record(s)"
                f"{score_str}"
            )


if __name__ == "__main__":
    main()
