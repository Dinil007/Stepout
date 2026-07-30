"""Report rendering and export utilities for AI match analysis."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from app.ai.schemas import JsonDict

LOGGER = logging.getLogger(__name__)


class AIReportGenerator:
    """Writes markdown, JSON, PDF, validation, and performance artifacts."""

    def export_all(self, output_dir: Path, report: JsonDict) -> JsonDict:
        output_dir.mkdir(parents=True, exist_ok=True)
        exports = {
            "ai_match_summary.md": self._write_text(
                output_dir / "ai_match_summary.md",
                report.get("match_summary_markdown", ""),
            ),
            "ai_match_summary.pdf": self._write_pdf(
                output_dir / "ai_match_summary.pdf",
                report.get("match_summary_markdown", ""),
            ),
            "ai_team_report.json": self._write_json(
                output_dir / "ai_team_report.json",
                report.get("team_analysis", {}),
            ),
            "ai_player_reports.json": self._write_json(
                output_dir / "ai_player_reports.json",
                report.get("player_reports", {}),
            ),
            "coach_report.md": self._write_text(
                output_dir / "coach_report.md",
                report.get("coach_report_markdown", ""),
            ),
            "opposition_report.md": self._write_text(
                output_dir / "opposition_report.md",
                report.get("opposition_report_markdown", ""),
            ),
            "recommendations.json": self._write_json(
                output_dir / "recommendations.json",
                report.get("recommendations", {}),
            ),
            "ai_validation_report.json": self._write_json(
                output_dir / "ai_validation_report.json",
                report.get("validation", {}),
            ),
            "ai_performance_report.json": self._write_json(
                output_dir / "ai_performance_report.json",
                report.get("performance", {}),
            ),
            "ai_regression_report.json": self._write_json(
                output_dir / "ai_regression_report.json",
                report.get("regression", {}),
            ),
        }
        return exports

    def _write_text(self, path: Path, content: str) -> str:
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _write_json(self, path: Path, payload: JsonDict) -> str:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    def _write_pdf(self, path: Path, markdown: str) -> str:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            pdf = canvas.Canvas(str(path), pagesize=A4)
            _, height = A4
            y = height - 48
            for line in self._wrap_lines(markdown):
                if y < 48:
                    pdf.showPage()
                    y = height - 48
                pdf.drawString(48, y, line[:105])
                y -= 14
            pdf.save()
        except Exception as exc:  # pragma: no cover - fallback for minimal envs
            LOGGER.warning("Falling back to minimal PDF writer: %s", exc)
            self._write_minimal_pdf(path, markdown)
        return str(path)

    def _wrap_lines(self, markdown: str, width: int = 95) -> Iterable[str]:
        for raw_line in markdown.splitlines():
            line = raw_line.strip() or " "
            while len(line) > width:
                yield line[:width]
                line = line[width:]
            yield line

    def _write_minimal_pdf(self, path: Path, markdown: str) -> None:
        lines = list(self._wrap_lines(markdown, width=80))[:55]
        stream_lines = ["BT", "/F1 10 Tf", "50 790 Td"]
        for index, line in enumerate(lines):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if index:
                stream_lines.append("0 -14 Td")
            stream_lines.append(f"({safe}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        objects: List[bytes] = [
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
            b"5 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream\n"
            + stream + b"\nendstream endobj\n",
        ]
        content = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for obj in objects:
            offsets.append(len(content))
            content.extend(obj)
        xref = len(content)
        content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for offset in offsets[1:]:
            content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        content.extend(
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(bytes(content))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

