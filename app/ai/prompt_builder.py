"""Prompt templates for constrained AI match analysis."""

from __future__ import annotations

import json
from typing import Optional

from app.ai.schemas import JsonDict


class PromptBuilder:
    """Builds provider-neutral prompts from computed analytics."""

    def build_report_prompt(self, context: JsonDict) -> str:
        return self._base_prompt(
            instruction=(
                "Generate a professional football match analyst report with: "
                "match summary, team analysis, every-player reports, coach report, "
                "opposition analysis, and recommendations."
            ),
            context=context,
        )

    def build_query_prompt(self, context: JsonDict, question: str) -> str:
        return self._base_prompt(
            instruction=f"Answer this analyst question using only the context: {question}",
            context=context,
        )

    def _base_prompt(self, instruction: str, context: JsonDict) -> str:
        compact_context = json.dumps(context, indent=2, sort_keys=True)
        xg = context.get("xg", {})
        xg_summary = xg.get("summary", {}) if xg.get("summary") else {}
        xg_total = xg_summary.get("total_xg", 0)
        xg_shots_count = xg_summary.get("total_shots", 0)
        xg_section = ""
        if xg_summary and xg_shots_count > 0:
            xg_avg = xg_summary.get("average_xg", 0)
            best = xg_summary.get("highest_xg_shot") or {}
            worst = xg_summary.get("lowest_xg_shot") or {}
            top_finishers = xg_summary.get("top_finishers", [])
            best_line = (
                f"   - Highest xG chance: Player #{best.get('player', '?')} ({best.get('team', '?')}) — "
                f"xG {best.get('xg', 0)} {'(GOAL)' if best.get('goal') else '(MISSED)'}\n"
            ) if best else ""
            worst_line = (
                f"   - Lowest xG chance: Player #{worst.get('player', '?')} ({worst.get('team', '?')}) — "
                f"xG {worst.get('xg', 0)}\n"
            ) if worst else ""
            xg_section = (
                f"6. Expected Goals (xG) — do not invent xG values; use the computed values below:\n"
                f"   - Total xG: {xg_total} from {xg_shots_count} shots (avg {xg_avg} per shot)\n"
                f"{best_line}{worst_line}"
            )
            if top_finishers:
                finisher_lines = []
                for pid, fin_data in top_finishers:
                    finisher_lines.append(
                        f"     * Player #{pid}: Goals−xG = {fin_data.get('goals_minus_xg', 0):.3f}"
                    )
                xg_section += "   - Top finishers (by Goals − xG):\n" + "\n".join(finisher_lines) + "\n"

        xa = context.get("xa", {})
        xa_summary = xa.get("summary", {}) if xa.get("summary") else {}
        xa_total = xa_summary.get("total_xa", 0)
        xa_passes_count = xa_summary.get("total_passes", 0)
        xa_section = ""
        if xa_summary and xa_passes_count > 0:
            xa_avg = xa_summary.get("average_xa", 0)
            best_creator = xa_summary.get("best_chance_creator") or {}
            top_creators = xa_summary.get("top_creators", [])
            creator_line = (
                f"   - Best chance creator: Player #{best_creator.get('player', '?')} "
                f"({best_creator.get('team', '?')}) — xA {best_creator.get('xA', 0):.3f}"
                f"\n"
            ) if best_creator else ""
            xa_section = (
                f"7. Expected Assists (xA) — do not invent xA values; use the computed values below:\n"
                f"   - Total xA: {xa_total} from {xa_passes_count} linked passes (avg {xa_avg} per pass)\n"
                f"{creator_line}"
            )
            if top_creators:
                creator_lines = []
                for pid, cr_data in top_creators:
                    creator_lines.append(
                        f"     * Player #{pid}: Total xA = {cr_data.get('total_xa', 0):.3f}"
                    )
                xa_section += "   - Top chance creators (by Total xA):\n" + "\n".join(creator_lines) + "\n"

        xt = context.get("xt", {})
        xt_summary = xt.get("summary", {}) if xt.get("summary") else {}
        xt_total = xt_summary.get("total_xt", 0)
        xt_actions_count = xt_summary.get("total_actions", 0)
        xt_section = ""
        if xt_summary and xt_actions_count > 0:
            xt_avg = xt_summary.get("average_xt", 0)
            best_action = xt_summary.get("highest_xt_action") or {}
            top_players = xt_summary.get("top_players", [])
            best_line = (
                f"   - Highest xT action: {best_action.get('action', '?')} by Player #{best_action.get('player_id', '?')} "
                f"({best_action.get('team', '?')}) — xT {best_action.get('xt', 0):.3f}\n"
            ) if best_action else ""
            xt_section = (
                f"8. Expected Threat (xT) — do not invent xT values; use the computed values below:\n"
                f"   - Total xT: {xt_total} from {xt_actions_count} actions (avg {xt_avg} per action)\n"
                f"{best_line}"
            )
            if top_players:
                top_lines = []
                for pid, tp_data in top_players:
                    top_lines.append(
                        f"     * Player #{pid}: Total xT = {tp_data.get('total_xt', 0):.3f}"
                    )
                xt_section += "   - Top threat creators (by Total xT):\n" + "\n".join(top_lines) + "\n"

        formation = context.get("formation", {})
        formation_section = ""
        if formation:
            formation_section = (
                "9. Formation Intelligence — use the structured tactical context below:\n"
            )
            windows = formation.get("windows", [])
            transitions = formation.get("transitions", [])
            metrics = formation.get("metrics", {})
            if windows:
                formation_section += f"   - Formation windows analyzed: {len(windows)}\n"
            if transitions:
                formation_section += f"   - Formation transitions detected: {len(transitions)}\n"
            if metrics:
                formation_section += (
                    f"   - Latest metrics: width={metrics.get('team_width', 'N/A')}, "
                    f"length={metrics.get('team_length', 'N/A')}, compactness={metrics.get('compactness', 'N/A')}, "
                    f"defensive_line={metrics.get('defensive_line', 'N/A')}, "
                    f"midfield_line={metrics.get('midfield_line', 'N/A')}, "
                    f"forward_line={metrics.get('forward_line', 'N/A')}\n"
                )

        pressing = context.get("pressing", {})
        pressing_section = ""
        if pressing:
            pressing_section = (
                "10. Pressing Intelligence — use the structured pressing context below:\n"
            )
            p_metrics = pressing.get("metrics", {})
            p_detection = pressing.get("detection", {})
            p_events = pressing.get("events", [])
            p_sequences = pressing.get("sequences", [])
            p_timeline = pressing.get("timeline", [])
            if p_metrics:
                pressing_section += (
                    f"   - Total pressures: {p_metrics.get('total_pressures', 'N/A')}\n"
                    f"   - Successful pressures: {p_metrics.get('successful_pressures', 'N/A')}\n"
                    f"   - Pressure success rate: {p_metrics.get('pressure_success_rate', 'N/A')}\n"
                    f"   - Average pressure time: {p_metrics.get('average_pressure_time', 'N/A')}s\n"
                    f"   - Average closing speed: {p_metrics.get('average_closing_speed', 'N/A')} m/s\n"
                    f"   - PPDA: {p_metrics.get('ppda', 'N/A')}\n"
                    f"   - High press count: {p_metrics.get('high_press_count', 'N/A')}\n"
                    f"   - Mid block count: {p_metrics.get('mid_block_count', 'N/A')}\n"
                    f"   - Low block count: {p_metrics.get('low_block_count', 'N/A')}\n"
                )
            if p_detection:
                pressing_section += (
                    f"   - Detected pressing style: {p_detection.get('pressing_style', 'N/A')}\n"
                    f"   - Detection confidence: {p_detection.get('confidence', 'N/A')}\n"
                )
            if p_events:
                pressing_section += f"   - Pressure events recorded: {len(p_events)}\n"
            if p_sequences:
                pressing_section += f"   - Pressing sequences detected: {len(p_sequences)}\n"
            if p_timeline:
                pressing_section += f"   - Pressing timeline frames: {len(p_timeline)}\n"

        sections = [
            xg_section,
            xa_section,
            xt_section,
            formation_section,
            pressing_section,
        ]
        clean_sections = "\n".join(section for section in sections if section)
        return (
            "You are a professional football match analyst.\n"
            "Generate a structured report using ONLY the provided analytics context.\n"
            "Do not invent statistics or player names outside the context.\n\n"
            f"{instruction}\n\n"
            "Context:\n"
            f"{compact_context}\n\n"
            "Structured Analytics Sections:\n"
            f"{clean_sections}\n"
        )

    def build_markdown_from_sections(
        self,
        title: str,
        summary: str,
        sections: Optional[JsonDict] = None,
    ) -> str:
        body = [f"# {title}", "", summary.strip()]
        for name, value in (sections or {}).items():
            body.extend(["", f"## {name}", "", json.dumps(value, indent=2, sort_keys=True)])
        return "\n".join(body).strip() + "\n"