"""Download page: CSV, Excel and the PDF 'Fraud Investigation Report'."""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config

P = config.PASTEL
RISK = config.RISK_COLOURS


def to_csv_bytes(results):
    return results.to_csv(index=False).encode("utf-8")


def to_excel_bytes(results, summary, high_table, model_info):
    """Excel file with 4 sheets, header row in pastel lavender."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(list(summary.items()), columns=["Item", "Value"]).to_excel(writer, sheet_name="Summary", index=False)
        results.to_excel(writer, sheet_name="All applicants", index=False)
        high_table.to_excel(writer, sheet_name="High risk", index=False)
        pd.DataFrame(list(model_info.items()), columns=["Item", "Value"]).to_excel(writer, sheet_name="Model info", index=False)
        from openpyxl.styles import Alignment, Font, PatternFill
        for ws in writer.book.worksheets:
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor=P["lavender"].lstrip("#"))
                cell.font = Font(bold=True, color="3D3A50")
                cell.alignment = Alignment(horizontal="center")
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 45)
            ws.freeze_panes = "A2"
        ws = writer.book["All applicants"]
        fills = {"High": RISK["High"], "Medium": RISK["Medium"], "Low": RISK["Low"]}
        header = [c.value for c in ws[1]]
        risk_col = header.index("Risk") + 1
        for row in ws.iter_rows(min_row=2, min_col=risk_col, max_col=risk_col):
            for cell in row:
                cell.fill = PatternFill("solid", fgColor=fills.get(cell.value, "FFFFFF").lstrip("#"))
    return buffer.getvalue()


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_risk_counts(results):
    counts = results["Risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    bars = ax.bar(counts.index, counts.values, color=[RISK[k] for k in counts.index], edgecolor="#8c86a8", linewidth=0.8)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9, color=P["text"])
    ax.set_title("Applicants by risk band", color=P["text"], fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_png(fig)


def chart_score_hist(results, thresholds):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.hist(results["Fraud Score"], bins=30, color=P["blue"], edgecolor="white")
    ax.axvline(thresholds["medium"], color="#E0B84F", linestyle="--", linewidth=1.2, label="Medium threshold")
    ax.axvline(thresholds["high"], color="#E27A98", linestyle="--", linewidth=1.2, label="High threshold")
    ax.set_title("Distribution of fraud scores", color=P["text"], fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_png(fig)


def chart_age(results):
    g = results.groupby("Age band")["Fraud Score"].mean()
    order = [x for x in ["Under 30", "30-39", "40-49", "50-59", "60 and over"] if x in g.index]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.bar(order, g[order].values, color=P["lavender"], edgecolor="#8c86a8", linewidth=0.8)
    ax.set_title("Average fraud score by age band", color=P["text"], fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_png(fig)


def to_pdf_bytes(results, summary, top_table, thresholds, model_info, case_ref="", investigator=""):
    """PDF report. top_table has: Applicant ID, Fraud Score, Risk, Main reasons."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Fraud Investigation Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=colors.HexColor(P["text"]), fontSize=22, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#6c4fa0"), spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], textColor=colors.HexColor(P["text"]), leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=colors.HexColor(P["muted"]))
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10.5)

    story = []
    banner = Table([[Paragraph("Fraud Investigation Report", h1)],
                    [Paragraph("Predictive modelling for fraud detection using financial application data", body)]],
                   colWidths=[17 * cm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EADCF7")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (-1, -1), 10),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [banner, Spacer(1, 10)]

    meta = [["Report date", datetime.now().strftime("%d %B %Y %H:%M")],
            ["Case reference", case_ref or "-"], ["Prepared by", investigator or "-"],
            ["Model", f"{model_info['nice_name']} (version {model_info['version']})"]]
    t = Table(meta, colWidths=[4 * cm, 13 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(P["text"])),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3ECFB")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2D7F0"))]))
    story += [t, Paragraph("1. Summary", h2)]

    rows = [[k, str(v)] for k, v in summary.items()]
    t = Table(rows, colWidths=[8 * cm, 9 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(P["text"])),
                           ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#FAF6FD")]),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2D7F0"))]))
    story += [t, Spacer(1, 8)]

    story.append(Paragraph("2. Charts", h2))
    imgs = [Image(chart_risk_counts(results), width=5.6 * cm, height=4 * cm),
            Image(chart_score_hist(results, thresholds), width=5.6 * cm, height=4 * cm),
            Image(chart_age(results), width=5.6 * cm, height=4 * cm)]
    story += [Table([imgs], colWidths=[5.8 * cm] * 3), Spacer(1, 6)]

    story.append(Paragraph(f"3. Highest risk applicants (top {len(top_table)})", h2))
    story.append(Paragraph("The reasons are the features that pushed the score up the most (SHAP). They are a starting point for the "
                           "investigator, not proof of fraud.", small))
    data_rows = [["Applicant ID", "Fraud score", "Risk", "Main reasons"]]
    for _, r in top_table.iterrows():
        data_rows.append([str(r["Applicant ID"]), f"{r['Fraud Score']:.2f}", r["Risk"], Paragraph(str(r["Main reasons"]), cell)])
    t = Table(data_rows, colWidths=[2.6 * cm, 2.2 * cm, 1.9 * cm, 10.3 * cm], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9C4EF")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(P["text"])), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2D7F0"))]
    for i, (_, r) in enumerate(top_table.iterrows(), start=1):
        style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(RISK[r["Risk"]])))
    t.setStyle(TableStyle(style))
    story += [t, Paragraph("4. Recommended actions", h2)]
    actions = [
        "<b>High risk</b>: refer to a human reviewer for manual verification. No application is rejected automatically.",
        "<b>Medium risk</b>: request additional verification, then decide.",
        "<b>Low risk</b>: continue with standard processing.",
        "Record the reviewer's outcome so the model can be checked and retrained later.",
    ]
    for a in actions:
        story.append(Paragraph("&bull; " + a, body))
    story.append(Paragraph("5. Method and limitations", h2))
    story.append(Paragraph(
        f"Scores come from {model_info['nice_name']}, trained on the Bank Account Fraud (BAF) synthetic bank-application dataset "
        f"using months 0-3, with validation on months 4-5. Risk bands: High at or above {thresholds['high']:.3f}, Medium at or above "
        f"{thresholds['medium']:.3f}. This is a research prototype built for an Honours dissertation. It was trained on synthetic "
        "bank data, not SASSA data, so its scores must not be used on their own to make decisions about real people. "
        "Any use with real applicants needs approval, POPIA compliance and human review.", body))
    story.append(Spacer(1, 18))
    sign = Table([["Reviewer signature: ______________________", "Date: ______________"]], colWidths=[10 * cm, 7 * cm])
    sign.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story.append(sign)

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(P["muted"]))
        canvas.drawString(2 * cm, 1 * cm, "Fraud Investigation Report - research prototype, decision support only")
        canvas.drawRightString(19 * cm, 1 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
