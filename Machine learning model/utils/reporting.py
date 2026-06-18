"""Rich clinical report generation with an optional PDF backend."""

from html import escape
from io import BytesIO
import textwrap

from .recommendations import DISCLAIMER, generate_clinical_summary, generate_recommendations


def _top_features(results):
    frame = results.get("feature_contributions")
    if frame is None:
        return []
    return [
        {"feature": row["Feature"], "contribution": float(row["Contribution"])}
        for _, row in frame.head(5).iterrows()
    ]


def report_sections(results, metadata):
    if not results:
        return {"title": "GlucoSight AI Clinical Report", "empty": True, "disclaimer": DISCLAIMER}
    probability = float(results["probability"])
    patient = results["user_input"]
    top = _top_features(results)
    return {
        "title": "GlucoSight AI Clinical Report",
        "empty": False,
        "assessment_id": results.get("assessment_id", "N/A"),
        "risk": results["risk_level"],
        "probability": probability,
        "confidence": max(probability, 1 - probability),
        "model": metadata.get("model_type", "Unknown"),
        "patient": patient,
        "top_features": top,
        "summary": results.get("clinical_summary") or generate_clinical_summary(probability, patient, top),
        "recommendations": results.get("recommendations") or generate_recommendations(probability, patient),
        "disclaimer": DISCLAIMER,
    }


def build_text_report(results, metadata):
    data = report_sections(results, metadata)
    if data["empty"]:
        return f"{data['title']}\n\nNo assessment has been generated yet.\n\n{data['disclaimer']}\n"
    lines = [
        data["title"], "=" * len(data["title"]), "",
        f"Assessment: {data['assessment_id']}", f"Model: {data['model']}",
        f"Risk category: {data['risk']}", f"Risk probability: {data['probability'] * 100:.1f}%",
        f"Classification confidence: {data['confidence'] * 100:.1f}%", "", "Patient inputs", "--------------",
    ]
    lines.extend(f"{name}: {value}" for name, value in data["patient"].items())
    lines.extend(["", "Top contributing features", "-------------------------"])
    for item in data["top_features"]:
        direction = "increases" if item["contribution"] >= 0 else "reduces"
        lines.append(f"{item['feature']}: {direction} estimated risk ({item['contribution']:+.4f})")
    lines.extend(["", "AI clinical summary", "-------------------", data["summary"], "", "Recommendations", "---------------"])
    for item in data["recommendations"]:
        lines.append(f"{item['category']} - {item['title']}: {item['detail']}")
    lines.extend(["", "Disclaimer", "----------", data["disclaimer"]])
    return "\n".join(lines) + "\n"


def build_html_report(results, metadata):
    data = report_sections(results, metadata)
    if data["empty"]:
        body = "<p>No assessment has been generated yet.</p>"
    else:
        inputs = "".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in data["patient"].items())
        features = "".join(
            f"<li><strong>{escape(item['feature'])}</strong>: {'increases' if item['contribution'] >= 0 else 'reduces'} estimated risk ({item['contribution']:+.4f})</li>"
            for item in data["top_features"]
        )
        recommendations = "".join(
            f"<div class='recommendation'><strong>{escape(item['category'])}: {escape(item['title'])}</strong><p>{escape(item['detail'])}</p></div>"
            for item in data["recommendations"]
        )
        body = f"""
        <div class="metrics"><div><small>RISK</small><strong>{escape(data['risk'])}</strong></div><div><small>PROBABILITY</small><strong>{data['probability'] * 100:.1f}%</strong></div><div><small>CONFIDENCE</small><strong>{data['confidence'] * 100:.1f}%</strong></div></div>
        <h2>AI Clinical Summary</h2><p>{escape(data['summary'])}</p>
        <h2>Patient Inputs</h2><table>{inputs}</table>
        <h2>Top Contributing Features</h2><ol>{features}</ol>
        <h2>Recommendations</h2>{recommendations}
        """
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(data['title'])}</title><style>
    body{{font-family:Inter,Arial,sans-serif;margin:0;background:#f4f7fb;color:#172033}}main{{max-width:900px;margin:32px auto;background:white;padding:40px;border-radius:18px}}header{{border-bottom:3px solid #3267c5;padding-bottom:18px}}.brand{{color:#3267c5;font-weight:800}}h1{{margin:6px 0}}h2{{margin-top:28px;color:#213756}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}}.metrics div,.recommendation{{background:#f2f6fc;border:1px solid #dbe6f5;border-radius:10px;padding:14px}}small,strong{{display:block}}table{{width:100%;border-collapse:collapse}}td{{padding:9px;border-bottom:1px solid #e4e9f1}}footer{{margin-top:30px;padding:16px;background:#fff5e5;border-radius:10px;font-size:12px}}
    </style></head><body><main><header><div class="brand">GLUCOSIGHT AI</div><h1>{escape(data['title'])}</h1><p>Explainable diabetes risk intelligence</p></header>{body}<footer><strong>Disclaimer</strong>{escape(data['disclaimer'])}</footer></main></body></html>"""


def build_pdf_report(results, metadata):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except Exception:
        return None
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter
    left, y = 0.7 * inch, height - 0.7 * inch
    pdf.setTitle("GlucoSight AI Clinical Report")
    for line_number, line in enumerate(build_text_report(results, metadata).splitlines()):
        if y < 0.65 * inch:
            pdf.showPage(); y = height - 0.7 * inch
        pdf.setFont("Helvetica-Bold" if line_number == 0 or line in {"Patient inputs", "Top contributing features", "AI clinical summary", "Recommendations", "Disclaimer"} else "Helvetica", 16 if line_number == 0 else 9)
        if line and set(line) in ({"="}, {"-"}):
            continue
        for wrapped in textwrap.wrap(line, 100) or [""]:
            pdf.drawString(left, y, wrapped); y -= 0.17 * inch
    pdf.save(); buffer.seek(0)
    return buffer.getvalue()

