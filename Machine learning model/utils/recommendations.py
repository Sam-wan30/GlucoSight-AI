"""Patient-specific recommendations and local clinical narrative generation."""

DISCLAIMER = (
    "Educational decision-support output only. This assessment is not a diagnosis "
    "and does not replace evaluation by a qualified healthcare professional."
)


def _risk_label(probability):
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "moderate"
    return "low"


def generate_recommendations(probability, patient):
    """Return four professional recommendation groups for one assessment."""
    glucose = float(patient["Glucose"])
    bmi = float(patient["BMI"])
    bp = float(patient["BloodPressure"])
    age = int(patient["Age"])
    risk = _risk_label(probability)

    if bmi >= 30:
        lifestyle = (
            f"BMI is {bmi:.1f}. Discuss a sustainable 5-10% weight-reduction target, "
            "nutrition quality, sleep, and at least 150 minutes of weekly activity with a clinician."
        )
    elif bmi >= 25:
        lifestyle = (
            f"BMI is {bmi:.1f}. Prioritize gradual weight reduction, regular aerobic and "
            "resistance activity, and a high-fiber eating pattern."
        )
    else:
        lifestyle = (
            f"BMI is {bmi:.1f}. Maintain a balanced eating pattern, regular activity, "
            "adequate sleep, and a stable healthy weight."
        )

    if risk == "high" or glucose >= 126:
        follow_up = (
            "Arrange prompt clinical review and confirmatory testing such as HbA1c or fasting "
            "plasma glucose. A model prediction alone cannot establish diabetes."
        )
        urgency = "Prompt review recommended"
        urgency_detail = "Seek timely clinician assessment, especially if symptoms of hyperglycemia are present."
    elif risk == "moderate" or glucose >= 100:
        follow_up = (
            "Schedule primary-care follow-up for guideline-appropriate diabetes screening and "
            "a personalized prevention plan."
        )
        urgency = "Preventive follow-up"
        urgency_detail = "This is not an emergency result, but delaying preventive review is not advised."
    else:
        follow_up = (
            "Continue routine preventive care and diabetes screening based on age, family history, "
            "pregnancy history, and clinician guidance."
        )
        urgency = "Routine follow-up"
        urgency_detail = "No model-based urgency is indicated; reassess if symptoms or risk factors change."

    monitoring_parts = [f"Track glucose trends (current input {glucose:.0f} mg/dL)"]
    if bp >= 90:
        monitoring_parts.append(f"review elevated blood pressure input ({bp:.0f} mmHg) promptly")
    elif bp >= 80:
        monitoring_parts.append(f"repeat and review blood pressure ({bp:.0f} mmHg)")
    else:
        monitoring_parts.append(f"continue routine blood pressure checks ({bp:.0f} mmHg)")
    if age >= 45:
        monitoring_parts.append(f"maintain regular screening because age is {age}")

    return [
        {"category": "Lifestyle", "title": "Personalized risk reduction", "detail": lifestyle, "tone": "green"},
        {"category": "Clinical follow-up", "title": "Confirm and contextualize", "detail": follow_up, "tone": "red" if risk == "high" else "yellow"},
        {"category": "Monitoring", "title": "Track the clinical trajectory", "detail": "; ".join(monitoring_parts).capitalize() + ".", "tone": "yellow"},
        {"category": "Urgency", "title": urgency, "detail": urgency_detail, "tone": "red" if risk == "high" else "green"},
    ]


def generate_clinical_summary(probability, patient, top_features=None):
    """Generate a deterministic, professional narrative without an external API."""
    risk = _risk_label(probability)
    confidence = max(probability, 1 - probability)
    glucose = float(patient["Glucose"])
    bmi = float(patient["BMI"])
    bp = float(patient["BloodPressure"])
    age = int(patient["Age"])

    drivers = []
    for item in (top_features or [])[:3]:
        name = item.get("feature", item.get("Feature", "factor"))
        direction = item.get("direction")
        if not direction:
            value = item.get("contribution", item.get("Contribution", 0))
            direction = "increased" if float(value) >= 0 else "reduced"
        drivers.append(f"{name} ({direction} estimated risk)")
    driver_text = ", ".join(drivers) if drivers else "the submitted clinical measurements"

    measurement_notes = []
    if glucose >= 126:
        measurement_notes.append(f"glucose is elevated at {glucose:.0f} mg/dL")
    elif glucose >= 100:
        measurement_notes.append(f"glucose is above the usual fasting reference range at {glucose:.0f} mg/dL")
    if bmi >= 30:
        measurement_notes.append(f"BMI is in the obesity range at {bmi:.1f}")
    elif bmi >= 25:
        measurement_notes.append(f"BMI is in the overweight range at {bmi:.1f}")
    if bp >= 80:
        measurement_notes.append(f"the blood pressure input is {bp:.0f} mmHg and merits clinical context")
    measurements = "; ".join(measurement_notes) or "no major threshold-based alerts were identified in glucose, BMI, or blood pressure"

    return (
        f"The model estimates a {probability * 100:.1f}% probability of diabetes, placing this "
        f"assessment in the {risk}-risk category with {confidence * 100:.1f}% classification confidence. "
        f"The leading model signals are {driver_text}. For this {age}-year-old patient, {measurements}. "
        "Interpret the result alongside symptoms, medical history, repeat measurements, and appropriate "
        f"diagnostic testing. {DISCLAIMER}"
    )

