def check_workers(result):
    names = result.names
    dets = [(names[int(b.cls[0])], float(b.conf[0]), b.xyxy[0].tolist()) for b in result.boxes]
    persons = [d for d in dets if d[0] == "person"]
    others = [d for d in dets if d[0] != "person"]

    statuses = []
    for _, _, (x1, y1, x2, y2) in persons:
        found = set()
        for name, _, (a1, b1, a2, b2) in others:
            cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                found.add(name)

        issues = []
        if "no_helmet" in found:
            issues.append("Helmet Missing")
        if "no_vest" in found:
            issues.append("Safety Vest Missing")

        if issues:
            statuses.append("Violation: " + ", ".join(issues))
        elif {"helmet", "vest"} <= found:
            statuses.append("Compliant")
        else:
            statuses.append("Unclear")
    return statuses


def dashboard(statuses):
    total = len(statuses)
    ok = sum(s == "Compliant" for s in statuses)
    return {
        "Total Workers": total,
        "Compliant": ok,
        "Non-Compliant": total - ok,
        "Compliance %": round(100 * ok / max(total, 1), 1),
        "Helmet Violations": sum("Helmet" in s for s in statuses),
        "Vest Violations": sum("Vest" in s for s in statuses),
    }