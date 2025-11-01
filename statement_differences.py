from statement_similarity_analysis import analyze_statements, extract_key_points, compare_key_points

def generate_difference_report(suspect_statement, witness_statement):
    suspect_points = extract_key_points(suspect_statement)
    witness_points = extract_key_points(witness_statement)
    comparison = compare_key_points(suspect_points, witness_points)

    report_lines = []
    report_lines.append("🧾 STATEMENTS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Suspect: {suspect_statement}")
    report_lines.append(f"Witness: {witness_statement}")
    report_lines.append("-" * 60)

    for category, data in comparison.items():
        match = data["matching"]
        diff = data["different"]
        if match or diff:
            report_lines.append(f"\n CATEGORY: {category.upper()}")
            if match:
                report_lines.append(f"   Matching elements: {', '.join(match)}")
            if diff:
                report_lines.append(f"  Different elements: {', '.join(diff)}")

    return "\n".join(report_lines)

if __name__ == "__main__":
    suspect = "I was at the store at 8 AM and talked to the cashier. I didn’t take anything."
    witness = "The suspect entered around 8 AM and took something without paying."
    print(generate_difference_report(suspect, witness))
