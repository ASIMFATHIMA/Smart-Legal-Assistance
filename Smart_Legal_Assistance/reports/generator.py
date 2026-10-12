import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_case_summary_pdf(filename: str, summary: str, topics: list, explanation: str, output_path: str):
    """Generates a PDF report for a case summary."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    heading_style = styles['Heading2']
    heading_style.textColor = colors.darkblue
    
    body_style = styles['Normal']
    
    story = []
    
    # Title
    story.append(Paragraph(f"Smart Legal Assistance Report", title_style))
    story.append(Spacer(1, 12))
    
    # Document Info
    story.append(Paragraph(f"<b>Document:</b> {filename}", body_style))
    story.append(Spacer(1, 12))
    
    # Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(summary, body_style))
    story.append(Spacer(1, 12))
    
    # Topics
    story.append(Paragraph("Key Legal Topics", heading_style))
    topic_items = [ListItem(Paragraph(topic, body_style)) for topic in topics]
    story.append(ListFlowable(topic_items, bulletType='bullet'))
    story.append(Spacer(1, 12))
    
    # Explanation
    story.append(Paragraph("Simplified Explanation", heading_style))
    story.append(Paragraph(explanation, body_style))
    
    doc.build(story)
    return output_path
