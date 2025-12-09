"""
PDF Report Generator
====================
Generate comprehensive PDF reports for dataset creation.

Features:
- Dataset summary and statistics
- Processing metrics
- Visual charts and graphs
- Professional formatting
- Export to PDF

Dependencies:
    - reportlab: PDF generation library
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
from pathlib import Path
import json


class DatasetReportGenerator:
    """Generate professional PDF reports for dataset creation."""
    
    def __init__(self, output_path=None):
        """
        Initialize the report generator.
        
        Args:
            output_path: Path where PDF will be saved. If None, uses temp location.
        """
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        # Info text style
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#555555'),
            spaceAfter=6
        ))
    
    def generate_report(self, report_data):
        """
        Generate a comprehensive PDF report.
        
        Args:
            report_data: Dictionary containing all report information
                - dataset_name: Name of the dataset
                - class_names: List of class names
                - statistics: Dict with processing statistics
                - split_info: Dict with train/test/valid counts
                - processing_time: Time taken for processing
                - zip_info: Information about the ZIP file
                - uploaded_count: Number of uploaded images
                - duplicates_removed: Number of duplicates
                - augmented_count: Number of augmented images
                - export_format: Label export format (if any)
                - has_labels: Boolean indicating if labels were included
        
        Returns:
            str: Path to generated PDF file
        """
        if not self.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"dataset_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=36
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add title
        title = Paragraph("Mini Dataset Generator Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Add generation date
        date_text = Paragraph(
            f"<i>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
            self.styles['InfoText']
        )
        elements.append(date_text)
        elements.append(Spacer(1, 20))
        
        # Dataset Overview Section
        elements.append(Paragraph("Dataset Overview", self.styles['SectionHeading']))
        overview_data = [
            ['Dataset Name', report_data.get('dataset_name', 'N/A')],
            ['Classes', ', '.join(report_data.get('class_names', []))],
            ['Number of Classes', str(len(report_data.get('class_names', [])))],
            ['Export Format', report_data.get('export_format', 'YOLO')],
            ['Labels Included', 'Yes' if report_data.get('has_labels', False) else 'No'],
            ['Processing Time', f"{report_data.get('processing_time', 0):.2f} seconds"],
        ]
        overview_table = self._create_table(overview_data)
        elements.append(overview_table)
        elements.append(Spacer(1, 20))
        
        # Processing Statistics Section
        elements.append(Paragraph("Processing Statistics", self.styles['SectionHeading']))
        stats = report_data.get('statistics', {})
        processing_data = [
            ['Metric', 'Count'],
            ['Images Uploaded', str(report_data.get('uploaded_count', 0))],
            ['Duplicates Removed', str(report_data.get('duplicates_removed', 0))],
            ['Unique Images', str(stats.get('unique_images', 0))],
            ['Augmented Images', str(report_data.get('augmented_count', 0))],
            ['Total Dataset Images', str(stats.get('total_images', 0))],
        ]
        processing_table = self._create_table(processing_data, has_header=True)
        elements.append(processing_table)
        elements.append(Spacer(1, 20))
        
        # Dataset Split Section
        elements.append(Paragraph("Dataset Split Information", self.styles['SectionHeading']))
        split_info = report_data.get('split_info', {})
        split_data = [
            ['Split', 'Images', 'Percentage'],
            ['Training Set', str(split_info.get('train_images', 0)), f"{split_info.get('train_ratio', 0)*100:.0f}%"],
            ['Test Set', str(split_info.get('test_images', 0)), f"{split_info.get('test_ratio', 0)*100:.0f}%"],
            ['Validation Set', str(split_info.get('valid_images', 0)), f"{split_info.get('valid_ratio', 0)*100:.0f}%"],
            ['Total', str(split_info.get('total_images', 0)), '100%'],
        ]
        split_table = self._create_table(split_data, has_header=True, highlight_last=True)
        elements.append(split_table)
        elements.append(Spacer(1, 20))
        
        # File Information Section
        elements.append(Paragraph("Output File Information", self.styles['SectionHeading']))
        zip_info = report_data.get('zip_info', {})
        file_data = [
            ['Files in ZIP', str(zip_info.get('file_count', 0))],
            ['Compressed Size', f"{zip_info.get('compressed_size_mb', 0):.2f} MB"],
            ['Uncompressed Size', f"{zip_info.get('uncompressed_size_mb', 0):.2f} MB"],
            ['Compression Ratio', f"{zip_info.get('compression_ratio', 0):.1f}%"],
        ]
        file_table = self._create_table(file_data)
        elements.append(file_table)
        elements.append(Spacer(1, 20))
        
        # Label Information (if applicable)
        if report_data.get('has_labels', False):
            elements.append(Paragraph("Label Information", self.styles['SectionHeading']))
            label_data = [
                ['Label Format', report_data.get('export_format', 'YOLO')],
                ['Labels Matched', str(report_data.get('labels_matched', 0))],
                ['Label Source', 'User Provided' if report_data.get('has_labels') else 'Not Provided'],
            ]
            label_table = self._create_table(label_data)
            elements.append(label_table)
            elements.append(Spacer(1, 20))
        
        # Next Steps Section
        elements.append(Paragraph("Next Steps", self.styles['SectionHeading']))
        next_steps = [
            "1. Extract the downloaded ZIP file to your working directory",
            "2. Review the dataset structure (train/test/valid folders)",
            "3. If labels are included, verify annotations are correct",
            "4. Update data.yaml with correct paths if needed",
            "5. Train your YOLO model using the dataset",
            "6. Evaluate model performance on test set",
            "7. Fine-tune hyperparameters as needed"
        ]
        for step in next_steps:
            elements.append(Paragraph(step, self.styles['InfoText']))
        elements.append(Spacer(1, 20))
        
        # Footer section
        elements.append(Spacer(1, 30))
        footer_text = Paragraph(
            "<i>This report was automatically generated by Mini Dataset Generator.<br/>"
            "For questions or issues, please refer to the documentation.</i>",
            self.styles['InfoText']
        )
        elements.append(footer_text)
        
        # Build PDF
        doc.build(elements)
        
        return self.output_path
    
    def _create_table(self, data, has_header=False, highlight_last=False):
        """
        Create a formatted table.
        
        Args:
            data: 2D list of table data
            has_header: Boolean indicating if first row is header
            highlight_last: Boolean indicating if last row should be highlighted
        
        Returns:
            Table object
        """
        table = Table(data, colWidths=[3*inch, 3*inch])
        
        # Base style
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4') if has_header else colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke if has_header else colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold' if has_header else 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]
        
        # Highlight last row if requested
        if highlight_last:
            style.extend([
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#d4edda')),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ])
        
        table.setStyle(TableStyle(style))
        return table
    
    def save_report_data(self, report_data, json_path):
        """
        Save report data as JSON for future reference.
        
        Args:
            report_data: Dictionary containing report information
            json_path: Path where JSON will be saved
        
        Returns:
            str: Path to saved JSON file
        """
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return json_path


# Example usage
if __name__ == "__main__":
    # Sample report data
    sample_data = {
        'dataset_name': 'my_dataset',
        'class_names': ['cat', 'dog', 'bird'],
        'statistics': {
            'unique_images': 50,
            'total_images': 250
        },
        'split_info': {
            'train_images': 175,
            'test_images': 37,
            'valid_images': 38,
            'total_images': 250,
            'train_ratio': 0.7,
            'test_ratio': 0.15,
            'valid_ratio': 0.15
        },
        'processing_time': 45.3,
        'zip_info': {
            'file_count': 500,
            'compressed_size_mb': 125.6,
            'uncompressed_size_mb': 256.3,
            'compression_ratio': 51.0
        },
        'uploaded_count': 60,
        'duplicates_removed': 10,
        'augmented_count': 200,
        'export_format': 'YOLO',
        'has_labels': False
    }
    
    generator = DatasetReportGenerator("sample_report.pdf")
    pdf_path = generator.generate_report(sample_data)
    print(f"Report generated: {pdf_path}")
