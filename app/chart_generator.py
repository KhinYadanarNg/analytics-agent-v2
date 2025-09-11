import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ChartGenerator:
    @staticmethod
    def generate_bar_chart_base64(chart_data: List[Dict[str, Any]], title: str = "Success/Fail Rate", show_only: str = "both") -> str:
        """
        Generate a bar chart from chart data and return as base64 string.
        
        Args:
            chart_data: List of dicts with 'status', 'percentage', 'count' keys
            title: Chart title
            show_only: Filter to show only specific status ('success', 'fail', or 'both')
            
        Returns:
            Base64 encoded PNG image string
        """
        if not chart_data:
            return ChartGenerator._generate_empty_chart_base64("No data available")
        
        # Filter chart data based on show_only parameter
        if show_only != "both":
            chart_data = [item for item in chart_data if item.get("status", "").lower() == show_only.lower()]
            if not chart_data:
                return ChartGenerator._generate_empty_chart_base64(f"No {show_only} data available")
        
        # Extract data for plotting
        labels = [item.get("status", "Unknown") for item in chart_data]
        percentages = [item.get("percentage", 0) for item in chart_data]
        counts = [item.get("count", 0) for item in chart_data]
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        colors = ['#2E8B57' if label.lower() == 'success' else '#DC143C' for label in labels]
        
        bars = plt.bar(labels, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, count, percentage) in enumerate(zip(bars, counts, percentages)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{percentage}%\n({count} records)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        plt.xlabel('Status', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylim(0, max(percentages) * 1.2 if percentages else 100)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Improve layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_string = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()  # Important: close the figure to free memory
        
        return base64_string
    
    @staticmethod
    def _generate_empty_chart_base64(message: str = "No data available") -> str:
        """Generate an empty chart with a message."""
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, message, ha='center', va='center', 
                fontsize=16, transform=plt.gca().transAxes)
        plt.title("Chart", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_string = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return base64_string

# Initialize chart generator
chart_generator = ChartGenerator()
