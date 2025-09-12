import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ChartGenerator:
    @staticmethod
    def generate_bar_chart_base64(chart_data: List[Dict[str, Any]], title: str = "Success/Fail Rate", show_only: str = "both", total_records: int = None, file_name: str = None) -> str:
        """
        Generate a bar chart from chart data and return as base64 string.
        
        Args:
            chart_data: List of dicts with 'status', 'percentage', 'count' keys
            title: Chart title
            show_only: Filter to show only specific status ('success', 'fail', or 'both')
            total_records: Total number of records to display in the chart
            file_name: Name of the file being analyzed
            
        Returns:
            Base64 encoded PNG image string
        """
        if not chart_data:
            # This should not be reached as empty data is handled at plan executor level
            # But keeping for backward compatibility
            return None
        
        # Calculate total records from chart data if not provided
        if total_records is None:
            total_records = sum(item.get("count", 0) for item in chart_data)
        
        # If total records is 0, return None (empty chart)
        if total_records == 0:
            return None
        
        # Filter chart data based on show_only parameter
        if show_only != "both":
            filtered_data = [item for item in chart_data if item.get("status", "").lower() == show_only.lower()]
            if not filtered_data:
                # If no data exists for the requested status, create a 0% entry
                # This handles cases where we want to show "0% success" instead of "No data"
                filtered_data = [{"status": show_only, "percentage": 0.0, "count": 0}]
            chart_data = filtered_data
        
        # Extract data for plotting
        labels = [item.get("status", "Unknown") for item in chart_data]
        percentages = [item.get("percentage", 0) for item in chart_data]
        counts = [item.get("count", 0) for item in chart_data]
        
        # Create the plot with better sizing
        plt.figure(figsize=(12, 8))
        
        # Define colors to match your image
        colors = []
        for label in labels:
            if label.lower() == 'success':
                colors.append('#4CAF50')  # Green for success
            elif label.lower() == 'fail':
                colors.append('#E91E63')  # Pink for fail (matching your image)
            else:
                colors.append('#9E9E9E')  # Gray for unknown
        
        # Ensure 0% bars are visible by giving them minimum height
        display_percentages = []
        for p in percentages:
            if p == 0:
                display_percentages.append(0.5)  # Minimum visible height for 0% bars
            else:
                display_percentages.append(p)
        
        bars = plt.bar(labels, display_percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars with better formatting
        for i, (bar, count, percentage) in enumerate(zip(bars, counts, percentages)):
            # For 0% bars, position text higher to make it visible
            if percentage == 0:
                y_position = 2  # Position above the x-axis
                va_alignment = 'bottom'
            else:
                y_position = bar.get_height() + 0.5
                va_alignment = 'bottom'
            
            plt.text(bar.get_x() + bar.get_width()/2, y_position,
                    f'{percentage:.1f}%\n({count} records)', 
                    ha='center', va=va_alignment, fontsize=12, fontweight='bold')
        
        plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Status', fontsize=14, fontweight='bold')
        
        # Create dynamic title based on show_only and file_name
        if file_name:
            if show_only == "success":
                chart_title = f"Success Rate for {file_name}"
            elif show_only == "fail":
                chart_title = f"Fail Rate for {file_name}"
            else:
                chart_title = f"Success/Fail Rate for {file_name}"
        else:
            chart_title = title
        
        # Enhanced title with total records subtitle
        if total_records is not None:
            full_title = f"{chart_title}\nTotal Records: {total_records:,}"
        else:
            full_title = chart_title
            
        plt.title(full_title, fontsize=16, fontweight='bold', pad=30)
        
        # Fix y-axis limits to ensure 0% bars are visible
        max_percentage = max(percentages) if percentages else 0
        if max_percentage == 0:
            # When all percentages are 0, set a reasonable y-axis limit
            plt.ylim(0, 10)  # Show 0-10% range so 0% bars are visible
        else:
            plt.ylim(0, max_percentage * 1.2)
        
        # Add total records box in top-right corner (matching your image style)
        if total_records is not None:
            plt.text(0.97, 0.92, f"Total: {total_records:,} records", 
                    transform=plt.gca().transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                             edgecolor='black', alpha=0.9),
                    fontsize=12, fontweight='bold')
        
        # Add subtle grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Improve layout and styling
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
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
