"""
Enhanced chart generation service with support for filtered reports.
Generates chart images based on success/failure rates with filtering options.
"""
import io
import base64
import logging
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger("chart_generator")

class ChartGenerator:
    """Generates chart images from analytics data with filtering support."""
    
    def __init__(self):
        # Set style for better-looking charts
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
    def generate_chart(self, 
                      chart_data: List[Dict], 
                      chart_type: str = "bar",
                      title: Optional[str] = None,
                      file_name: Optional[str] = None,
                      report_type: str = "both") -> str:
        """
        Generate a chart image from the data and return as base64 string.
        
        Args:
            chart_data: List of dicts with 'status', 'percentage', 'count'
            chart_type: Type of chart ('bar', 'pie', 'donut', 'line', 'stacked')
            title: Optional custom title for the chart
            file_name: Optional file name to include in title
            report_type: Type of report ("success", "failure", or "both")
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Validate and prepare data
            if not chart_data:
                return self._generate_no_data_image(report_type)
            
            # Extract data
            statuses = []
            percentages = []
            counts = []
            
            for item in chart_data:
                statuses.append(item.get('status', '').capitalize())
                percentages.append(float(item.get('percentage', 0)))
                counts.append(int(item.get('count', 0)))
            
            # Generate chart based on type
            if chart_type.lower() == "pie":
                img_base64 = self._generate_pie_chart(statuses, percentages, counts, title, file_name, report_type)
            elif chart_type.lower() == "donut":
                img_base64 = self._generate_donut_chart(statuses, percentages, counts, title, file_name, report_type)
            elif chart_type.lower() == "line":
                img_base64 = self._generate_line_chart(statuses, percentages, counts, title, file_name, report_type)
            elif chart_type.lower() == "stacked":
                img_base64 = self._generate_stacked_bar_chart(statuses, percentages, counts, title, file_name, report_type)
            else:  # Default to bar chart
                img_base64 = self._generate_bar_chart(statuses, percentages, counts, title, file_name, report_type)
            
            return img_base64
            
        except Exception as e:
            logger.exception(f"Error generating chart: {e}")
            return self._generate_error_image(str(e))
    
    def _generate_bar_chart(self, statuses, percentages, counts, title, file_name, report_type):
        """Generate a bar chart with appropriate coloring based on report type."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine colors based on report type and status
        colors = []
        for s in statuses:
            if report_type == "success":
                colors.append('#28a745')  # Green for success only
            elif report_type == "failure":
                colors.append('#dc3545')  # Red for failure only
            else:
                # Both - use appropriate colors
                colors.append('#28a745' if s.lower() == 'success' else '#dc3545')
        
        # Create bars
        bars = ax.bar(statuses, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, percentage, count in zip(bars, percentages, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage:.1f}%\n({count})',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize chart
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Status', fontsize=12, fontweight='bold')
        
        # Set title based on report type
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            title_text = self._get_title_text(report_type, file_name)
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Set y-axis limit
        if report_type != "both" and percentages:
            # For single status reports, adjust scale
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, max(100, max(percentages) * 1.1) if percentages else 100)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _generate_pie_chart(self, statuses, percentages, counts, title, file_name, report_type):
        """Generate a pie chart for the specified report type."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter out zero values
        non_zero_data = [(s, p, c) for s, p, c in zip(statuses, percentages, counts) if p > 0]
        
        if not non_zero_data:
            return self._generate_no_data_image(report_type)
        
        statuses, percentages, counts = zip(*non_zero_data)
        
        # Colors based on report type
        colors = []
        for s in statuses:
            if report_type == "success":
                colors.append('#28a745')
            elif report_type == "failure":
                colors.append('#dc3545')
            else:
                colors.append('#28a745' if s.lower() == 'success' else '#dc3545')
        
        # For single-status reports, show the percentage vs remaining
        if report_type != "both" and len(statuses) == 1:
            # Add complementary slice
            complementary_pct = 100 - percentages[0]
            if complementary_pct > 0:
                statuses = list(statuses) + ["Other"]
                percentages = list(percentages) + [complementary_pct]
                counts = list(counts) + [0]  # No specific count for "Other"
                colors.append('#cccccc')  # Gray for other
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(percentages, 
                                          labels=statuses,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          explode=[0.05] * len(statuses),
                                          shadow=True)
        
        # Enhance text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext, count in zip(autotexts, counts):
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
            if count > 0:  # Only show count for actual data
                current_text = autotext.get_text()
                autotext.set_text(f'{current_text}\n({count})')
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            title_text = self._get_title_text(report_type, file_name)
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_donut_chart(self, statuses, percentages, counts, title, file_name, report_type):
        """Generate a donut chart for the specified report type."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter out zero values
        non_zero_data = [(s, p, c) for s, p, c in zip(statuses, percentages, counts) if p > 0]
        
        if not non_zero_data:
            return self._generate_no_data_image(report_type)
        
        statuses, percentages, counts = zip(*non_zero_data)
        
        # Colors based on report type
        colors = []
        for s in statuses:
            if report_type == "success":
                colors.append('#28a745')
            elif report_type == "failure":
                colors.append('#dc3545')
            else:
                colors.append('#28a745' if s.lower() == 'success' else '#dc3545')
        
        # For single-status reports, show the percentage vs remaining
        total_count = sum(counts)
        if report_type != "both" and len(statuses) == 1:
            complementary_pct = 100 - percentages[0]
            if complementary_pct > 0:
                statuses = list(statuses) + ["Other"]
                percentages = list(percentages) + [complementary_pct]
                counts = list(counts) + [0]
                colors.append('#cccccc')
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(percentages,
                                          labels=statuses,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          pctdistance=0.85,
                                          explode=[0.05] * len(statuses),
                                          shadow=True)
        
        # Create circle for donut hole
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # Add center text based on report type
        center_text = self._get_center_text(report_type, total_count)
        ax.text(0, 0, center_text, 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Enhance text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext, count in zip(autotexts, counts):
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
            if count > 0:
                current_text = autotext.get_text()
                autotext.set_text(f'{current_text}\n({count})')
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            title_text = self._get_title_text(report_type, file_name)
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _get_title_text(self, report_type, file_name):
        """Generate appropriate title based on report type."""
        if report_type == "success":
            title_text = 'Success Rate Analysis'
        elif report_type == "failure":
            title_text = 'Failure Rate Analysis'
        else:
            title_text = 'Success/Failure Rate Analysis'
        
        if file_name:
            title_text += f'\nFile: {file_name}'
        
        return title_text
    
    def _get_center_text(self, report_type, total_count):
        """Generate center text for donut charts based on report type."""
        if report_type == "success":
            return f'Success\n{total_count}'
        elif report_type == "failure":
            return f'Failures\n{total_count}'
        else:
            return f'Total\n{total_count}'
    
    def _generate_line_chart(self, statuses, percentages, counts, title, file_name, report_type):
        """Generate a line chart for the specified report type."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(statuses))
        
        # Determine line color based on report type
        if report_type == "success":
            line_color = '#28a745'
            marker_face = '#28a745'
            marker_edge = '#1e7e34'
        elif report_type == "failure":
            line_color = '#dc3545'
            marker_face = '#dc3545'
            marker_edge = '#bd2130'
        else:
            line_color = '#007bff'
            marker_face = '#28a745'
            marker_edge = '#dc3545'
        
        # Create line plot
        ax.plot(x_pos, percentages, marker='o', linewidth=2, markersize=10, 
               color=line_color, markerfacecolor=marker_face, markeredgewidth=2,
               markeredgecolor=marker_edge)
        
        # Add value labels
        for x, y, count in zip(x_pos, percentages, counts):
            ax.annotate(f'{y:.1f}%\n({count})', 
                       xy=(x, y), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=11,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Customize chart
        ax.set_xticks(x_pos)
        ax.set_xticklabels(statuses)
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Status', fontsize=12, fontweight='bold')
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            title_text = self._get_title_text(report_type, file_name)
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Set y-axis limit
        ax.set_ylim(0, max(100, max(percentages) * 1.1) if percentages else 100)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_stacked_bar_chart(self, statuses, percentages, counts, title, file_name, report_type):
        """Generate a horizontal stacked bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for stacked bar
        success_pct = 0
        fail_pct = 0
        success_count = 0
        fail_count = 0
        
        for status, pct, cnt in zip(statuses, percentages, counts):
            if status.lower() == 'success':
                success_pct = pct
                success_count = cnt
            else:
                fail_pct = pct
                fail_count = cnt
        
        # Create horizontal stacked bar based on report type
        bar_height = 0.5
        y_pos = [0]
        
        if report_type == "success":
            # Only show success bar
            if success_pct > 0:
                success_bar = ax.barh(y_pos, success_pct, bar_height, 
                                     label='Success', color='#28a745', alpha=0.8)
                ax.text(success_pct/2, 0, f'Success\n{success_pct:.1f}% ({success_count})', 
                       ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        elif report_type == "failure":
            # Only show failure bar
            if fail_pct > 0:
                fail_bar = ax.barh(y_pos, fail_pct, bar_height,
                                  label='Fail', color='#dc3545', alpha=0.8)
                ax.text(fail_pct/2, 0, f'Fail\n{fail_pct:.1f}% ({fail_count})',
                       ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        else:
            # Show both stacked
            if success_pct > 0:
                success_bar = ax.barh(y_pos, success_pct, bar_height, 
                                     label='Success', color='#28a745', alpha=0.8)
                ax.text(success_pct/2, 0, f'Success\n{success_pct:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white', fontsize=12)
            
            if fail_pct > 0:
                fail_bar = ax.barh(y_pos, fail_pct, bar_height,
                                  left=success_pct, label='Fail', color='#dc3545', alpha=0.8)
                ax.text(success_pct + fail_pct/2, 0, f'Fail\n{fail_pct:.1f}%',
                       ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        
        # Customize chart
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            title_text = self._get_title_text(report_type, file_name)
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_no_data_image(self, report_type="both"):
        """Generate a placeholder image when no data is available."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if report_type == "success":
            message = 'No Success Data Available'
        elif report_type == "failure":
            message = 'No Failure Data Available'
        else:
            message = 'No Data Available'
            
        ax.text(0.5, 0.5, message, 
               ha='center', va='center', fontsize=20, fontweight='bold',
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_error_image(self, error_msg):
        """Generate an error image."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error generating chart:\n{error_msg}', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Close figure to free memory
        return img_base64
    
    def detect_chart_type_from_prompt(self, prompt: str) -> str:
        """
        Detect the desired chart type from user prompt.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Chart type string ('bar', 'pie', 'donut', 'line', 'stacked')
        """
        prompt_lower = prompt.lower()
        
        # Check for specific chart type mentions
        if any(word in prompt_lower for word in ['pie chart', 'pie graph', 'pie']):
            return 'pie'
        elif any(word in prompt_lower for word in ['donut chart', 'donut graph', 'donut', 'doughnut']):
            return 'donut'
        elif any(word in prompt_lower for word in ['line chart', 'line graph', 'trend', 'timeline']):
            return 'line'
        elif any(word in prompt_lower for word in ['stacked', 'stacked bar', 'horizontal bar']):
            return 'stacked'
        elif any(word in prompt_lower for word in ['bar chart', 'bar graph', 'bars']):
            return 'bar'
        
        # Default to bar chart
        return 'bar'

# Initialize chart generator
chart_generator = ChartGenerator()