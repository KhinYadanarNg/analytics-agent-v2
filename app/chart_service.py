import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
from io import BytesIO
from typing import Dict, Any, List
import json

class ChartService:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    async def generate_chart(self, data: List[Dict], chart_type: str, columns: List[str]) -> Dict[str, Any]:
        """Generate chart from query results"""
        
        if not data:
            return {
                "success": False,
                "error": "No data to visualize"
            }
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            if chart_type == "bar":
                chart_data = self._generate_bar_chart(df)
            elif chart_type == "line":
                chart_data = self._generate_line_chart(df)
            elif chart_type == "pie":
                chart_data = self._generate_pie_chart(df)
            else:  # default to table
                return self._generate_table_view(df)
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "success": True,
                "chart_type": chart_type,
                "chart_image": f"data:image/png;base64,{chart_base64}",
                "chart_data": chart_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Chart generation failed: {str(e)}"
            }
    
    def _generate_bar_chart(self, df: pd.DataFrame) -> Dict:
        """Generate bar chart"""
        # Use first column as x-axis, second as y-axis
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            plt.bar(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{y_col} by {x_col}')
            plt.xticks(rotation=45)
            
            return {
                "x_axis": x_col,
                "y_axis": y_col,
                "data_points": len(df)
            }
    
    def _generate_line_chart(self, df: pd.DataFrame) -> Dict:
        """Generate line chart"""
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            plt.plot(df[x_col], df[y_col], marker='o')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{y_col} over {x_col}')
            plt.xticks(rotation=45)
            
            return {
                "x_axis": x_col,
                "y_axis": y_col,
                "data_points": len(df)
            }
    
    def _generate_pie_chart(self, df: pd.DataFrame) -> Dict:
        """Generate pie chart"""
        if len(df.columns) >= 2:
            labels_col = df.columns[0]
            values_col = df.columns[1]
            
            plt.pie(df[values_col], labels=df[labels_col], autopct='%1.1f%%')
            plt.title(f'{values_col} Distribution by {labels_col}')
            
            return {
                "labels": labels_col,
                "values": values_col,
                "categories": len(df)
            }
    
    def _generate_table_view(self, df: pd.DataFrame) -> Dict:
        """Generate table view (no chart)"""
        return {
            "success": True,
            "chart_type": "table",
            "table_data": df.to_dict('records'),
            "columns": df.columns.tolist(),
            "row_count": len(df)
        }

# Initialize chart service
chart_service = ChartService()
