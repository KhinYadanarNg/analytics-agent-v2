import boto3
from typing import Dict, Any, List, Optional
import os

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")

aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

class DatabaseService:
    async def get_success_rate_by_file_id(self, file_id: str) -> Dict[str, Any]:
        """
        Calculate the percentage of success and fail records for a given file_id in the tracker table.
        Returns a dict with chart data for visualization, ready for LLM chart creation.
        Pass the returned dict directly to the LLM or your charting tool.
        """
        try:
            response = self.tracker_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('file_id').eq(file_id)
            )
            items = response.get('Items', [])
            print(f"[DEBUG] Items returned for file_id {file_id}: {items}")
            total = len(items)
            if total == 0:
                return {
                    "success": True,
                    "chart_data": [],
                    "row_count": 0,
                    "message": "No records found for this file_id."
                }
            success_count = 0
            fail_count = 0
            for item in items:
                status = item.get('final_status', None)
                print(f"[DEBUG] Item final_status: {status}")
                if status is not None:
                    status_clean = str(status).strip().lower()
                    if status_clean == 'success':
                        success_count += 1
                    elif status_clean == 'fail':
                        fail_count += 1
            success_rate = round((success_count / total) * 100, 2)
            fail_rate = round((fail_count / total) * 100, 2)
            chart_data = self.format_chart_data(success_rate, success_count, fail_rate, fail_count)
            print(f"[DEBUG] Success count: {success_count}, Fail count: {fail_count}, Total: {total}")
            return {
                "success": True,
                "chart_data": chart_data,
                "row_count": total,
                "chart_type": "bar"
            }
        except Exception as e:
            print(f"[ERROR] Exception in get_success_rate_by_file_id: {e}")
            return {
                "success": False,
                "error": str(e),
                "chart_data": [],
                "row_count": 0
            }

    @staticmethod
    def format_chart_data(success_rate, success_count, fail_rate, fail_count):
        """
        Helper to format chart data for LLM chart creation.
        """
        return [
            {"status": "success", "percentage": success_rate, "count": success_count},
            {"status": "fail", "percentage": fail_rate, "count": fail_count}
        ]
    async def get_file_id_by_name(self, file_name: str) -> Optional[str]:
       # print(f"Looking up file_id for file_name: {file_name}")
        """
        Retrieve file_id from header table using file_name.
        Requires a GSI on file_name in the header table.
        """
        try:
            response = self.header_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('file_name').eq(file_name)
            )
            #print(f"DynamoDB scan response: {response}")
            items = response.get('Items', [])
            if items:
                return items[0].get('id')
            return None
        except Exception as e:
            return None
        
    def __init__(self, tracker_table_name: str = "MasterDataTaskTrackerSIT", header_table_name: str = "MasterDataHeaderSIT"):
      
       
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        self.tracker_table = self.dynamodb.Table(tracker_table_name)
        self.header_table = self.dynamodb.Table(header_table_name)

    async def get_records_by_status(self, file_id: str, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve records from tracker table by file_id and optional final_status.
        Requires a GSI on file_id and final_status for efficient querying.
        """
        try:
            if status:
                response = self.tracker_table.scan(

                     FilterExpression=boto3.dynamodb.conditions.Attr('file_id').eq(file_id) &
                    boto3.dynamodb.conditions.Attr('final_status').eq(status)

                )
            else:
                response = self.tracker_table.scan(
                    FilterExpression=boto3.dynamodb.conditions.Attr('file_id').eq(file_id)
                )
            items = response.get('Items', [])
            return {
                "success": True,
                "data": items,
                "row_count": len(items)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "row_count": 0
            }

    async def get_file_status(self, file_id: str) -> Dict[str, Any]:
        """
        Retrieve file status from header table by file_id.
        """
        try:
            response = self.header_table.get_item(Key={"file_id": file_id})
            item = response.get('Item')
            if item:
                return {"success": True, "data": item}
            else:
                return {"success": False, "error": "File not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize database service
db_service = DatabaseService()
