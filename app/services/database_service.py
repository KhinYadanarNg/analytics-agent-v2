import boto3
import logging
from typing import Dict, Any, List, Optional
import os

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

class DatabaseService:
    logger = logging.getLogger("database_service")

    async def get_success_rate_by_file_name(self, file_name: Optional[str] = None, org_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the percentage of success and fail records for a given file_id or file_name in the tracker table.

        If `file_id` is not provided, `file_name` will be used to look up the file id via `get_file_id_by_name`.
        If `org_id` is provided, results will be filtered to that organization.
        If start_date or end_date are provided, results will be filtered by created_date range.

        Returns a dict with chart data for visualization and includes the resolved `file_id` and `org_id` for debugging.
        """
        # Initialize file_id at the beginning, outside try block
        file_id = None
        
        try:
            # Resolve file_id from file_name when needed
            if file_name:
                file_id = await self.get_file_id_by_name(file_name)
                self.logger.info("Resolved file_name '%s' -> file_id '%s'", file_name, file_id)

            if not file_id:
                self.logger.error("Could not resolve file_id from file_name: %s", file_name)
                return {
                    "success": False,
                    "error": f"Could not resolve file_id from file_name: {file_name}",
                    "chart_data": [],
                    "row_count": 0,
                    "file_name": file_name,
                    "org_id": org_id
                }

            # Build filter expression
            filter_expr = boto3.dynamodb.conditions.Attr('file_id').eq(file_id)
            self.logger.info("Base filter: file_id=%s", file_id)
            
            # Always filter out records with empty/null organization_id
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').exists()
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').ne('')
            self.logger.info("Added filters to exclude empty/null organization_id")
            
            if org_id:
                # Log the org_id being used for filtering
                self.logger.info("Applying organization filter with org_id: %s", org_id)
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').eq(org_id)
                self.logger.debug("Complete filter: file_id=%s AND organization_id=%s (excluding empty orgs)", file_id, org_id)
            else:
                self.logger.warning("No org_id provided - but will still exclude records with empty organization_id")
            
            # Add date range filtering
            if start_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').gte(start_date)
                self.logger.debug("Adding start_date filter: %s", start_date)
            if end_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').lte(end_date)
                self.logger.debug("Adding end_date filter: %s", end_date)

            self.logger.info("Executing DynamoDB scan with filters...")
            response = self.tracker_table.scan(
                FilterExpression=filter_expr
            )

            items = response.get('Items', [])
            total = len(items)
            self.logger.info("DynamoDB returned %d items for file_id=%s org_id=%s", total, file_id, org_id)
            
            # Log organization IDs of returned items to debug filtering
            if items:
                org_ids_found = []
                for item in items:
                    item_org_id = item.get('organization_id', 'NO_ORG_ID')
                    if item_org_id not in org_ids_found:
                        org_ids_found.append(item_org_id)
                self.logger.info("Organization IDs in results: %s", org_ids_found)
                
                if org_id and len(org_ids_found) > 1:
                    self.logger.error("ERROR: Multiple organizations found when filtering by org_id=%s: %s", org_id, org_ids_found)
            
            if total == 0:
                # Build descriptive message based on filters applied
                message_parts = ["No records found for this file"]
                if start_date and end_date:
                    if start_date == end_date:
                        message_parts.append(f"on {start_date}")
                    else:
                        message_parts.append(f"between {start_date} and {end_date}")
                elif start_date:
                    message_parts.append(f"from {start_date} onwards")
                elif end_date:
                    message_parts.append(f"until {end_date}")
                
                return {
                    "success": True,
                    "chart_data": [],
                    "row_count": 0,
                    "message": " ".join(message_parts) + ".",
                    "file_id": file_id,
                    "file_name": file_name,
                    "org_id": org_id,
                    "date_filter": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }

            success_count = 0
            fail_count = 0
            status_values = []
            unknown_status_count = 0
            
            for item in items:
                status = item.get('final_status', None)
                item_org_id = item.get('organization_id', 'NO_ORG_ID')
                
                if status is not None:
                    status_clean = str(status).strip().lower()
                    status_values.append(f"{status_clean}(org:{item_org_id})")
                    
                    if status_clean == 'success':
                        success_count += 1
                    elif status_clean == 'fail':
                        fail_count += 1
                    else:
                        unknown_status_count += 1
                        self.logger.warning("Unknown status found: '%s' for org_id=%s", status_clean, item_org_id)
                else:
                    unknown_status_count += 1
                    self.logger.warning("Item with missing final_status for org_id=%s", item_org_id)

            self.logger.info("Status breakdown: Success=%d, Fail=%d, Unknown=%d", success_count, fail_count, unknown_status_count)
            self.logger.info("Detailed status values: %s", status_values)
            
            # Calculate percentages
            if total > 0:
                success_rate = round((success_count / total) * 100, 2)
                fail_rate = round((fail_count / total) * 100, 2)
            else:
                success_rate = fail_rate = 0
            self.logger.info("Computed success=%s%% (%d) fail=%s%% (%d) out of total=%d for file_id=%s org_id=%s", 
                           success_rate, success_count, fail_rate, fail_count, total, file_id, org_id)
            chart_data = self.format_chart_data(success_rate, success_count, fail_rate, fail_count)

            return {
                "success": True,
                "chart_data": chart_data,
                "row_count": total,
                "chart_type": "bar",
                "file_id": file_id,
                "file_name": file_name,
                "org_id": org_id
            }
        except Exception as e:
            self.logger.exception("Error in get_success_rate_by_file_name for file_name: %s", file_name)
            return {
                "success": False,
                "error": str(e),
                "chart_data": [],
                "row_count": 0,
                "file_id": file_id,
                "file_name": file_name,
                "org_id": org_id
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
    
    async def get_success_rate_by_domain_name(self, domain_name: Optional[str] = None, org_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the percentage of success and fail records for a given domain_name in the tracker table.

        If `org_id` is provided, results will be filtered to that organization.
        If start_date or end_date are provided, results will be filtered by created_date range.

        Returns a dict with chart data for visualization and includes the domain_name and org_id for debugging.
        """
        try:
            if not domain_name:
                self.logger.error("Domain name is required")
                return {
                    "success": False,
                    "error": "Domain name is required",
                    "chart_data": [],
                    "row_count": 0,
                    "domain_name": domain_name,
                    "org_id": org_id
                }

            # Build filter expression
            filter_expr = boto3.dynamodb.conditions.Attr('domain_name').eq(domain_name)
            self.logger.info("Base filter: domain_name=%s", domain_name)
            
            # Always filter out records with empty/null organization_id
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').exists()
            filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').ne('')
            self.logger.info("Added filters to exclude empty/null organization_id")
            
            if org_id:
                # Log the org_id being used for filtering
                self.logger.info("Applying organization filter with org_id: %s", org_id)
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('organization_id').eq(org_id)
                self.logger.debug("Complete filter: domain_name=%s AND organization_id=%s (excluding empty orgs)", domain_name, org_id)
            else:
                self.logger.warning("No org_id provided - but will still exclude records with empty organization_id")
            
            # Add date range filtering
            if start_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').gte(start_date)
                self.logger.debug("Adding start_date filter: %s", start_date)
            if end_date:
                filter_expr = filter_expr & boto3.dynamodb.conditions.Attr('created_date').lte(end_date)
                self.logger.debug("Adding end_date filter: %s", end_date)

            self.logger.info("Executing DynamoDB scan with filters...")
            response = self.tracker_table.scan(
                FilterExpression=filter_expr
            )

            items = response.get('Items', [])
            total = len(items)
            self.logger.info("DynamoDB returned %d items for domain_name=%s org_id=%s", total, domain_name, org_id)
            
            # Log organization IDs of returned items to debug filtering
            if items:
                org_ids_found = []
                for item in items:
                    item_org_id = item.get('organization_id', 'NO_ORG_ID')
                    if item_org_id not in org_ids_found:
                        org_ids_found.append(item_org_id)
                self.logger.info("Organization IDs in results: %s", org_ids_found)
                
                if org_id and len(org_ids_found) > 1:
                    self.logger.error("ERROR: Multiple organizations found when filtering by org_id=%s: %s", org_id, org_ids_found)
            
            if total == 0:
                # Build descriptive message based on filters applied
                message_parts = ["No records found for this domain"]
                if start_date and end_date:
                    if start_date == end_date:
                        message_parts.append(f"on {start_date}")
                    else:
                        message_parts.append(f"between {start_date} and {end_date}")
                elif start_date:
                    message_parts.append(f"from {start_date} onwards")
                elif end_date:
                    message_parts.append(f"until {end_date}")
                
                return {
                    "success": True,
                    "chart_data": [],
                    "row_count": 0,
                    "message": " ".join(message_parts) + ".",
                    "domain_name": domain_name,
                    "org_id": org_id,
                    "date_filter": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }

            success_count = 0
            fail_count = 0
            status_values = []
            unknown_status_count = 0
            
            for item in items:
                status = item.get('final_status', None)
                item_org_id = item.get('organization_id', 'NO_ORG_ID')
                
                if status is not None:
                    status_clean = str(status).strip().lower()
                    status_values.append(f"{status_clean}(org:{item_org_id})")
                    
                    if status_clean == 'success':
                        success_count += 1
                    elif status_clean == 'fail':
                        fail_count += 1
                    else:
                        unknown_status_count += 1
                        self.logger.warning("Unknown status found: '%s' for org_id=%s", status_clean, item_org_id)
                else:
                    unknown_status_count += 1
                    self.logger.warning("Item with missing final_status for org_id=%s", item_org_id)

            self.logger.info("Status breakdown: Success=%d, Fail=%d, Unknown=%d", success_count, fail_count, unknown_status_count)
            self.logger.info("Detailed status values: %s", status_values)
            
            # Calculate percentages
            if total > 0:
                success_rate = round((success_count / total) * 100, 2)
                fail_rate = round((fail_count / total) * 100, 2)
            else:
                success_rate = fail_rate = 0
            self.logger.info("Computed success=%s%% (%d) fail=%s%% (%d) out of total=%d for domain_name=%s org_id=%s", 
                           success_rate, success_count, fail_rate, fail_count, total, domain_name, org_id)
            chart_data = self.format_chart_data(success_rate, success_count, fail_rate, fail_count)

            return {
                "success": True,
                "chart_data": chart_data,
                "row_count": total,
                "chart_type": "bar",
                "domain_name": domain_name,
                "org_id": org_id
            }
        except Exception as e:
            self.logger.exception("Error in get_success_rate_by_domain_name for domain_name: %s", domain_name)
            return {
                "success": False,
                "error": str(e),
                "chart_data": [],
                "row_count": 0,
                "domain_name": domain_name,
                "org_id": org_id
            }
        
    async def get_file_id_by_name(self, file_name: str) -> Optional[str]:
        """
        Retrieve file_id from header table using file_name.
        Requires a GSI on file_name in the header table.
        """
        try:
            response = self.header_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('file_name').eq(file_name)
            )
            items = response.get('Items', [])
            if items:
                return items[0].get('id')
            return None
        except Exception as e:
            self.logger.exception("Error in get_file_id_by_name for file_name: %s", file_name)
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

# Initialize database service
db_service = DatabaseService()