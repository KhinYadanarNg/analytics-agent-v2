tools = [
    {
        "type": "function",
        "function": {
            "name": "get_records_by_status",
            "description": "Retrieves records by file name and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file, e.g., 'customer.csv'."
                    },
                    "status": {
                        "type": "string",
                        "description": "The status to filter records by, e.g., 'success'."
                    }
                },
                "required": ["file_name", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_success_rate_by_file_name",
            "description": "Calculates the success and fail rate for a file and returns chart data. Can filter to show only success rate or both rates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file, e.g., 'customer_sample_values.csv'."
                    },
                    "show_only": {
                        "type": "string",
                        "description": "Filter to show only specific status. Options: 'success', 'fail', or 'both'. Default is 'both'.",
                        "enum": ["success", "fail", "both"]
                    }
                },
                "required": ["file_name"]
            }
        }
    }
]

# Simple function schema mapping (single source of truth for required params)
function_schemas = {
    "get_records_by_status": {"required": ["file_name", "status"]},
    "get_success_rate_by_file_name": {"required": ["file_name"]},
    "list_available_files": {"required": []}
}
