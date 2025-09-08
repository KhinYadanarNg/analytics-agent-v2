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
            "description": "Calculates the success and fail rate for a file and returns chart data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file, e.g., 'customer_sample_values.csv'."
                    }
                },
                "required": ["file_name"]
            }
        }
    }
]
