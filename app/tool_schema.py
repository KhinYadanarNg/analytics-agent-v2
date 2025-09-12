tools = [
    {
        "type": "function",
        "function": {
            "name": "list_available_files",
            "description": "Lists all available data files in the system.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_success_rate_by_file_name",
            "description": "Calculates the success and fail rate for a file and returns chart data. Can filter to show only success rate or both rates. Supports date range filtering.",
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
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date for filtering in YYYY-MM-DD format. Optional."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date for filtering in YYYY-MM-DD format. Optional."
                    }
                },
                "required": ["file_name"]
            }
        }
    }
]
