# Excel Analyzer

A simple Flask web application that allows users to upload Excel files and analyze their content. Built with Flask and pandas.

## Features

- Upload Excel files (.xlsx and .xls)
- Analyze basic information (row count, column count, column names)
- Detect transaction-related data
- Preview first 10 rows of data
- Responsive user interface with Bootstrap

## Requirements

- Python 3.9+
- Flask
- pandas
- openpyxl
- gunicorn (for production)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rahulbala1799/mc.git
   cd mc
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Development

Run the application in development mode:

```
flask run
```

Then open your browser and go to http://localhost:5000

### Production

For production deployment, use Gunicorn:

```
gunicorn --workers=2 --bind 0.0.0.0:8080 app:app
```

## Deployment

This application is set up for deployment on Railway using Docker. The Dockerfile is configured to handle pandas installation properly.

## License

MIT 