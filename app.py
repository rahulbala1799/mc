import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-12345')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        session['filepath'] = filepath
        session['filename'] = filename
        
        return redirect(url_for('analyze'))
    
    flash('Invalid file type. Please upload an Excel file (.xlsx or .xls)')
    return redirect(url_for('index'))

@app.route('/analyze')
def analyze():
    filepath = session.get('filepath')
    filename = session.get('filename')
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    try:
        # Load Excel file with pandas
        df = pd.read_excel(filepath)
        
        # Basic analysis
        row_count = len(df)
        column_count = len(df.columns)
        columns = df.columns.tolist()
        
        # Create a preview table
        preview_html = df.head(10).to_html(classes='table table-striped table-hover', index=False)
        
        # Look for transaction-related columns
        transaction_columns = []
        common_transaction_terms = ['amount', 'transaction', 'date', 'description', 'category', 'payment']
        
        for col in columns:
            if any(term in str(col).lower() for term in common_transaction_terms):
                transaction_columns.append(col)
        
        is_transaction_data = len(transaction_columns) > 0
        
        return render_template(
            'analysis.html',
            filename=filename,
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            preview_html=preview_html,
            is_transaction_data=is_transaction_data,
            transaction_columns=transaction_columns
        )
        
    except Exception as e:
        flash(f'Error analyzing file: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 