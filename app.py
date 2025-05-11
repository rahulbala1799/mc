import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-12345')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_ticket_data(df):
    """Process ticket data for analysis"""
    # Convert date columns to datetime
    df['Logged - Date'] = pd.to_datetime(df['Logged - Date'], errors='coerce')
    df['Ticket solved - Date'] = pd.to_datetime(df['Ticket solved - Date'], errors='coerce')
    
    # Calculate resolution time in days
    df['Resolution Time (Days)'] = (df['Ticket solved - Date'] - df['Logged - Date']).dt.total_seconds() / (24 * 60 * 60)
    
    # Fill NaN regions with 'Unknown'
    df['Region'] = df['Region'].fillna('Unknown')
    
    return df

def calculate_resolution_metrics(df):
    """Calculate various resolution time metrics"""
    metrics = {}
    
    # Overall metrics
    all_tickets_mean = df['Resolution Time (Days)'].mean()
    solved_tickets = df[df['Ticket status'] == 'Solved']
    solved_tickets_mean = solved_tickets['Resolution Time (Days)'].mean()
    
    metrics['all_tickets_avg'] = round(all_tickets_mean, 1) if not pd.isna(all_tickets_mean) else "N/A"
    metrics['solved_tickets_avg'] = round(solved_tickets_mean, 1) if not pd.isna(solved_tickets_mean) else "N/A"
    
    # By region
    region_metrics = solved_tickets.groupby('Region')['Resolution Time (Days)'].mean().sort_values()
    metrics['by_region'] = region_metrics.round(1).to_dict()
    
    # By group
    group_metrics = solved_tickets.groupby('Level 3 Group')['Resolution Time (Days)'].mean().sort_values()
    metrics['by_group'] = group_metrics.round(1).to_dict()
    
    # By engineer
    engineer_metrics = solved_tickets.groupby('Assignee name')['Resolution Time (Days)'].mean().sort_values()
    metrics['by_engineer'] = engineer_metrics.round(1).to_dict()
    
    # By priority
    priority_metrics = solved_tickets.groupby('Priority')['Resolution Time (Days)'].mean().sort_values()
    metrics['by_priority'] = priority_metrics.round(1).to_dict()
    
    # Tickets count
    metrics['total_tickets'] = len(df)
    metrics['solved_tickets'] = len(solved_tickets)
    metrics['open_tickets'] = len(df[df['Ticket status'] == 'Open'])
    metrics['hold_tickets'] = len(df[df['Ticket status'] == 'Hold'])
    
    return metrics

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
        
        # Try to detect if it's ticket data
        try:
            df = pd.read_excel(filepath)
            expected_columns = ['Ticket ID', 'Logged - Date', 'Ticket status', 'Region', 
                               'Assignee name', 'Level 3 Group', 'Ticket solved - Date', 'Priority']
            
            # Check if this is ticket data
            if all(col in df.columns for col in expected_columns):
                session['is_ticket_data'] = True
            else:
                session['is_ticket_data'] = False
        except:
            session['is_ticket_data'] = False
        
        return redirect(url_for('analyze'))
    
    flash('Invalid file type. Please upload an Excel file (.xlsx or .xls)')
    return redirect(url_for('index'))

@app.route('/analyze')
def analyze():
    filepath = session.get('filepath')
    filename = session.get('filename')
    is_ticket_data = session.get('is_ticket_data', False)
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    try:
        # Load Excel file with pandas
        df = pd.read_excel(filepath)
        
        # If it's ticket data, redirect to the ticket analysis page
        if is_ticket_data:
            return redirect(url_for('ticket_overview'))
        
        # Otherwise show general analysis
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

@app.route('/ticket_overview')
def ticket_overview():
    filepath = session.get('filepath')
    filename = session.get('filename')
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    try:
        # Load and process ticket data
        df = pd.read_excel(filepath)
        df = process_ticket_data(df)
        
        # Calculate metrics
        all_tickets_mean = df['Resolution Time (Days)'].mean()
        solved_tickets = df[df['Ticket status'] == 'Solved']
        solved_tickets_mean = solved_tickets['Resolution Time (Days)'].mean()
        
        # Calculate percentage of tickets solved within SLA (7 days as example)
        within_sla = solved_tickets[solved_tickets['Resolution Time (Days)'] <= 7]
        within_sla_percentage = round((len(within_sla) / len(solved_tickets)) * 100) if len(solved_tickets) > 0 else 0
        
        # Calculate percentages for ticket status
        total_tickets = len(df)
        solved_count = len(solved_tickets)
        open_count = len(df[df['Ticket status'] == 'Open'])
        hold_count = len(df[df['Ticket status'] == 'Hold'])
        
        solved_percentage = round((solved_count / total_tickets) * 100) if total_tickets > 0 else 0
        open_percentage = round((open_count / total_tickets) * 100) if total_tickets > 0 else 0
        hold_percentage = round((hold_count / total_tickets) * 100) if total_tickets > 0 else 0
        
        # By region
        region_metrics = solved_tickets.groupby('Region')['Resolution Time (Days)'].mean().sort_values()
        
        # By group
        group_metrics = solved_tickets.groupby('Level 3 Group')['Resolution Time (Days)'].mean().sort_values()
        
        # By engineer
        engineer_metrics = solved_tickets.groupby('Assignee name')['Resolution Time (Days)'].mean().sort_values()
        
        # By priority
        priority_metrics = solved_tickets.groupby('Priority')['Resolution Time (Days)'].mean().sort_values()

        # --- Prepare data for Dashboard Charts ---
        
        # Add month and year columns for analysis
        df['Month'] = pd.to_datetime(df['Logged - Date']).dt.strftime('%Y-%m')
        
        # Open vs Solved by Month
        monthly_status = df.groupby(['Month', 'Ticket status']).size().unstack(fill_value=0).reset_index()
        if 'Solved' not in monthly_status.columns:
            monthly_status['Solved'] = 0
        if 'Open' not in monthly_status.columns:
            monthly_status['Open'] = 0
        if 'Hold' not in monthly_status.columns:
            monthly_status['Hold'] = 0
            
        monthly_data = {
            'labels': monthly_status['Month'].tolist(),
            'solved': monthly_status['Solved'].tolist(),
            'open': monthly_status['Open'].tolist(),
            'hold': monthly_status['Hold'].tolist()
        }
        
        # Backlog Analysis with error handling
        try:
            # Sort DataFrame by month to process chronologically
            df_sorted = df.sort_values('Logged - Date')
            
            # Ensure Ticket ID exists and is not null
            if 'Ticket ID' not in df.columns or df['Ticket ID'].isnull().any():
                raise ValueError("Missing or null Ticket IDs found")
            
            # Get unique months in sorted order
            unique_months = df_sorted['Month'].dropna().unique().tolist()
            
            if not unique_months:
                raise ValueError("No valid months found for backlog analysis")
            
            # Initialize backlog tracking
            backlog_data = {
                'labels': [],
                'backlog_count': [],
                'new_tickets': [],
                'solved_tickets': [],
                'priority_breakdown': [],
                'region_breakdown': []
            }
            
            # Track tickets present in each month
            current_tickets = set()
            backlog_tickets = set()
            
            for month in unique_months:
                # Get tickets from this month (with validation)
                month_df = df[df['Month'] == month]
                if month_df.empty:
                    continue
                
                # New tickets this month
                new_ticket_ids = set(month_df['Ticket ID'].dropna().astype(str).tolist())
                new_count = len(new_ticket_ids)
                
                # Solved tickets this month
                solved_month_df = month_df[month_df['Ticket status'] == 'Solved']
                solved_ticket_ids = set(solved_month_df['Ticket ID'].dropna().astype(str).tolist())
                solved_count = len(solved_ticket_ids)
                
                # Update current tickets
                current_tickets = current_tickets.union(new_ticket_ids)
                
                # Remove solved tickets from tracking
                current_tickets = current_tickets - solved_ticket_ids
                
                # Backlog is anything in current_tickets that wasn't new this month
                backlog_tickets = current_tickets - new_ticket_ids
                backlog_count = len(backlog_tickets)
                
                # Get backlog tickets from the DataFrame (with validation)
                backlog_df = df[df['Ticket ID'].astype(str).isin(backlog_tickets)]
                
                # Get priority breakdown for backlog
                priority_breakdown = {}
                if not backlog_df.empty and 'Priority' in backlog_df.columns:
                    priority_counts = backlog_df['Priority'].value_counts().to_dict()
                    for priority, count in priority_counts.items():
                        if pd.notna(priority):  # Skip NaN priorities
                            priority_breakdown[str(priority)] = count
                
                # Get region breakdown for backlog
                region_breakdown = {}
                if not backlog_df.empty and 'Region' in backlog_df.columns:
                    region_counts = backlog_df['Region'].value_counts().to_dict()
                    for region, count in region_counts.items():
                        if pd.notna(region):  # Skip NaN regions
                            region_breakdown[str(region)] = count
                
                # Save data for this month
                backlog_data['labels'].append(month)
                backlog_data['backlog_count'].append(backlog_count)
                backlog_data['new_tickets'].append(new_count)
                backlog_data['solved_tickets'].append(solved_count)
                backlog_data['priority_breakdown'].append(priority_breakdown)
                backlog_data['region_breakdown'].append(region_breakdown)
            
            # Calculate current backlog age distribution
            current_open_tickets = df[(df['Ticket status'] == 'Open') | (df['Ticket status'] == 'Hold')]
            
            # Calculate age of tickets in days
            current_date = pd.Timestamp.today()
            current_open_tickets['age_days'] = (current_date - pd.to_datetime(current_open_tickets['Logged - Date'])).dt.days
            
            # Check for negative ages (future dates) and reset them to 0
            current_open_tickets.loc[current_open_tickets['age_days'] < 0, 'age_days'] = 0
            
            # Age buckets
            age_buckets = {
                '< 30 days': 0,
                '30-60 days': 0,
                '60-90 days': 0,
                '> 90 days': 0
            }
            
            for _, ticket in current_open_tickets.iterrows():
                if pd.isna(ticket['age_days']):
                    continue
                    
                age = ticket['age_days']
                if age < 30:
                    age_buckets['< 30 days'] += 1
                elif age < 60:
                    age_buckets['30-60 days'] += 1
                elif age < 90:
                    age_buckets['60-90 days'] += 1
                else:
                    age_buckets['> 90 days'] += 1
            
            backlog_age_data = {
                'labels': list(age_buckets.keys()),
                'counts': list(age_buckets.values())
            }
            
            # Calculate oldest ticket age safely
            oldest_ticket_age = 0
            if not current_open_tickets.empty and not current_open_tickets['age_days'].isnull().all():
                oldest_ticket_age = int(current_open_tickets['age_days'].max())
            
        except Exception as e:
            print(f"Error in backlog calculation: {str(e)}")
            # Provide default empty data if backlog calculation fails
            backlog_data = {
                'labels': ['No Data'],
                'backlog_count': [0],
                'new_tickets': [0],
                'solved_tickets': [0],
                'priority_breakdown': [{}],
                'region_breakdown': [{}]
            }
            backlog_age_data = {
                'labels': ['< 30 days', '30-60 days', '60-90 days', '> 90 days'],
                'counts': [0, 0, 0, 0]
            }
            oldest_ticket_age = 0
        
        # Open vs Solved by Region
        region_status = df.groupby(['Region', 'Ticket status']).size().unstack(fill_value=0).reset_index()
        # Ensure all statuses are present
        if 'Solved' not in region_status.columns:
            region_status['Solved'] = 0
        if 'Open' not in region_status.columns:
            region_status['Open'] = 0
        if 'Hold' not in region_status.columns:
            region_status['Hold'] = 0
            
        region_data = {
            'labels': region_status['Region'].tolist(),
            'solved': region_status['Solved'].tolist(),
            'open': region_status['Open'].tolist(),
            'hold': region_status['Hold'].tolist()
        }
        
        # Open vs Solved by Engineer (top 10 by total tickets)
        engineer_counts = df.groupby('Assignee name').size().sort_values(ascending=False)
        top_engineers = engineer_counts.head(10).index.tolist()
        
        engineer_status = df[df['Assignee name'].isin(top_engineers)].groupby(['Assignee name', 'Ticket status']).size().unstack(fill_value=0).reset_index()
        # Ensure all statuses are present
        if 'Solved' not in engineer_status.columns:
            engineer_status['Solved'] = 0
        if 'Open' not in engineer_status.columns:
            engineer_status['Open'] = 0
        if 'Hold' not in engineer_status.columns:
            engineer_status['Hold'] = 0
            
        engineer_data = {
            'labels': engineer_status['Assignee name'].tolist(),
            'solved': engineer_status['Solved'].tolist(),
            'open': engineer_status['Open'].tolist(),
            'hold': engineer_status['Hold'].tolist()
        }
        
        # Open vs Solved by Priority
        priority_status = df.groupby(['Priority', 'Ticket status']).size().unstack(fill_value=0).reset_index()
        # Ensure all statuses are present
        if 'Solved' not in priority_status.columns:
            priority_status['Solved'] = 0
        if 'Open' not in priority_status.columns:
            priority_status['Open'] = 0
        if 'Hold' not in priority_status.columns:
            priority_status['Hold'] = 0
            
        priority_data = {
            'labels': priority_status['Priority'].tolist(),
            'solved': priority_status['Solved'].tolist(),
            'open': priority_status['Open'].tolist(),
            'hold': priority_status['Hold'].tolist()
        }
        
        # Open vs Solved by Region and Group
        # Get top 5 most common groups for readability
        top_groups = df['Level 3 Group'].value_counts().head(5).index.tolist()
        
        # Handle edge case of no data
        if not top_groups:
            region_group_data = {
                'labels': ['No Data'],
                'datasets': [{
                    'label': 'No Data',
                    'data': [0],
                    'backgroundColor': 'rgba(200, 200, 200, 0.7)'
                }]
            }
        else:
            # Filter data to include only top groups
            filtered_df = df[df['Level 3 Group'].isin(top_groups)]
            
            # Create region-group combinations for x-axis labels
            region_group_combinations = filtered_df.groupby(['Region', 'Level 3 Group']).size().reset_index()
            region_group_labels = [f"{row['Region']} - {row['Level 3 Group']}" for _, row in region_group_combinations.iterrows()]
            
            # Create datasets for each status
            region_group_datasets = []
            for status in ['Solved', 'Open', 'Hold']:
                status_data = []
                for _, row in region_group_combinations.iterrows():
                    count = len(filtered_df[(filtered_df['Region'] == row['Region']) & 
                                           (filtered_df['Level 3 Group'] == row['Level 3 Group']) & 
                                           (filtered_df['Ticket status'] == status)])
                    status_data.append(count)
                
                # Different color for each status
                if status == 'Solved':
                    color = 'rgba(25, 135, 84, 0.7)'
                elif status == 'Open':
                    color = 'rgba(255, 193, 7, 0.7)'
                else:  # Hold
                    color = 'rgba(220, 53, 69, 0.7)'
                    
                region_group_datasets.append({
                    'label': status,
                    'data': status_data,
                    'backgroundColor': color
                })
            
            region_group_data = {
                'labels': region_group_labels,
                'datasets': region_group_datasets
            }
        
        # Generate preview table
        preview_html = df.head(10).to_html(classes='table table-striped table-hover', index=False)
        
        # Store the dataframe in a session variable for other routes
        session['ticket_data_processed'] = True
        
        # Calculate current backlog count
        current_backlog = open_count + hold_count
        
        metrics = {
            'all_tickets_avg': round(all_tickets_mean, 1) if not pd.isna(all_tickets_mean) else "N/A",
            'solved_tickets_avg': round(solved_tickets_mean, 1) if not pd.isna(solved_tickets_mean) else "N/A",
            'by_region': region_metrics.round(1).to_dict(),
            'by_group': group_metrics.round(1).to_dict(),
            'by_engineer': engineer_metrics.round(1).to_dict(),
            'by_priority': priority_metrics.round(1).to_dict(),
            'total_tickets': total_tickets,
            'solved_tickets': solved_count,
            'open_tickets': open_count,
            'hold_tickets': hold_count,
            'solved_percentage': solved_percentage,
            'open_percentage': open_percentage,
            'hold_percentage': hold_percentage,
            'within_sla_percentage': within_sla_percentage,
            'current_backlog': current_backlog,
            'oldest_ticket_age': oldest_ticket_age if 'oldest_ticket_age' in locals() else 0
        }
        
        return render_template(
            'ticket_overview.html',
            filename=filename,
            metrics=metrics,
            preview_html=preview_html,
            monthly_data=monthly_data,
            region_data=region_data,
            engineer_data=engineer_data,
            priority_data=priority_data,
            region_group_data=region_group_data,
            backlog_data=backlog_data,
            backlog_age_data=backlog_age_data
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error analyzing ticket data: {str(e)}')
        return redirect(url_for('index'))

@app.route('/region_analysis')
def region_analysis():
    filepath = session.get('filepath')
    filename = session.get('filename')
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    if not session.get('ticket_data_processed'):
        return redirect(url_for('ticket_overview'))
    
    try:
        # Load and process ticket data
        df = pd.read_excel(filepath)
        df = process_ticket_data(df)
        
        # Get sorted list of regions
        regions = sorted(df['Region'].unique())
        
        # Regional analysis - prepare comprehensive metrics
        solved_tickets = df[df['Ticket status'] == 'Solved']
        
        # Region metrics dictionary to store all metrics
        region_metrics = {}
        
        # Prepare summary data for rankings
        regions_by_speed = []
        regions_by_volume = []
        regions_by_slowest = []
        
        # Set thresholds for highlighting
        fast_regions = set()
        slow_regions = set()
        high_volume_regions = set()
        
        # Process each region
        for region in regions:
            # Get region data
            region_data = df[df['Region'] == region]
            region_solved = solved_tickets[solved_tickets['Region'] == region]
            
            # Basic counts
            total_tickets = len(region_data)
            solved_count = len(region_data[region_data['Ticket status'] == 'Solved'])
            open_count = len(region_data[region_data['Ticket status'] == 'Open'])
            hold_count = len(region_data[region_data['Ticket status'] == 'Hold'])
            
            # Average resolution time
            avg_resolution = region_solved['Resolution Time (Days)'].mean()
            avg_resolution_rounded = round(avg_resolution, 1) if not pd.isna(avg_resolution) else "N/A"
            
            # Add to sorting lists
            if not pd.isna(avg_resolution):
                regions_by_speed.append((region, avg_resolution_rounded))
                regions_by_slowest.append((region, avg_resolution_rounded))
                
                # Mark as fast or slow based on avg resolution time
                if avg_resolution < 3:  # Example threshold for "fast"
                    fast_regions.add(region)
                elif avg_resolution > 7:  # Example threshold for "slow"
                    slow_regions.add(region)
            
            regions_by_volume.append((region, total_tickets))
            
            # Mark as high volume
            if total_tickets > 100:  # Example threshold for "high volume"
                high_volume_regions.add(region)
            
            # Priority distribution
            priority_counts = region_data['Priority'].value_counts().to_dict()
            # Calculate percentages
            priority_percentages = {}
            for priority, count in priority_counts.items():
                priority_percentages[priority] = round((count / total_tickets) * 100)
            
            # Prepare chart data
            priority_chart_data = {
                'labels': list(priority_counts.keys()),
                'counts': list(priority_counts.values())
            }
            
            # Group distribution
            group_counts = region_data['Level 3 Group'].value_counts().to_dict()
            # Calculate percentages
            group_percentages = {}
            for group, count in group_counts.items():
                group_percentages[group] = round((count / total_tickets) * 100)
            
            # Engineer data
            engineer_data = {}
            for engineer in region_data['Assignee name'].unique():
                engineer_tickets = region_solved[region_solved['Assignee name'] == engineer]
                count = len(engineer_tickets)
                if count > 0:
                    avg_time = round(engineer_tickets['Resolution Time (Days)'].mean(), 1)
                    # Determine color based on resolution time
                    if avg_time < 3:
                        color = "success"
                        percentage = min(avg_time / 10 * 100, 100)  # Scale for visual representation
                    elif avg_time < 5:
                        color = "info"
                        percentage = min(avg_time / 10 * 100, 100)
                    elif avg_time < 7:
                        color = "warning"
                        percentage = min(avg_time / 10 * 100, 100)
                    else:
                        color = "danger"
                        percentage = 100
                    
                    engineer_data[engineer] = {
                        'count': count,
                        'avg_time': avg_time,
                        'color': color,
                        'percentage': percentage
                    }
            
            # Sort engineer data by count in descending order
            engineer_data = dict(sorted(engineer_data.items(), key=lambda x: x[1]['count'], reverse=True)[:10])
            
            # Store all metrics for this region
            region_metrics[region] = {
                'total_tickets': total_tickets,
                'solved_tickets': solved_count,
                'open_tickets': open_count,
                'hold_tickets': hold_count,
                'avg_resolution': avg_resolution_rounded,
                'priority_distribution': priority_counts,
                'priority_percentages': priority_percentages,
                'priority_chart_data': priority_chart_data,
                'group_distribution': group_counts,
                'group_percentages': group_percentages,
                'engineer_data': engineer_data
            }
        
        # Sort the lists for rankings
        regions_by_speed.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else float('inf'))
        regions_by_slowest.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else float('-inf'), reverse=True)
        regions_by_volume.sort(key=lambda x: x[1], reverse=True)
        
        # Average resolution time by region (for the overview table)
        region_avg = solved_tickets.groupby('Region')['Resolution Time (Days)'].mean().sort_values()
        region_avg_html = region_avg.round(1).reset_index().to_html(
            classes='table table-striped table-hover', 
            index=False,
            columns=['Region', 'Resolution Time (Days)'],
            header=['Region', 'Avg. Resolution Time (Days)']
        )
        
        # Ticket count by region (for reference - not displayed in new design)
        region_count = df.groupby(['Region', 'Ticket status']).size().unstack(fill_value=0)
        if 'Solved' not in region_count.columns:
            region_count['Solved'] = 0
        if 'Open' not in region_count.columns:
            region_count['Open'] = 0
        if 'Hold' not in region_count.columns:
            region_count['Hold'] = 0
            
        region_count['Total'] = region_count.sum(axis=1)
        region_count_html = region_count.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        # Region priority distribution (for reference - not displayed in new design)
        region_priority = df.groupby(['Region', 'Priority']).size().unstack(fill_value=0)
        region_priority_html = region_priority.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        return render_template(
            'region_analysis.html',
            filename=filename,
            region_avg_html=region_avg_html,
            region_count_html=region_count_html,
            region_priority_html=region_priority_html,
            regions=regions,
            region_metrics=region_metrics,
            regions_by_speed=regions_by_speed,
            regions_by_slowest=regions_by_slowest,
            regions_by_volume=regions_by_volume,
            fast_regions=fast_regions,
            slow_regions=slow_regions,
            high_volume_regions=high_volume_regions
        )
        
    except Exception as e:
        flash(f'Error in region analysis: {str(e)}')
        return redirect(url_for('ticket_overview'))

@app.route('/group_analysis')
def group_analysis():
    filepath = session.get('filepath')
    filename = session.get('filename')
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    if not session.get('ticket_data_processed'):
        return redirect(url_for('ticket_overview'))
    
    try:
        # Load and process ticket data
        df = pd.read_excel(filepath)
        df = process_ticket_data(df)
        
        # Group analysis
        solved_tickets = df[df['Ticket status'] == 'Solved']
        
        # Average resolution time by group
        group_avg = solved_tickets.groupby('Level 3 Group')['Resolution Time (Days)'].mean().sort_values()
        group_avg_html = group_avg.round(1).reset_index().to_html(
            classes='table table-striped table-hover', 
            index=False,
            columns=['Level 3 Group', 'Resolution Time (Days)'],
            header=['Group', 'Avg. Resolution Time (Days)']
        )
        
        # Ticket count by group
        group_count = df.groupby(['Level 3 Group', 'Ticket status']).size().unstack(fill_value=0)
        if 'Solved' not in group_count.columns:
            group_count['Solved'] = 0
        if 'Open' not in group_count.columns:
            group_count['Open'] = 0
        if 'Hold' not in group_count.columns:
            group_count['Hold'] = 0
            
        group_count['Total'] = group_count.sum(axis=1)
        group_count_html = group_count.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        # Group priority distribution
        group_priority = df.groupby(['Level 3 Group', 'Priority']).size().unstack(fill_value=0)
        group_priority_html = group_priority.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        return render_template(
            'group_analysis.html',
            filename=filename,
            group_avg_html=group_avg_html,
            group_count_html=group_count_html,
            group_priority_html=group_priority_html,
            groups=sorted(df['Level 3 Group'].unique())
        )
        
    except Exception as e:
        flash(f'Error in group analysis: {str(e)}')
        return redirect(url_for('ticket_overview'))

@app.route('/engineer_analysis')
def engineer_analysis():
    filepath = session.get('filepath')
    filename = session.get('filename')
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    if not session.get('ticket_data_processed'):
        return redirect(url_for('ticket_overview'))
    
    try:
        # Load and process ticket data
        df = pd.read_excel(filepath)
        df = process_ticket_data(df)
        
        # Engineer analysis
        solved_tickets = df[df['Ticket status'] == 'Solved']
        
        # Average resolution time by engineer
        engineer_avg = solved_tickets.groupby('Assignee name')['Resolution Time (Days)'].mean().sort_values()
        engineer_avg_html = engineer_avg.round(1).reset_index().to_html(
            classes='table table-striped table-hover', 
            index=False,
            columns=['Assignee name', 'Resolution Time (Days)'],
            header=['Engineer', 'Avg. Resolution Time (Days)']
        )
        
        # Ticket count by engineer
        engineer_count = df.groupby(['Assignee name', 'Ticket status']).size().unstack(fill_value=0)
        if 'Solved' not in engineer_count.columns:
            engineer_count['Solved'] = 0
        if 'Open' not in engineer_count.columns:
            engineer_count['Open'] = 0
        if 'Hold' not in engineer_count.columns:
            engineer_count['Hold'] = 0
            
        engineer_count['Total'] = engineer_count.sum(axis=1)
        engineer_count_html = engineer_count.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        # Engineer distribution by region
        engineer_region = df.groupby(['Assignee name', 'Region']).size().unstack(fill_value=0)
        engineer_region_html = engineer_region.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        return render_template(
            'engineer_analysis.html',
            filename=filename,
            engineer_avg_html=engineer_avg_html,
            engineer_count_html=engineer_count_html,
            engineer_region_html=engineer_region_html,
            engineers=sorted(df['Assignee name'].unique())
        )
        
    except Exception as e:
        flash(f'Error in engineer analysis: {str(e)}')
        return redirect(url_for('ticket_overview'))

@app.route('/priority_analysis')
def priority_analysis():
    filepath = session.get('filepath')
    filename = session.get('filename')
    
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    if not session.get('ticket_data_processed'):
        return redirect(url_for('ticket_overview'))
    
    try:
        # Load and process ticket data
        df = pd.read_excel(filepath)
        df = process_ticket_data(df)
        
        # Priority analysis
        solved_tickets = df[df['Ticket status'] == 'Solved']
        
        # Average resolution time by priority
        priority_avg = solved_tickets.groupby('Priority')['Resolution Time (Days)'].mean().sort_values()
        priority_avg_html = priority_avg.round(1).reset_index().to_html(
            classes='table table-striped table-hover', 
            index=False,
            columns=['Priority', 'Resolution Time (Days)'],
            header=['Priority', 'Avg. Resolution Time (Days)']
        )
        
        # Ticket count by priority
        priority_count = df.groupby(['Priority', 'Ticket status']).size().unstack(fill_value=0)
        if 'Solved' not in priority_count.columns:
            priority_count['Solved'] = 0
        if 'Open' not in priority_count.columns:
            priority_count['Open'] = 0
        if 'Hold' not in priority_count.columns:
            priority_count['Hold'] = 0
            
        priority_count['Total'] = priority_count.sum(axis=1)
        priority_count_html = priority_count.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        # Priority distribution by region
        priority_region = df.groupby(['Priority', 'Region']).size().unstack(fill_value=0)
        priority_region_html = priority_region.reset_index().to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        return render_template(
            'priority_analysis.html',
            filename=filename,
            priority_avg_html=priority_avg_html,
            priority_count_html=priority_count_html,
            priority_region_html=priority_region_html,
            priorities=sorted(df['Priority'].unique())
        )
        
    except Exception as e:
        flash(f'Error in priority analysis: {str(e)}')
        return redirect(url_for('ticket_overview'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 