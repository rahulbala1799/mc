import pandas as pd
import numpy as np
from datetime import datetime
import traceback

def calculate_backlog_age_distribution(df):
    """Calculate age distribution of backlog tickets."""
    try:
        # Get current backlog (tickets without solved date)
        current_backlog = df[df['Ticket solved - Date'].isna()]
        
        # Calculate age of tickets in days
        current_date = pd.Timestamp.today()
        current_backlog['age_days'] = (current_date - pd.to_datetime(current_backlog['Logged - Date'])).dt.days
        
        # Correct any negative ages (future dates) to 0
        current_backlog.loc[current_backlog['age_days'] < 0, 'age_days'] = 0
        
        # Age buckets
        age_buckets = {
            '< 7 days': 0,
            '7-14 days': 0,
            '14-30 days': 0,
            '30-60 days': 0,
            '> 60 days': 0
        }
        
        # Populate age buckets
        for _, ticket in current_backlog.iterrows():
            if pd.isna(ticket['age_days']):
                continue
                
            age = ticket['age_days']
            if age < 7:
                age_buckets['< 7 days'] += 1
            elif age < 14:
                age_buckets['7-14 days'] += 1
            elif age < 30:
                age_buckets['14-30 days'] += 1
            elif age < 60:
                age_buckets['30-60 days'] += 1
            else:
                age_buckets['> 60 days'] += 1
        
        # Prepare data for chart
        backlog_age_data = {
            'labels': list(age_buckets.keys()),
            'counts': list(age_buckets.values())
        }
        
        # Calculate average age
        avg_age = current_backlog['age_days'].mean()
        if pd.isna(avg_age):
            avg_age = 0
            
        # Find oldest ticket age
        max_age = current_backlog['age_days'].max()
        if pd.isna(max_age):
            max_age = 0
            
        return backlog_age_data, round(avg_age, 1), round(max_age, 1)
    
    except Exception as e:
        print(f"Error calculating backlog age distribution: {str(e)}")
        traceback.print_exc()
        return {
            'labels': ['< 7 days', '7-14 days', '14-30 days', '30-60 days', '> 60 days'],
            'counts': [0, 0, 0, 0, 0]
        }, 0, 0

def calculate_backlog_composition(df):
    """Calculate backlog composition by priority, region, and group."""
    try:
        # Get current backlog
        current_backlog = df[df['Ticket solved - Date'].isna()]
        total_backlog = len(current_backlog)
        
        # Empty result dictionaries
        priority_data = {'labels': [], 'data': [], 'percentages': {}}
        region_data = {'labels': [], 'data': [], 'percentages': {}}
        group_data = {'labels': [], 'data': [], 'percentages': {}}
        
        # Priority distribution
        if not current_backlog.empty and 'Priority' in current_backlog.columns:
            priority_counts = current_backlog['Priority'].value_counts()
            priority_data['labels'] = priority_counts.index.tolist()
            priority_data['data'] = priority_counts.values.tolist()
            
            # Calculate percentages
            for priority, count in zip(priority_data['labels'], priority_data['data']):
                if total_backlog > 0:
                    priority_data['percentages'][priority] = round((count / total_backlog) * 100)
                else:
                    priority_data['percentages'][priority] = 0
        
        # Region distribution
        if not current_backlog.empty and 'Region' in current_backlog.columns:
            region_counts = current_backlog['Region'].value_counts()
            region_data['labels'] = region_counts.index.tolist()
            region_data['data'] = region_counts.values.tolist()
            
            # Calculate percentages
            for region, count in zip(region_data['labels'], region_data['data']):
                if total_backlog > 0:
                    region_data['percentages'][region] = round((count / total_backlog) * 100)
                else:
                    region_data['percentages'][region] = 0
        
        # Group distribution
        if not current_backlog.empty and 'Level 3 Group' in current_backlog.columns:
            group_counts = current_backlog['Level 3 Group'].value_counts()
            group_data['labels'] = group_counts.index.tolist()[:7]  # Limit to top 7 groups
            group_data['data'] = group_counts.values.tolist()[:7]
            
            # Calculate percentages
            for group, count in zip(group_data['labels'], group_data['data']):
                if total_backlog > 0:
                    group_data['percentages'][group] = round((count / total_backlog) * 100)
                else:
                    group_data['percentages'][group] = 0
        
        return priority_data, region_data, group_data, total_backlog
    
    except Exception as e:
        print(f"Error calculating backlog composition: {str(e)}")
        traceback.print_exc()
        return {
            'labels': [], 'data': [], 'percentages': {}
        }, {
            'labels': [], 'data': [], 'percentages': {}
        }, {
            'labels': [], 'data': [], 'percentages': {}
        }, 0

def calculate_backlog_trend(df):
    """Calculate the monthly backlog trend."""
    try:
        # Add month and year columns if they don't exist
        if 'Month' not in df.columns:
            df['Month'] = pd.to_datetime(df['Logged - Date']).dt.strftime('%Y-%m')
        
        # Sort DataFrame by month
        df_sorted = df.sort_values('Logged - Date')
        
        # Get unique months in sorted order
        unique_months = sorted(df_sorted['Month'].dropna().unique().tolist())
        
        if not unique_months:
            raise ValueError("No valid months found for backlog analysis")
        
        # Initialize backlog tracking
        backlog_data = {
            'labels': [],
            'backlog_count': [],
            'new_tickets': [],
            'solved_tickets': []
        }
        
        # Convert month strings to datetime for proper comparison
        month_dates = [pd.to_datetime(month + "-01") for month in unique_months]
        
        # Calculate backlog for each month
        for i, month in enumerate(unique_months):
            month_date = month_dates[i]
            month_end = month_date + pd.offsets.MonthEnd(1)
            
            # Get all tickets created up to this month (including this month)
            tickets_up_to_now = df[pd.to_datetime(df['Logged - Date']) <= month_end]
            
            # New tickets this month
            new_tickets_this_month = df[df['Month'] == month]
            new_count = len(new_tickets_this_month)
            
            # Solved tickets this month
            solved_month_df = df[(df['Month'] == month) & (df['Ticket status'] == 'Solved')]
            solved_count = len(solved_month_df)
            
            # Backlog: tickets logged on or before this month, and either not solved, or solved after this month
            backlog_df = tickets_up_to_now[
                (tickets_up_to_now['Ticket solved - Date'].isna()) |
                (pd.to_datetime(tickets_up_to_now['Ticket solved - Date']) > month_end)
            ]
            
            # Special case for the last month
            if i == len(unique_months) - 1:
                current_backlog_count = len(df[df['Ticket solved - Date'].isna()])
                if current_backlog_count != len(backlog_df):
                    backlog_df = df[df['Ticket solved - Date'].isna()]
                    
            backlog_count = len(backlog_df)
            
            # Save data for this month
            backlog_data['labels'].append(month)
            backlog_data['backlog_count'].append(backlog_count)
            backlog_data['new_tickets'].append(new_count)
            backlog_data['solved_tickets'].append(solved_count)
        
        # Calculate monthly change and percentage change
        backlog_data['monthly_change'] = []
        backlog_data['monthly_change_pct'] = []
        
        for i in range(len(backlog_data['backlog_count'])):
            if i == 0:
                backlog_data['monthly_change'].append(0)
                backlog_data['monthly_change_pct'].append(0)
            else:
                change = backlog_data['backlog_count'][i] - backlog_data['backlog_count'][i-1]
                backlog_data['monthly_change'].append(change)
                
                if backlog_data['backlog_count'][i-1] > 0:
                    pct_change = round((change / backlog_data['backlog_count'][i-1]) * 100, 1)
                else:
                    pct_change = 0
                backlog_data['monthly_change_pct'].append(pct_change)
        
        return backlog_data
        
    except Exception as e:
        print(f"Error calculating backlog trend: {str(e)}")
        traceback.print_exc()
        return {
            'labels': ['Current Month'],
            'backlog_count': [0],
            'new_tickets': [0],
            'solved_tickets': [0],
            'monthly_change': [0],
            'monthly_change_pct': [0]
        }

def calculate_assignee_backlog(df):
    """Calculate backlog distribution by assignee."""
    try:
        # Get current backlog
        current_backlog = df[df['Ticket solved - Date'].isna()]
        
        # Get backlog by assignee
        assignee_counts = current_backlog['Assignee name'].value_counts()
        
        # Prepare data for chart, limiting to top 10 assignees
        assignee_data = {
            'labels': assignee_counts.index.tolist()[:10],
            'data': assignee_counts.values.tolist()[:10]
        }
        
        # Get priority distribution for each assignee
        assignee_priority_data = {}
        
        for assignee in assignee_data['labels']:
            assignee_tickets = current_backlog[current_backlog['Assignee name'] == assignee]
            if 'Priority' in assignee_tickets.columns:
                priority_counts = assignee_tickets['Priority'].value_counts().to_dict()
                assignee_priority_data[assignee] = priority_counts
        
        return assignee_data, assignee_priority_data
        
    except Exception as e:
        print(f"Error calculating assignee backlog: {str(e)}")
        traceback.print_exc()
        return {
            'labels': [],
            'data': []
        }, {}

def calculate_jira_backlog(df):
    """Calculate backlog statistics related to JIRA IDs."""
    try:
        # Ensure JIRA ID column exists
        jira_id_column = None
        possible_names = ['JIRA ID', 'JIRAID', 'JIRA_ID', 'JIRA-ID', 'JIRA']
        
        for col in df.columns:
            if col.upper().replace(' ', '') in [name.upper().replace(' ', '') for name in possible_names]:
                jira_id_column = col
                break
        
        if not jira_id_column:
            return {}, 0, 0, 0, 0
        
        # Get current backlog
        current_backlog = df[df['Ticket solved - Date'].isna()]
        
        # Clean up JIRA ID column
        current_backlog['JIRA ID'] = current_backlog[jira_id_column].astype(str).str.strip()
        has_jira = current_backlog['JIRA ID'].str.lower() != 'no jira id'
        no_jira = ~has_jira
        
        # Count tickets with and without JIRA IDs
        with_jira_count = has_jira.sum()
        no_jira_count = no_jira.sum()
        
        # Calculate percentages
        total = len(current_backlog)
        with_jira_pct = round((with_jira_count / total) * 100, 1) if total > 0 else 0
        no_jira_pct = round((no_jira_count / total) * 100, 1) if total > 0 else 0
        
        # Prepare data for chart
        jira_data = {
            'labels': ['With JIRA ID', 'No JIRA ID'],
            'data': [with_jira_count, no_jira_count]
        }
        
        return jira_data, with_jira_count, no_jira_count, with_jira_pct, no_jira_pct
        
    except Exception as e:
        print(f"Error calculating JIRA backlog: {str(e)}")
        traceback.print_exc()
        return {
            'labels': ['With JIRA ID', 'No JIRA ID'],
            'data': [0, 0]
        }, 0, 0, 0, 0

def calculate_risk_assessment(df):
    """Identify high-risk tickets in the backlog."""
    try:
        # Get current backlog
        current_backlog = df[df['Ticket solved - Date'].isna()]
        
        # Calculate age of tickets in days
        current_date = pd.Timestamp.today()
        current_backlog['age_days'] = (current_date - pd.to_datetime(current_backlog['Logged - Date'])).dt.days
        
        # Correct any negative ages (future dates) to 0
        current_backlog.loc[current_backlog['age_days'] < 0, 'age_days'] = 0
        
        # Identify high priority tickets
        high_priority_terms = ['high', 'critical', 'urgent', 'p1', '1']
        
        # Create a mask for high priority tickets
        high_priority_mask = current_backlog['Priority'].astype(str).str.lower().apply(
            lambda x: any(term in x for term in high_priority_terms)
        )
        
        # Get high priority backlog
        high_priority_backlog = current_backlog[high_priority_mask]
        
        # Find aging high priority tickets (e.g., > 7 days old)
        aging_high_priority = high_priority_backlog[high_priority_backlog['age_days'] > 7]
        
        # Count high-risk tickets
        high_priority_count = len(high_priority_backlog)
        aging_high_priority_count = len(aging_high_priority)
        
        # Calculate risk percentage
        risk_percentage = round((high_priority_count / len(current_backlog)) * 100, 1) if len(current_backlog) > 0 else 0
        aging_percentage = round((aging_high_priority_count / high_priority_count) * 100, 1) if high_priority_count > 0 else 0
        
        # Get top 10 oldest high priority tickets for detailed table
        oldest_high_priority = high_priority_backlog.sort_values('age_days', ascending=False).head(10)
        high_risk_tickets = []
        
        for _, ticket in oldest_high_priority.iterrows():
            high_risk_tickets.append({
                'ticket_id': ticket.get('Ticket ID', 'N/A'),
                'priority': ticket.get('Priority', 'N/A'),
                'age_days': int(ticket.get('age_days', 0)),
                'region': ticket.get('Region', 'N/A'),
                'assignee': ticket.get('Assignee name', 'N/A'),
                'group': ticket.get('Level 3 Group', 'N/A')
            })
        
        return high_priority_count, aging_high_priority_count, risk_percentage, aging_percentage, high_risk_tickets
        
    except Exception as e:
        print(f"Error calculating risk assessment: {str(e)}")
        traceback.print_exc()
        return 0, 0, 0, 0, []

def calculate_resolution_projection(df):
    """Project when the current backlog would be cleared at current rates."""
    try:
        # Get current backlog
        current_backlog = df[df['Ticket solved - Date'].isna()]
        current_backlog_count = len(current_backlog)
        
        # Get average resolution rate over the last 3 months
        if 'Month' not in df.columns:
            df['Month'] = pd.to_datetime(df['Logged - Date']).dt.strftime('%Y-%m')
        
        # Sort months
        months = sorted(df['Month'].dropna().unique().tolist())
        
        # Not enough data for projection
        if len(months) < 3:
            return current_backlog_count, 0, 0, 0
        
        # Get last 3 months
        last_three_months = months[-3:]
        
        # Calculate monthly resolution rates
        monthly_resolved = []
        
        for month in last_three_months:
            resolved_count = len(df[(df['Month'] == month) & (df['Ticket status'] == 'Solved')])
            monthly_resolved.append(resolved_count)
        
        # Calculate average monthly resolution rate
        avg_monthly_resolution = sum(monthly_resolved) / len(monthly_resolved)
        
        # Calculate projection
        if avg_monthly_resolution > 0:
            months_to_clear = current_backlog_count / avg_monthly_resolution
            weeks_to_clear = months_to_clear * 4.33  # Approximate weeks in a month
        else:
            months_to_clear = 0
            weeks_to_clear = 0
        
        return current_backlog_count, round(avg_monthly_resolution, 1), round(months_to_clear, 1), round(weeks_to_clear, 1)
        
    except Exception as e:
        print(f"Error calculating resolution projection: {str(e)}")
        traceback.print_exc()
        return 0, 0, 0, 0

def prepare_backlog_analysis(df):
    """Prepare all data needed for the backlog analysis page."""
    # Ensure data is processed
    if 'Resolution Time (Days)' not in df.columns:
        # Calculate resolution time in days
        df['Resolution Time (Days)'] = (pd.to_datetime(df['Ticket solved - Date']) - 
                                        pd.to_datetime(df['Logged - Date'])).dt.total_seconds() / (24 * 60 * 60)
    
    # Run all analyses
    backlog_age_data, avg_age, max_age = calculate_backlog_age_distribution(df)
    priority_data, region_data, group_data, total_backlog = calculate_backlog_composition(df)
    backlog_trend_data = calculate_backlog_trend(df)
    assignee_data, assignee_priority_data = calculate_assignee_backlog(df)
    jira_data, with_jira_count, no_jira_count, with_jira_pct, no_jira_pct = calculate_jira_backlog(df)
    high_priority_count, aging_high_priority_count, risk_percentage, aging_percentage, high_risk_tickets = calculate_risk_assessment(df)
    current_backlog_count, avg_monthly_resolution, months_to_clear, weeks_to_clear = calculate_resolution_projection(df)
    
    # Compile all data into a single dictionary
    backlog_analysis = {
        'backlog_age_data': backlog_age_data,
        'avg_age': avg_age,
        'max_age': max_age,
        'priority_data': priority_data,
        'region_data': region_data,
        'group_data': group_data,
        'total_backlog': total_backlog,
        'backlog_trend_data': backlog_trend_data,
        'assignee_data': assignee_data,
        'assignee_priority_data': assignee_priority_data,
        'jira_data': jira_data,
        'with_jira_count': with_jira_count,
        'no_jira_count': no_jira_count,
        'with_jira_pct': with_jira_pct,
        'no_jira_pct': no_jira_pct,
        'high_priority_count': high_priority_count,
        'aging_high_priority_count': aging_high_priority_count,
        'risk_percentage': risk_percentage,
        'aging_percentage': aging_percentage,
        'high_risk_tickets': high_risk_tickets,
        'current_backlog_count': current_backlog_count,
        'avg_monthly_resolution': avg_monthly_resolution,
        'months_to_clear': months_to_clear,
        'weeks_to_clear': weeks_to_clear
    }
    
    return backlog_analysis 