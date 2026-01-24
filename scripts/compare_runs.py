#!/usr/bin/env python3
"""
Compare Two WandB Runs and Generate HTML Report

This script fetches and visualizes data from two experiment runs for easy comparison
and generates an HTML report with all comparison components.

Usage:
    python compare_runs.py <run_id_1> <run_id_2> [--output report.html] [--entity ENTITY] [--project PROJECT]
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from collections import defaultdict
import base64
from io import BytesIO
import re

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import wandb
import wandb
from wandb.apis.public import Run

# Import image libraries
import requests
from PIL import Image
from urllib.parse import urlparse


class RunComparator:
    """Class to compare two WandB runs and generate HTML report."""
    
    def __init__(self, run_ids: List[str], entity: str = None, project: str = None):
        self.run_ids = run_ids
        self.entity = entity or 'mllab-ts-universit-di-trieste'
        self.project = project or 'CounterFactualDPG'
        self.api = wandb.Api()
        self.runs_data = {}
        
        # Set style for plots
        sns.set_style('whitegrid')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
    def fetch_run_data(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive information from a WandB run."""
        run_path = f"{self.entity}/{self.project}/{run_id}"
        print(f"Fetching run: {run_path}")
        
        try:
            run = self.api.run(run_path)
            
            # Collect all run information
            run_data = {
                'meta': {
                    'id': run.id,
                    'name': run.name,
                    'display_name': run.display_name,
                    'state': run.state,
                    'url': run.url,
                    'path': run.path,
                    'entity': run.entity,
                    'project': run.project,
                    'created_at': run.created_at,
                    'updated_at': getattr(run, 'updated_at', None),
                    'notes': run.notes,
                    'tags': list(run.tags) if run.tags else [],
                    'group': run.group,
                    'job_type': run.job_type,
                },
                'config': dict(run.config),
                'summary': {},
                'history': [],
                'history_keys': [],
                'system_metrics': {},
                'files': [],
                'artifacts': [],
            }
            
            # Get summary metrics
            for key, value in run.summary.items():
                if not key.startswith('_'):
                    try:
                        run_data['summary'][key] = float(value)
                    except (ValueError, TypeError):
                        run_data['summary'][key] = value
            
            # Get history (time-series data)
            try:
                history = run.history(pandas=False)
                if history:
                    run_data['history'] = list(history)
                    # Extract unique keys from history
                    all_keys = set()
                    for row in history:
                        all_keys.update(row.keys())
                    run_data['history_keys'] = sorted(list(all_keys))
            except Exception as e:
                print(f"  Warning: Could not fetch history: {e}")
            
            # Get files
            try:
                files = run.files()
                run_data['files'] = [
                    {
                        'name': f.name,
                        'size': f.size,
                        'mimetype': getattr(f, 'mimetype', None),
                        'url': f.url,
                    }
                    for f in files
                ]
            except Exception as e:
                print(f"  Warning: Could not fetch files: {e}")
            
            # Get artifacts
            try:
                artifacts = run.logged_artifacts()
                run_data['artifacts'] = [
                    {
                        'name': a.name,
                        'type': a.type,
                        'version': a.version,
                        'size': a.size,
                    }
                    for a in artifacts
                ]
            except Exception as e:
                print(f"  Warning: Could not fetch artifacts: {e}")
            
            print(f"  Successfully fetched {len(run_data['history'])} history steps")
            return run_data
            
        except Exception as e:
            print(f"  Error fetching run: {e}")
            return None
    
    def load_all_runs(self) -> bool:
        """Load data for all runs."""
        print(f"\nLoading data for {len(self.run_ids)} runs...")
        for run_id in self.run_ids:
            self.runs_data[run_id] = self.fetch_run_data(run_id)
        
        success_count = len([r for r in self.runs_data.values() if r])
        print(f"\nLoaded data for {success_count}/{len(self.run_ids)} runs")
        return success_count == len(self.run_ids)
    
    def format_value(self, value, max_length=50):
        """Format values for display."""
        if value is None:
            return "N/A"
        if isinstance(value, (list, dict)):
            return f"{type(value).__name__}({len(value)})"
        str_val = str(value)
        if len(str_val) > max_length:
            return str_val[:max_length-3] + "..."
        return str_val
    
    def get_comparison_tables_html(self) -> str:
        """Generate HTML for comparison tables."""
        html = "<div class='section'>\n"
        html += "<h2>📊 Comparison Tables</h2>\n"
        
        # Metadata comparison
        meta_data = [self.runs_data[run_id]['meta'] for run_id in self.run_ids]
        df_meta = pd.DataFrame(meta_data, index=[f"Run {i+1} ({run_id})" for i, run_id in enumerate(self.run_ids)]).T
        
        # Summary metrics comparison
        df_summary = pd.DataFrame({
            f"Run {i+1} ({run_id})": self.runs_data[run_id]['summary']
            for i, run_id in enumerate(self.run_ids)
        })
        
        # Configuration comparison
        df_config = pd.DataFrame({
            f"Run {i+1} ({run_id})": self.runs_data[run_id]['config']
            for i, run_id in enumerate(self.run_ids)
        })
        
        html += "<div class='comparison-container'>\n"
        html += "<div class='comparison-card'>\n<h3>📋 Run Metadata</h3>\n"
        html += df_meta.to_html(classes='data-table', index=True)
        html += "</div>\n"
        
        html += "<div class='comparison-card'>\n<h3>📈 Summary Metrics</h3>\n"
        html += df_summary.head(30).to_html(classes='data-table', index=True)
        html += "</div>\n"
        
        html += "<div class='comparison-card'>\n<h3>⚙️ Configuration</h3>\n"
        html += df_config.head(30).to_html(classes='data-table', index=True)
        html += "</div>\n"
        html += "</div>\n</div>\n"
        
        return html
    
    def get_training_history_plot(self) -> Optional[str]:
        """Generate training history plot as HTML."""
        # Find common history keys
        all_history_keys = [set(self.runs_data[run_id]['history_keys']) for run_id in self.run_ids]
        common_metrics = sorted(set.intersection(*all_history_keys))
        
        if not common_metrics:
            return None
        
        # Filter to numeric metrics
        numeric_metrics = []
        for key in common_metrics:
            is_numeric = True
            for run_id in self.run_ids:
                for row in self.runs_data[run_id]['history']:
                    if key in row and row[key] is not None:
                        if not isinstance(row[key], (int, float)):
                            is_numeric = False
                            break
            if is_numeric and any(key in row for run_id in self.run_ids for row in self.runs_data[run_id]['history']):
                numeric_metrics.append(key)
        
        if not numeric_metrics:
            return None
        
        # Create subplots for each metric
        n_metrics = min(len(numeric_metrics), 8)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_metrics[:n_metrics]
        )
        
        colors = ['#2e86de', '#10ac84', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c', '#34495e'][:len(self.run_ids)]
        
        for idx, metric in enumerate(numeric_metrics[:n_metrics]):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            for i, run_id in enumerate(self.run_ids):
                history = self.runs_data[run_id]['history']
                steps = []
                values = []
                for row_data in history:
                    if metric in row_data and row_data[metric] is not None:
                        steps.append(row_data.get('_step', len(values)))
                        values.append(row_data[metric])
                
                if steps and values:
                    fig.add_trace(
                        go.Scatter(
                            x=steps,
                            y=values,
                            mode='lines+markers',
                            name=f"Run {i+1} ({run_id})",
                            legendgroup=f"group_{i}" if idx == 0 else None,
                            showlegend=(idx == 0),
                            line=dict(color=colors[i], width=2),
                            marker=dict(size=4)
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="<b>Training History: Side-by-Side Comparison</b>",
            hovermode='x unified',
            template='plotly'
        )
        
        html = "<div class='section'>\n<h2>📉 Training History Comparison</h2>\n"
        html += fig.to_html(include_plotlyjs='cdn', div_id='training_history')
        html += "</div>\n"
        return html
    
    def get_summary_metrics_plot(self) -> Optional[str]:
        """Generate summary metrics bar chart as HTML."""
        # Extract numeric summary metrics for comparison
        metric_values = []
        metric_names = []
        
        # Get all common summary keys
        all_summary_keys = [set(self.runs_data[run_id]['summary'].keys()) for run_id in self.run_ids]
        common_summary_metrics = set.intersection(*all_summary_keys)
        
        for key in common_summary_metrics:
            if not key.startswith('_') and isinstance(self.runs_data[self.run_ids[0]]['summary'].get(key), (int, float)):
                try:
                    values = []
                    for run_id in self.run_ids:
                        values.append(float(self.runs_data[run_id]['summary'].get(key, 0)))
                    
                    # Filter out extreme values for better visualization
                    if all(abs(v) < 1e6 for v in values):
                        metric_names.append(key)
                        metric_values.append(values)
                except (ValueError, TypeError):
                    pass
        
        if not metric_values:
            return None
        
        column_names = [f"Run {i+1} ({run_id})" for i, run_id in enumerate(self.run_ids)]
        df_metrics = pd.DataFrame(metric_values, columns=column_names, index=metric_names)
        
        # Bar chart comparison
        fig = go.Figure()
        
        colors = ['#2e86de', '#10ac84', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c', '#34495e']
        
        for i, run_id in enumerate(self.run_ids):
            col_name = f"Run {i+1} ({run_id})"
            fig.add_trace(go.Bar(
                name=col_name,
                x=metric_names,
                y=df_metrics[col_name],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='<b>Summary Metrics Comparison</b>',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            height=600,
            hovermode='x unified',
            xaxis={'tickangle': -45},
            template='plotly_white'
        )
        
        html = "<div class='section'>\n<h2>📊 Summary Metrics Comparison</h2>\n"
        html += fig.to_html(include_plotlyjs='cdn', div_id='summary_metrics')
        html += "</div>\n"
        return html
    
    def get_difference_analysis_html(self) -> Optional[str]:
        """Generate difference analysis HTML."""
        # Calculate percentage differences (comparing first run to others)
        differences = []
        all_summary_keys = [set(self.runs_data[run_id]['summary'].keys()) for run_id in self.run_ids]
        common_keys = set.intersection(*all_summary_keys)
        
        for key in common_keys:
            if not key.startswith('_') and isinstance(self.runs_data[self.run_ids[0]]['summary'].get(key), (int, float)):
                try:
                    baseline_val = float(self.runs_data[self.run_ids[0]]['summary'][key])
                    
                    # Compare with other runs
                    for i in range(1, len(self.run_ids)):
                        run_id = self.run_ids[i]
                        if (isinstance(self.runs_data[run_id]['summary'].get(key), (int, float))):
                            compare_val = float(self.runs_data[run_id]['summary'][key])
                            
                            if abs(baseline_val) > 1e-10:  # Avoid division by zero
                                pct_diff = ((compare_val - baseline_val) / abs(baseline_val)) * 100
                                absolute_diff = compare_val - baseline_val
                                
                                # Filter for meaningful differences
                                if abs(baseline_val) < 1e6 and abs(compare_val) < 1e6:
                                    differences.append({
                                        'Metric': key,
                                        'Baseline (Run 1)': baseline_val,
                                        'Comparison (Run ' + str(i+1) + ')': compare_val,
                                        'Absolute Diff': absolute_diff,
                                        'Percent Diff (%)': pct_diff,
                                        'Better Run': f'Run {i+1}' if pct_diff > 0 else 'Run 1',
                                        'Compared with': run_id
                                    })
                except (ValueError, TypeError):
                    pass
        
        if not differences:
            return None
        
        df_diff = pd.DataFrame(differences).sort_values('Absolute Diff', key=abs, ascending=False)
        df_diff_top = df_diff.head(15)
        
        # Visualization of differences
        fig = go.Figure()
        
        # Color based on whether Run 2 improved
        colors = ['#10ac84' if diff > 0 else '#ee5253' for diff in df_diff_top['Percent Diff (%)']]
        
        fig.add_trace(go.Bar(
            x=df_diff_top['Metric'],
            y=df_diff_top['Percent Diff (%)'],
            marker_color=colors,
            hovertemplate='%{x}<br>Percent Change: %{y:.2f}%<extra></extra>',
        ))
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='<b>Percentage Difference: Run 2 vs Run 1</b>',
            xaxis_title='Metric',
            yaxis_title='Percentage Difference (%)',
            height=600,
            xaxis={'tickangle': -45},
            template='plotly_white'
        )
        
        html = "<div class='section'>\n"
        html += "<h2>📊 Metric Difference Analysis</h2>\n"
        html += "<div class='comparison-container'>\n"
        html += "<div class='comparison-card'>\n<h3>Top Metric Differences</h3>\n"
        html += df_diff_top.to_html(classes='data-table', index=False)
        html += "</div>\n"
        html += "<div style='flex: 1;'>\n"
        html += fig.to_html(include_plotlyjs='cdn', div_id='difference_plot')
        html += "</div>\n"
        html += "</div>\n</div>\n"
        
        return html
    
    def get_files_artifacts_html(self) -> str:
        """Generate HTML for files and artifacts comparison."""
        html = "<div class='section'>\n<h2>📁 Files and Artifacts Comparison</h2>\n"
        
        # Files comparison
        html += "<h3>Files Comparison</h3>\n"
        files_data = {'File Name': []}
        for i, run_id in enumerate(self.run_ids):
            files_data[f'Run {i+1} Size (KB)'] = []
        files_data['Available In'] = []
        
        all_files = set()
        for run_id in self.run_ids:
            for f in self.runs_data[run_id]['files']:
                all_files.add(f['name'])
        
        for filename in sorted(all_files):
            available_in = []
            for i, run_id in enumerate(self.run_ids):
                if filename in [f['name'] for f in self.runs_data[run_id]['files']]:
                    available_in.append(f'Run {i+1}')
                    f = next((f for f in self.runs_data[run_id]['files'] if f['name'] == filename), None)
                    if f:
                        files_data[f'Run {i+1} Size (KB)'].append(f"{f['size'] / 1024:.2f}" if f['size'] else "Unknown")
                else:
                    files_data[f'Run {i+1} Size (KB)'].append("N/A")
            
            files_data['File Name'].append(filename)
            files_data['Available In'].append(', '.join(available_in))
        
        df_files = pd.DataFrame(files_data)
        html += df_files.to_html(classes='data-table')
        
        # Artifacts comparison
        html += "<h3>Artifacts Comparison</h3>\n"
        artifacts_data = {'Artifact': [], 'Type': []}
        for i, run_id in enumerate(self.run_ids):
            artifacts_data[f'Run {i+1} Version'] = []
        artifacts_data['Available In'] = []
        
        all_artifacts = set()
        for run_id in self.run_ids:
            for a in self.runs_data[run_id]['artifacts']:
                all_artifacts.add(a['name'])
        
        for artifact_name in sorted(all_artifacts):
            available_in = []
            artifact_type = "N/A"
            
            for i, run_id in enumerate(self.run_ids):
                if artifact_name in [a['name'] for a in self.runs_data[run_id]['artifacts']]:
                    available_in.append(f'Run {i+1}')
                    a = next((a for a in self.runs_data[run_id]['artifacts'] if a['name'] == artifact_name), None)
                    if a:
                        artifacts_data[f'Run {i+1} Version'].append(a['version'])
                        artifact_type = a['type']
                else:
                    artifacts_data[f'Run {i+1} Version'].append("N/A")
            
            artifacts_data['Artifact'].append(artifact_name)
            artifacts_data['Type'].append(artifact_type)
            artifacts_data['Available In'].append(', '.join(available_in))
        
        df_artifacts = pd.DataFrame(artifacts_data)
        html += df_artifacts.to_html(classes='data-table')
        
        html += "</div>\n"
        return html
    
    def get_images_comparison_html(self) -> str:
        """Generate HTML for images comparison."""
        
        def download_image(url):
            """Download image from URL and return PIL Image."""
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"  Error downloading image: {e}")
            return None
        
        def get_image_files(run_data):
            """Get all image files from run data."""
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
            image_files = []
            
            for f in run_data['files']:
                name_lower = f['name'].lower()
                if any(name_lower.endswith(ext) for ext in image_extensions):
                    image_files.append(f)
            
            return image_files
        
        def extract_image_base_name(filename):
            """Extract base name from image filename, removing hash suffix."""
            match = re.match(r'^(.+)_[a-f0-9]{16,}(\.[^.]+)$', filename)
            if match:
                return match.group(1) + match.group(2)
            return filename
        
        html = "<div class='section'>\n<h2>🖼️ Images Comparison</h2>\n"
        
        # Get image files from all runs
        all_run_images = {}
        for run_id in self.run_ids:
            all_run_images[run_id] = get_image_files(self.runs_data[run_id])
        
        # Group images by their base name (without hash)
        image_groups = defaultdict(lambda: {run_id: None for run_id in self.run_ids})
        
        for run_id in self.run_ids:
            for img in all_run_images[run_id]:
                base_name = extract_image_base_name(img['name'])
                image_groups[base_name][run_id] = img
        
        if not image_groups:
            html += "<p>⚠️ No images found in any run.</p>\n"
        else:
            # Display each image comparison
            colors = ['#2e86de', '#10ac84', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c', '#34495e']
            
            for base_name in sorted(image_groups.keys()):
                html += f"<div class='image-comparison'>\n"
                html += f"<h3>{base_name}</h3>\n"
                html += "<div class='image-container'>\n"
                
                for i, run_id in enumerate(self.run_ids):
                    img_file = image_groups[base_name][run_id]
                    html += f"<div class='image-item'>\n"
                    html += f"<h4 style='color: {colors[i % len(colors)]};'>Run {i+1} ({run_id})</h4>\n"
                    
                    if img_file:
                        image = download_image(img_file['url'])
                        if image:
                            buffered = BytesIO()
                            image.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            html += f'<img src="data:image/png;base64,{img_str}" alt="{base_name}" />\n'
                        else:
                            html += "<p style='color: #dc3545; font-style: italic;'>Image not found</p>\n"
                    else:
                        html += "<p style='color: #dc3545; font-style: italic;'>No matching image</p>\n"
                    
                    html += "</div>\n"
                
                html += "</div>\n</div>\n"
        
        html += "</div>\n"
        return html
    
    def get_summary_html(self) -> str:
        """Generate summary HTML."""
        html = "<div class='section'>\n<h2>📝 Run Comparison Summary</h2>\n"
        
        # Calculate common metrics
        all_summary_keys = [set(self.runs_data[run_id]['summary'].keys()) for run_id in self.run_ids]
        common_summary_keys = sorted(set.intersection(*all_summary_keys))
        
        all_config_keys = [set(self.runs_data[run_id]['config'].keys()) for run_id in self.run_ids]
        common_config_keys = sorted(set.intersection(*all_config_keys))
        
        all_history_keys = [set(self.runs_data[run_id]['history_keys']) for run_id in self.run_ids]
        common_metrics = sorted(set.intersection(*all_history_keys))
        
        for i, run_id in enumerate(self.run_ids, 1):
            meta = self.runs_data[run_id]['meta']
            html += f"<div class='summary-card'>\n"
            html += f"<h3>Run {i}: {run_id}</h3>\n"
            html += "<table class='summary-table'>\n"
            html += f"<tr><td><strong>Name:</strong></td><td>{meta['name']}</td></tr>\n"
            html += f"<tr><td><strong>Display:</strong></td><td>{meta['display_name']}</td></tr>\n"
            html += f"<tr><td><strong>State:</strong></td><td>{meta['state']}</td></tr>\n"
            html += f"<tr><td><strong>Created:</strong></td><td>{meta['created_at']}</td></tr>\n"
            html += f"<tr><td><strong>Tags:</strong></td><td>{', '.join(meta['tags']) if meta['tags'] else 'None'}</td></tr>\n"
            html += f"<tr><td><strong>Group:</strong></td><td>{meta['group'] or 'None'}</td></tr>\n"
            html += f"<tr><td><strong>Job Type:</strong></td><td>{meta['job_type']}</td></tr>\n"
            html += f"<tr><td><strong>History:</strong></td><td>{len(self.runs_data[run_id]['history'])} steps, {len(self.runs_data[run_id]['history_keys'])} metrics</td></tr>\n"
            html += f"<tr><td><strong>Files:</strong></td><td>{len(self.runs_data[run_id]['files'])}</td></tr>\n"
            html += f"<tr><td><strong>Artifacts:</strong></td><td>{len(self.runs_data[run_id]['artifacts'])}</td></tr>\n"
            html += f"<tr><td><strong>Config:</strong></td><td>{len(self.runs_data[run_id]['config'])} parameters</td></tr>\n"
            html += f"<tr><td><strong>Summary:</strong></td><td>{len(self.runs_data[run_id]['summary'])} metrics</td></tr>\n"
            html += f"<tr><td><strong>WandB URL:</strong></td><td><a href='{meta['url']}' target='_blank'>{meta['url']}</a></td></tr>\n"
            html += "</table>\n</div>\n"
        
        html += f"<div class='key-stats'>\n"
        html += f"<h3>Key Statistics</h3>\n"
        html += f"<ul>\n"
        html += f"<li>Common summary metrics: <strong>{len(common_summary_keys)}</strong></li>\n"
        html += f"<li>Common config keys: <strong>{len(common_config_keys)}</strong></li>\n"
        html += f"<li>Common history metrics: <strong>{len(common_metrics)}</strong></li>\n"
        html += f"</ul>\n"
        html += f"</div>\n"
        html += "</div>\n"
        
        return html
    
    def generate_html_report(self, output_file: str = "report.html"):
        """Generate complete HTML report."""
        print(f"\nGenerating HTML report: {output_file}")
        
        # Generate all sections
        sections = [
            self.get_summary_html(),
            self.get_comparison_tables_html(),
        ]
        
        # Add training history plot if available
        training_plot = self.get_training_history_plot()
        if training_plot:
            sections.append(training_plot)
        
        # Add summary metrics plot if available
        summary_plot = self.get_summary_metrics_plot()
        if summary_plot:
            sections.append(summary_plot)
        
        # Add difference analysis if available
        diff_analysis = self.get_difference_analysis_html()
        if diff_analysis:
            sections.append(diff_analysis)
        
        sections.extend([
            self.get_files_artifacts_html(),
            self.get_images_comparison_html(),
        ])
        
        # Generate complete HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Run Comparison Report - {self.run_ids[0]} vs {self.run_ids[1]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 50px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .section h3 {{
            color: #764ba2;
            margin-bottom: 15px;
            margin-top: 20px;
        }}
        
        .comparison-container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .comparison-card {{
            flex: 1;
            min-width: 300px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            overflow-x: auto;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        .summary-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .summary-table td {{
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .key-stats {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        
        .key-stats h3 {{
            color: white;
            margin-bottom: 10px;
        }}
        
        .key-stats ul {{
            list-style-type: none;
        }}
        
        .key-stats li {{
            padding: 5px 0;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        .data-table th, .data-table td {{
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        
        .data-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }}
        
        .data-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .data-table tr:hover {{
            background: #e3f2fd;
        }}
        
        .image-comparison {{
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        
        .image-container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .image-item {{
            flex: 1;
            min-width: 300px;
            text-align: center;
        }}
        
        .image-item img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            object-fit: contain;
        }}
        
        .plotly {{
            width: 100%;
        }}
        
        footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 30px;
        }}
        
        @media (max-width: 768px) {{
            .comparison-container {{
                flex-direction: column;
            }}
            
            .image-container {{
                flex-direction: column;
            }}
            
            .comparison-card, .image-item {{
                min-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 WandB Run Comparison Report</h1>
            <p>Comparing {self.run_ids[0]} vs {self.run_ids[1]}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </header>
        
        <div class="content">
"""
        
        html += "\n".join(sections)
        
        html += """
        </div>
        
        <footer>
            <p>Generated by CounterFactualDPG Run Comparison Tool</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # Write HTML to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ HTML report successfully generated: {output_file}")


def main():
    """Main function to run the comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare two WandB runs and generate an HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s poxmea6n buqt4b6u
  %(prog)s poxmea6n buqt4b6u --output comparison.html
  %(prog)s poxmea6n buqt4b6u --entity myentity --project myproject
        """
    )
    
    parser.add_argument(
        'run_ids',
        nargs=2,
        help='Two WandB run IDs to compare'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='report.html',
        help='Output HTML file path (default: report.html)'
    )
    
    parser.add_argument(
        '--entity', '-e',
        default='mllab-ts-universit-di-trieste',
        help='WandB entity name (default: mllab-ts-universit-di-trieste)'
    )
    
    parser.add_argument(
        '--project', '-p',
        default='CounterFactualDPG',
        help='WandB project name (default: CounterFactualDPG)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WANDB RUN COMPARISON TOOL")
    print("=" * 80)
    print(f"\nComparing runs:")
    print(f"  1. {args.run_ids[0]}")
    print(f"  2. {args.run_ids[1]}")
    print(f"\nEntity: {args.entity}")
    print(f"Project: {args.project}")
    print(f"Output: {args.output}")
    
    # Create comparator and load data
    comparator = RunComparator(args.run_ids, args.entity, args.project)
    
    if not comparator.load_all_runs():
        print("\n❌ Error: Failed to load all run data")
        sys.exit(1)
    
    # Generate HTML report
    print("\nGenerating comparison report...")
    comparator.generate_html_report(args.output)
    
    print("\n" + "=" * 80)
    print("✓ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOpen {args.output} in your browser to view the comparison report.")


if __name__ == '__main__':
    main()
