"""
Metrics Reporter: Generate comprehensive single-run analysis reports

From a senior data analyst perspective, each run report should include:
- Data quality metrics (pass rates, data loss funnel)
- Performance bottleneck analysis (timeline, latency distribution)
- Resource utilization (worker efficiency, parallelism)
- Data flow visualization (Sankey, funnel charts)
- Anomaly detection (outliers, performance issues)
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


class MetricsReporter:
    """Generate comprehensive single-run HTML reports with advanced visualizations."""

    def __init__(self, metrics_path: str):
        """Initialize reporter.

        Args:
            metrics_path: Path to metrics directory containing Parquet files
        """
        self.metrics_path = Path(metrics_path)
        self.runs_path = self.metrics_path / "runs"
        self.stages_path = self.metrics_path / "stages"
        self.operators_path = self.metrics_path / "operators"

        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self.jinja_env.filters["format_metric"] = self._format_metric_value

    @staticmethod
    def _format_metric_value(value: float | int, format_type: str, precision: int) -> str:
        """Format metric value according to type and precision.

        Args:
            value: The numeric value to format
            format_type: Type of formatting (percent, time, latency, number, integer)
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        if pd.isna(value):
            return "N/A"

        if format_type == "percent":
            return f"{value:.{precision}f}%"
        elif format_type == "time":
            return f"{value:.{precision}f}s"
        elif format_type == "latency":
            # Convert seconds to milliseconds
            return f"{value * 1000:.{precision}f}ms"
        elif format_type == "number":
            return f"{value:,.{precision}f}"
        elif format_type == "integer":
            return f"{int(value):,}"
        else:
            return str(value)

    def load_single_run_metrics(self, run_id: str) -> tuple[pd.Series | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Load metrics for a specific run.

        Args:
            run_id: Run ID to load

        Returns:
            Tuple of (run_series, stage_df, operator_df)
        """
        # Load run metrics
        run_series = None
        run_file = self.runs_path / f"{run_id}.parquet"
        if run_file.exists():
            run_df = pd.read_parquet(run_file)
            run_series = run_df.iloc[0] if len(run_df) > 0 else None

        # Load stage metrics for this run
        stage_df = None
        stage_files = list(self.stages_path.glob(f"stages_{run_id}*.parquet"))
        if stage_files:
            stage_df = pd.concat([pd.read_parquet(f) for f in stage_files], ignore_index=True)
            stage_df = stage_df[stage_df["run_id"] == run_id].copy()

        # Load operator metrics for this run
        operator_df = None
        operator_files = list(self.operators_path.glob(f"operators_{run_id}*.parquet"))
        if operator_files:
            operator_df = pd.concat([pd.read_parquet(f) for f in operator_files], ignore_index=True)
            operator_df = operator_df[operator_df["run_id"] == run_id].copy()

        return run_series, stage_df, operator_df

    def get_latest_run_id(self) -> str | None:
        """Get the most recent run ID.

        Returns:
            Latest run ID or None if no runs found
        """
        if not self.runs_path.exists():
            return None

        run_files = list(self.runs_path.glob("run_*.parquet"))
        if not run_files:
            return None

        # Sort by modification time, get latest
        latest_file = max(run_files, key=lambda f: f.stat().st_mtime)
        # Extract run_id from filename: {run_id}.parquet (run_id already contains "run_" prefix)
        run_id = latest_file.stem
        return run_id

    def generate_single_run_report(
        self,
        run_id: str | None = None,
        output_path: str | None = None,
    ) -> str:
        """Generate comprehensive single-run HTML report.

        Args:
            run_id: Run ID to analyze (if None, uses latest run)
            output_path: Output path for HTML report (if None, auto-generated)

        Returns:
            Path to generated report
        """
        # Use latest run if not specified
        if run_id is None:
            run_id = self.get_latest_run_id()
            if run_id is None:
                raise ValueError("No runs found in metrics directory")

        # Load metrics for this run
        run_series, stage_df, operator_df = self.load_single_run_metrics(run_id)

        if run_series is None:
            raise ValueError(f"Run {run_id} not found")

        # Generate HTML
        html = self._generate_single_run_html(run_id, run_series, stage_df, operator_df)

        # Determine output path
        if output_path is None:
            output_path = self.metrics_path / f"report_run_{run_id}.html"
        else:
            output_path = Path(output_path)

        # Write to file
        output_path.write_text(html, encoding="utf-8")

        return str(output_path)

    def _generate_single_run_html(
        self,
        run_id: str,
        run_series: pd.Series,
        stage_df: pd.DataFrame | None,
        operator_df: pd.DataFrame | None,
    ) -> str:
        """Generate HTML for single run analysis using Jinja2 template."""
        try:
            import plotly.graph_objects as go  # noqa: F401
            from plotly.subplots import make_subplots  # noqa: F401
        except ImportError:
            return self._generate_simple_html(run_id, run_series, stage_df, operator_df)

        # Prepare charts list
        charts = []

        # 2. Data Quality Funnel
        if stage_df is not None and len(stage_df) > 0:
            charts.append(
                {
                    "title": "🔻 Data Quality Funnel",
                    "description": "Visualize data filtering and pass rates at stage and operator levels",
                    "html": self._generate_data_funnel(stage_df, operator_df),
                    "alert": None,
                }
            )

        # 3. Data Flow Sankey Diagram
        if stage_df is not None and len(stage_df) > 0:
            charts.append(
                {
                    "title": "🌊 Data Flow Sankey",
                    "description": "Serial data flow through operators within each stage",
                    "html": self._generate_sankey_diagram(stage_df, operator_df),
                    "alert": None,
                }
            )

        # 4. Performance Timeline
        if stage_df is not None and len(stage_df) > 0:
            charts.append(
                {
                    "title": "⏱️ Performance Timeline",
                    "description": "Stage execution timeline showing duration of each stage",
                    "html": self._generate_timeline_chart(stage_df),
                    "alert": None,
                }
            )

        # 5. Bottleneck Analysis
        if stage_df is not None and operator_df is not None and len(operator_df) > 0:
            # Calculate bottleneck data
            slowest_stage = stage_df.loc[stage_df["total_time"].idxmax()]
            slowest_operator = operator_df.loc[operator_df["throughput"].idxmin()]

            charts.append(
                {
                    "title": "🔍 Bottleneck Analysis",
                    "description": "Identify performance bottlenecks in stages and operators",
                    "html": self._generate_bottleneck_chart(stage_df, operator_df),
                    "alert": {
                        "slowest_stage_name": slowest_stage["stage_name"],
                        "slowest_stage_time": slowest_stage["total_time"],
                        "slowest_stage_throughput": slowest_stage["avg_throughput"],
                        "slowest_operator_name": slowest_operator["operator_name"],
                        "slowest_operator_throughput": slowest_operator["throughput"],
                        "slowest_operator_latency": slowest_operator["avg_latency"],
                    },
                }
            )

        # 6. Latency Distribution Heatmap
        if operator_df is not None and len(operator_df) > 0:
            charts.append(
                {
                    "title": "🌡️ Latency Heatmap",
                    "description": "Latency percentile distribution across operators",
                    "html": self._generate_latency_heatmap(operator_df),
                    "alert": None,
                }
            )

        # 7. Throughput vs Latency Scatter
        if operator_df is not None and len(operator_df) > 0:
            charts.append(
                {
                    "title": "💡 Throughput vs Latency Analysis",
                    "description": "Performance scatter plot with bubble size representing workload",
                    "html": self._generate_throughput_latency_scatter(operator_df),
                    "alert": None,
                }
            )

        # 8. Stage Duration Waterfall
        if stage_df is not None and len(stage_df) > 0:
            charts.append(
                {
                    "title": "📊 Duration Waterfall",
                    "description": "Cumulative stage duration breakdown",
                    "html": self._generate_duration_waterfall(stage_df),
                    "alert": None,
                }
            )

        # 9. Detailed Metrics Tables
        tables = self._generate_detailed_tables(stage_df, operator_df)

        # Render template
        template = self.jinja_env.get_template("report.html.jinja2")
        return template.render(
            run_id=run_id,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            run_metrics=run_series,
            charts=charts,
            tables=tables,
        )

    def _generate_data_funnel(self, stage_df: pd.DataFrame, operator_df: pd.DataFrame | None) -> str:
        """Generate data quality funnel charts showing data loss at stage and operator levels."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Sort stages by order
        stage_df = stage_df.sort_values("stage_name")

        # Create subplot with 2 rows, 1 column (stacked vertically)
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Stage-Level Funnel", "Operator-Level Funnel"),
            specs=[[{"type": "funnel"}], [{"type": "funnel"}]],
            vertical_spacing=0.15,
        )

        # ===== Top: Stage-level funnel =====
        stages = stage_df["stage_name"].tolist()
        stage_input = stage_df["input_records"].tolist()
        stage_output = stage_df["output_records"].tolist()

        # Input funnel for stages
        fig.add_trace(
            go.Funnel(
                name="Stage Input",
                y=stages,
                x=stage_input,
                textinfo="value+percent initial",
                marker={"color": "#3498db"},
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Output funnel for stages
        fig.add_trace(
            go.Funnel(
                name="Stage Output",
                y=stages,
                x=stage_output,
                textinfo="value+percent previous",
                marker={"color": "#2ecc71"},
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # ===== Bottom: Operator-level funnel =====
        if operator_df is not None and len(operator_df) > 0:
            # Aggregate operators by stage and operator_name, preserve execution order
            op_agg = (
                operator_df.groupby(["stage_name", "operator_name"])
                .agg(
                    {
                        "input_records": "sum",
                        "output_records": "sum",
                        "pass_rate": "mean",
                        "timestamp": "min",  # Preserve execution order
                    }
                )
                .reset_index()
                .sort_values(["stage_name", "timestamp"])
            )

            # Format labels and collect data
            operators = []
            op_input = []
            op_output = []

            for _, row in op_agg.iterrows():
                # Format: "stage: operator"
                label = f"{row['stage_name']}: {row['operator_name']}"
                operators.append(label)
                op_input.append(row["input_records"])
                op_output.append(row["output_records"])

            # Input funnel for operators
            fig.add_trace(
                go.Funnel(
                    name="Operator Input",
                    y=operators,
                    x=op_input,
                    textinfo="value+percent initial",
                    marker={"color": "#e74c3c"},
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

            # Output funnel for operators
            fig.add_trace(
                go.Funnel(
                    name="Operator Output",
                    y=operators,
                    x=op_output,
                    textinfo="value+percent previous",
                    marker={"color": "#f39c12"},
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title_text="Data Quality Funnel - Stage & Operator Level Analysis",
            height=800,  # Increased height for vertical layout
            template="plotly_white",
            showlegend=True,
        )

        chart_html = fig.to_html(include_plotlyjs=False, div_id="funnel-chart", config={"responsive": True})
        return chart_html

    def _generate_sankey_diagram(self, stage_df: pd.DataFrame, operator_df: pd.DataFrame | None) -> str:
        """Generate Sankey diagram: Input splits into multiple filters per stage, then continues."""
        import plotly.graph_objects as go

        if operator_df is None or len(operator_df) == 0:
            return "<p>No operator data available</p>"

        # Aggregate operators by stage and operator_name (sum across workers)
        op_agg = (
            operator_df.groupby(["stage_name", "operator_name"])
            .agg(
                {
                    "input_records": "sum",
                    "output_records": "sum",
                    "pass_rate": "mean",
                    "timestamp": "min",
                }
            )
            .reset_index()
            .sort_values(["stage_name", "timestamp"])
        )

        # Calculate x positions for stages
        num_stages = len(stage_df)
        x_step = 0.8 / (num_stages + 1)

        # Build nodes with explicit positions
        node_labels = []
        node_colors = []
        node_x = []
        node_y = []

        # Node: Input
        node_labels.append("Input")
        node_colors.append("#3498db")
        node_x.append(0.05)
        node_y.append(0.5)
        input_idx = 0

        # For each stage: create filter nodes for each operator that filters
        stage_node_map = {}  # Maps stage_name to the retained/output node index
        filter_node_indices = {}  # Maps (stage_name, operator_name) to filter node index

        for stage_idx, (_, stage_row) in enumerate(stage_df.iterrows()):
            stage_name = stage_row["stage_name"]
            stage_x = 0.1 + (stage_idx + 1) * x_step

            # Get operators in this stage
            stage_ops = op_agg[op_agg["stage_name"] == stage_name]

            # Create filter nodes for each operator that filters data
            y_offset = 0.2
            for _, op_row in stage_ops.iterrows():
                filtered = op_row["input_records"] - op_row["output_records"]
                if filtered > 0:
                    filter_idx = len(node_labels)
                    filter_node_indices[(stage_name, op_row["operator_name"])] = filter_idx
                    node_labels.append(f"{op_row['operator_name']}\nFiltered")
                    node_colors.append("#95a5a6")
                    node_x.append(stage_x)
                    node_y.append(y_offset)
                    y_offset += 0.15

            # Create stage output node (retained data after all filters)
            stage_output_idx = len(node_labels)
            stage_node_map[stage_name] = stage_output_idx
            node_labels.append(f"{stage_name}\nOutput")
            node_colors.append("#667eea")
            node_x.append(stage_x)
            node_y.append(0.75)  # Bottom position

        # Node: Final Output
        output_idx = len(node_labels)
        node_labels.append("Output")
        node_colors.append("#27ae60")
        node_x.append(0.95)
        node_y.append(0.5)

        # Build links
        sources = []
        targets = []
        values = []
        link_colors = []

        # Process each stage
        prev_source_idx = input_idx
        prev_source_value = stage_df.iloc[0]["input_records"]

        for stage_idx, (_, stage_row) in enumerate(stage_df.iterrows()):
            stage_name = stage_row["stage_name"]
            stage_output_idx = stage_node_map[stage_name]

            # Get operators in this stage
            stage_ops = op_agg[op_agg["stage_name"] == stage_name]

            # Links from previous source to all filter nodes in this stage
            for _, op_row in stage_ops.iterrows():
                filtered = op_row["input_records"] - op_row["output_records"]
                if filtered > 0:
                    filter_idx = filter_node_indices[(stage_name, op_row["operator_name"])]
                    sources.append(prev_source_idx)
                    targets.append(filter_idx)
                    values.append(filtered)
                    link_colors.append("rgba(231, 76, 60, 0.4)")  # Red for filtered

            # Link from previous source to stage output (retained data)
            sources.append(prev_source_idx)
            targets.append(stage_output_idx)
            values.append(stage_row["output_records"])
            link_colors.append("rgba(39, 174, 96, 0.5)")  # Green for retained

            # Update previous source for next stage
            prev_source_idx = stage_output_idx
            prev_source_value = stage_row["output_records"]

        # Final link: last stage output → Final Output
        sources.append(prev_source_idx)
        targets.append(output_idx)
        values.append(prev_source_value)
        link_colors.append("rgba(39, 174, 96, 0.6)")

        # Create Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="white", width=2),
                        label=node_labels,
                        color=node_colors,
                        x=node_x,
                        y=node_y,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Data Flow Sankey Diagram - Stage Level with Filters",
            font=dict(size=10),
            height=700,
            template="plotly_white",
        )

        chart_html = fig.to_html(include_plotlyjs=False, div_id="dataflow-chart", config={"responsive": True})
        return chart_html

    def _generate_timeline_chart(self, stage_df: pd.DataFrame) -> str:
        """Generate timeline chart showing stage execution durations."""
        import plotly.graph_objects as go

        stage_df = stage_df.sort_values("stage_name")

        # Calculate start times (cumulative)
        start_times = [0]
        for i in range(len(stage_df) - 1):
            start_times.append(start_times[-1] + stage_df.iloc[i]["total_time"])

        fig = go.Figure()

        for i, (_idx, row) in enumerate(stage_df.iterrows()):
            fig.add_trace(
                go.Bar(
                    name=row["stage_name"],
                    x=[row["total_time"]],
                    y=[row["stage_name"]],
                    orientation="h",
                    marker={
                        "color": f"rgba({50 + i * 40}, {100 + i * 30}, {200 - i * 20}, 0.8)",
                    },
                    text=[f"{row['total_time']:.2f}s"],
                    textposition="inside",
                )
            )

        fig.update_layout(
            title="Stage Execution Timeline",
            xaxis_title="Duration (seconds)",
            yaxis_title="Stage",
            barmode="stack",
            height=400,
            template="plotly_white",
            showlegend=False,
        )

        chart_html = fig.to_html(include_plotlyjs=False, div_id="timeline-chart", config={"responsive": True})
        return chart_html

    def _generate_bottleneck_chart(self, stage_df: pd.DataFrame, operator_df: pd.DataFrame) -> str:
        """Generate bottleneck analysis chart."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Find slowest stage
        slowest_stage = stage_df.loc[stage_df["total_time"].idxmax()]

        # Find operator with lowest throughput
        slowest_operator = operator_df.loc[operator_df["throughput"].idxmin()]

        # Find operators with highest latency
        operator_summary = (
            operator_df.groupby("operator_name").agg({"avg_latency": "mean", "throughput": "mean"}).reset_index()
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Stage Duration (Identify Bottleneck)", "Operator Throughput (records/s)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        # Stage durations
        fig.add_trace(
            go.Bar(
                x=stage_df["stage_name"],
                y=stage_df["total_time"],
                name="Duration",
                marker_color=[
                    "#e74c3c" if name == slowest_stage["stage_name"] else "#3498db" for name in stage_df["stage_name"]
                ],
                text=[f"{d:.2f}s" for d in stage_df["total_time"]],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        # Operator throughput
        fig.add_trace(
            go.Bar(
                x=operator_summary["operator_name"],
                y=operator_summary["throughput"],
                name="Throughput",
                marker_color=[
                    "#e74c3c" if name == slowest_operator["operator_name"] else "#2ecc71"
                    for name in operator_summary["operator_name"]
                ],
                text=[f"{t:.1f}" for t in operator_summary["throughput"]],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Stage", row=1, col=1)
        fig.update_xaxes(title_text="Operator", row=1, col=2)
        fig.update_yaxes(title_text="Duration (s)", row=1, col=1)
        fig.update_yaxes(title_text="Throughput (rec/s)", row=1, col=2)

        fig.update_layout(height=500, template="plotly_white", showlegend=False)

        return fig.to_html(include_plotlyjs=False, div_id="bottleneck-chart", config={"responsive": True})

    def _generate_latency_heatmap(self, operator_df: pd.DataFrame) -> str:
        """Generate heatmap of latency percentiles."""
        import plotly.graph_objects as go

        # Prepare data for heatmap
        operators = operator_df["operator_name"].unique()

        data_matrix = []
        for op in operators:
            op_data = operator_df[operator_df["operator_name"] == op].iloc[0]
            data_matrix.append(
                [
                    op_data["min_latency"],
                    op_data["p50_latency"],
                    op_data["p95_latency"],
                    op_data["p99_latency"],
                    op_data["max_latency"],
                ]
            )

        fig = go.Figure(
            data=go.Heatmap(
                z=data_matrix,
                x=["Min", "P50", "P95", "P99", "Max"],
                y=list(operators),
                colorscale="RdYlGn_r",
                text=[[f"{v:.3f}s" for v in row] for row in data_matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar={"title": "Latency (s)"},
            )
        )

        fig.update_layout(
            title="Latency Heatmap - Percentile Distribution",
            xaxis_title="Percentile",
            yaxis_title="Operator",
            height=400,
            template="plotly_white",
        )

        chart_html = fig.to_html(include_plotlyjs=False, div_id="heatmap-chart", config={"responsive": True})
        return chart_html

    def _generate_throughput_latency_scatter(self, operator_df: pd.DataFrame) -> str:
        """Generate scatter plot of throughput vs latency with bubble size = input_records."""
        import plotly.graph_objects as go

        operator_summary = (
            operator_df.groupby("operator_name")
            .agg({"throughput": "mean", "avg_latency": "mean", "input_records": "sum", "error_count": "sum"})
            .reset_index()
        )

        # Convert latency from seconds to milliseconds for better readability
        operator_summary["avg_latency_ms"] = operator_summary["avg_latency"] * 1000

        fig = go.Figure(
            data=go.Scatter(
                x=operator_summary["avg_latency_ms"],
                y=operator_summary["throughput"],
                mode="markers+text",
                marker={
                    "size": operator_summary["input_records"] / operator_summary["input_records"].max() * 100,
                    "color": operator_summary["error_count"],
                    "colorscale": "Reds",
                    "showscale": True,
                    "colorbar": {"title": "Errors"},
                    "line": {"width": 1, "color": "white"},
                },
                text=operator_summary["operator_name"],
                textposition="top center",
                textfont={"size": 10},
                hovertemplate="<b>%{text}</b><br>Latency: %{x:.3f} ms<br>Throughput: %{y:,.0f} rec/s<extra></extra>",
            )
        )

        fig.update_layout(
            title="Throughput vs Latency Scatter (bubble size = input records, color = errors)",
            xaxis_title="Average Latency (ms)",
            yaxis_title="Throughput (records/s)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            template="plotly_white",
        )

        chart_html = fig.to_html(include_plotlyjs=False, div_id="scatter-chart", config={"responsive": True})
        return chart_html

    def _generate_duration_waterfall(self, stage_df: pd.DataFrame) -> str:
        """Generate waterfall chart showing cumulative stage durations."""
        import plotly.graph_objects as go

        stage_df = stage_df.sort_values("stage_name")

        # Waterfall data
        x = list(stage_df["stage_name"]) + ["Total"]
        y = list(stage_df["total_time"]) + [stage_df["total_time"].sum()]

        # Build measure list
        measure = ["relative"] * len(stage_df) + ["total"]

        fig = go.Figure(
            go.Waterfall(
                x=x,
                y=y,
                measure=measure,
                text=[f"{v:.2f}s" for v in y],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )

        fig.update_layout(
            title="Duration Waterfall - Cumulative Stage Time",
            xaxis_title="Stage",
            yaxis_title="Duration (seconds)",
            height=400,
            template="plotly_white",
        )

        chart_html = fig.to_html(include_plotlyjs=False, div_id="waterfall-chart", config={"responsive": True})
        return chart_html

    def _generate_detailed_tables(
        self, stage_df: pd.DataFrame | None, operator_df: pd.DataFrame | None
    ) -> list[dict[str, str | list[str] | list[dict[str, Any]]]]:
        """Generate detailed data tables as structured data.

        Returns:
            List of table dictionaries with title, columns, rows, and column format specs
        """
        tables = []

        if stage_df is not None and len(stage_df) > 0:
            # Select useful columns and exclude run_id
            stage_cols = [
                "stage_name",
                "num_workers",
                "input_records",
                "output_records",
                "pass_rate",
                "total_time",
                "avg_throughput",
                "error_count",
            ]
            stage_display = stage_df[[col for col in stage_cols if col in stage_df.columns]].copy()

            # Define formatting rules
            format_rules = {
                "pass_rate": ("percent", 1),
                "total_time": ("time", 2),
                "avg_throughput": ("number", 1),
                "input_records": ("integer", 0),
                "output_records": ("integer", 0),
            }

            tables.append(
                {
                    "title": "Stage Metrics",
                    "columns": list(stage_display.columns),
                    "rows": stage_display.to_dict("records"),
                    "formats": format_rules,
                }
            )

        if operator_df is not None and len(operator_df) > 0:
            # Aggregate to operator level (sum across workers)
            op_agg = (
                operator_df.groupby(["stage_name", "operator_name"])
                .agg(
                    {
                        "worker_id": "count",  # Count number of workers
                        "input_records": "sum",
                        "output_records": "sum",
                        "pass_rate": "mean",
                        "total_time": "max",  # Use max as bottleneck
                        "avg_latency": "mean",
                        "throughput": ["mean", "min", "max"],  # Worker throughput stats
                        "error_count": "sum",
                        "timestamp": "min",  # Preserve execution order
                    }
                )
                .reset_index()
            )

            # Flatten multi-level columns from aggregation
            op_agg.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in op_agg.columns.values]

            # Rename columns
            op_agg = op_agg.rename(
                columns={
                    "worker_id_count": "num_workers",
                    "input_records_sum": "input_records",
                    "output_records_sum": "output_records",
                    "pass_rate_mean": "pass_rate",
                    "total_time_max": "total_time",
                    "avg_latency_mean": "avg_latency",
                    "error_count_sum": "error_count",
                    "timestamp_min": "timestamp",
                    "throughput_mean": "worker_avg_throughput",
                    "throughput_min": "worker_min_throughput",
                    "throughput_max": "worker_max_throughput",
                }
            )

            # Calculate overall throughput (total records / max time)
            op_agg["overall_throughput"] = op_agg["input_records"] / op_agg["total_time"]

            # Sort by stage and timestamp to preserve execution order
            op_agg = op_agg.sort_values(["stage_name", "timestamp"])

            # Select and order columns for display
            op_cols = [
                "stage_name",
                "operator_name",
                "num_workers",
                "input_records",
                "output_records",
                "pass_rate",
                "total_time",
                "avg_latency",
                "overall_throughput",
                "worker_avg_throughput",
                "worker_min_throughput",
                "worker_max_throughput",
                "error_count",
            ]
            op_display = op_agg[[col for col in op_cols if col in op_agg.columns]].copy()

            # Define formatting rules
            format_rules = {
                "pass_rate": ("percent", 1),
                "total_time": ("time", 2),
                "avg_latency": ("latency", 2),
                "overall_throughput": ("number", 1),
                "worker_avg_throughput": ("number", 1),
                "worker_min_throughput": ("number", 1),
                "worker_max_throughput": ("number", 1),
                "input_records": ("integer", 0),
                "output_records": ("integer", 0),
            }

            tables.append(
                {
                    "title": "Operator Metrics (Aggregated by Operator)",
                    "columns": list(op_display.columns),
                    "rows": op_display.to_dict("records"),
                    "formats": format_rules,
                }
            )

        return tables

    def _generate_simple_html(
        self,
        run_id: str,
        run_series: pd.Series,
        stage_df: pd.DataFrame | None,
        operator_df: pd.DataFrame | None,
    ) -> str:
        """Generate simple HTML without plotly (fallback)."""
        sections = []

        sections.append(f"""
        <h2>Run Summary</h2>
        <table border="1">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Run ID</td><td>{run_id}</td></tr>
            <tr><td>Duration</td><td>{run_series["duration"]:.2f}s</td></tr>
            <tr><td>Input Records</td><td>{int(run_series["total_input_records"]):,}</td></tr>
            <tr><td>Output Records</td><td>{int(run_series["total_output_records"]):,}</td></tr>
            <tr><td>Pass Rate</td><td>{run_series["overall_pass_rate"]:.2f}%</td></tr>
        </table>
        """)

        if stage_df is not None:
            sections.append(f"<h2>Stage Metrics</h2>{stage_df.to_html(index=False)}")

        if operator_df is not None:
            sections.append(f"<h2>Operator Metrics</h2>{operator_df.to_html(index=False)}")

        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Run {run_id} Report</title></head>
<body style="font-family: Arial; padding: 20px;">
<h1>Pipeline Metrics Report - Run {run_id}</h1>
{"".join(sections)}
</body></html>"""

    def publish_to_huggingface(
        self,
        report_path: str,
        repo_id: str,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        """Publish HTML report to HuggingFace Space.

        Args:
            report_path: Path to HTML report file
            repo_id: HuggingFace repo ID (e.g., "username/space-name")
            token: HuggingFace API token (if None, uses HF_TOKEN env var)
            commit_message: Commit message for the upload

        Returns:
            URL of the published space
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for publishing to HuggingFace. "
                "Install it with: pip install huggingface_hub"
            ) from e

        api = HfApi(token=token)

        # Create repo if it doesn't exist (Space type)
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="static",
                exist_ok=True,
                token=token,
            )
        except Exception as e:
            print(f"Warning: Could not create repo (may already exist): {e}")

        # Upload the report as index.html
        if commit_message is None:
            commit_message = f"Update metrics report - {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"

        api.upload_file(
            path_or_fileobj=report_path,
            path_in_repo="index.html",
            repo_id=repo_id,
            repo_type="space",
            commit_message=commit_message,
            token=token,
        )

        space_url = f"https://huggingface.co/spaces/{repo_id}"
        print(f"Report published to: {space_url}")

        return space_url
