"""Dashboard regressions using the packaged SimBench calendar and controlled voltages.

Load helper definitions without executing the Streamlit page's top-level UI.
No models are fitted and no experiment outputs are changed.
"""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'seasonal'))
import dsse
from config import NETWORK
from seasonal_evaluation import save_daily_diagnostics, PRIMARY_SCOPE, ALL_SCOPE


def dashboard_helpers():
    names = {
        'bus_identity_lookup', 'frame_bus_label_lookup', 'bus_order_from_frame',
        'validate_day_grid', 'validate_day_detail', 'filter_detail_scope',
        'verify_day_metrics', 'annual_day_heatmap_figure',
        'annual_day_all_buses_figure', 'annual_day_bus_timeseries_figure',
    }
    source = ast.parse((ROOT / 'seasonal/dashboard.py').read_text())
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    for node in definitions:
        node.decorator_list = []
    namespace = dict(np=np, pd=pd, go=go, Path=Path, dsse=dsse,
                     season_for_day=dsse.season_for_day,
                     PRIMARY_SCOPE=PRIMARY_SCOPE, ALL_SCOPE=ALL_SCOPE)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT / 'seasonal/dashboard.py'), 'exec'), namespace)
    return namespace


class DailyHeatmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = dashboard_helpers()
        net = dsse.load_network(NETWORK)
        cls.timeline = dsse.build_simbench_timeline(net)
        dsse.validate_simbench_timeline(cls.timeline)
        cls.helpers['dashboard_timeline'] = lambda network: cls.timeline.copy()
        refs = dsse.get_reference_buses(net)
        cls.buses = [int(b) for b in net.bus.index if b not in refs][:3]
        cls.cfg = dict(network=NETWORK, prediction_buses=cls.buses,
                       measured_buses=cls.buses[:1], reference_buses=refs,
                       bus_identity=dsse.bus_identity_records(net), gaussian_95_multiplier=1.96)

    def fixture(self, day):
        timeline = self.timeline[self.timeline.day_of_year.eq(day)]
        rows = []
        for bus_index, bus in enumerate(self.buses):
            for step_index, row in enumerate(timeline.itertuples(index=False)):
                # Vary both dimensions to expose transposes and bus/step misalignment.
                mean = 1 + bus_index * .01 + step_index * .00001
                sigma = .001 + bus_index * .0001
                lower, upper = mean - 1.96 * sigma, mean + 1.96 * sigma
                truth = [mean, lower, upper, upper + .001][step_index % 4]
                rows.append(dict(bus=bus, time_step=row.simbench_step,
                                 simbench_step=row.simbench_step, timestamp=row.timestamp,
                                 date=row.date, step_in_day=row.step_in_day,
                                 day_of_year=day, time=row.timestamp[-5:], season=dsse.season_for_day(day),
                                 ground_truth_vm_pu=truth, predicted_vm_pu=mean,
                                 sigma_vm_pu=sigma, lower_95_vm_pu=lower, upper_95_vm_pu=upper))
        return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    def validate(self, frame, day):
        return self.helpers['validate_day_detail'](frame, day, self.cfg)

    def test_real_calendar_and_every_heatmap_cell(self):
        for day, count in [(1, 96), (87, 92), (304, 100)]:
            with self.subTest(day=day):
                detail = self.validate(self.fixture(day), day)
                steps = sorted(detail.time_step.unique())
                self.assertEqual(len(steps), count)
                buses = self.helpers['bus_order_from_frame'](detail)
                lookup = self.helpers['frame_bus_label_lookup'](detail)
                for metric in ['Absolute Error', '95% PI Width', 'Coverage Miss']:
                    figure = self.helpers['annual_day_heatmap_figure'](detail, metric)
                    heatmap = figure.data[0]
                    self.assertEqual(list(heatmap.x), steps)
                    self.assertEqual(list(heatmap.y), [lookup[b] for b in buses])
                    self.assertEqual(np.asarray(heatmap.z).shape, (3, count))
                    for i, bus in enumerate(buses):
                        ordered = detail[detail.bus.eq(bus)].sort_values('time_step')
                        expected = {
                            'Absolute Error': abs(ordered.predicted_vm_pu - ordered.ground_truth_vm_pu),
                            '95% PI Width': ordered.upper_95_vm_pu - ordered.lower_95_vm_pu,
                            'Coverage Miss': (~(ordered.ground_truth_vm_pu.ge(ordered.lower_95_vm_pu)
                                                & ordered.ground_truth_vm_pu.le(ordered.upper_95_vm_pu))).astype(int),
                        }[metric]
                        np.testing.assert_allclose(heatmap.z[i], expected)
                        self.assertEqual([c[0] for c in heatmap.customdata[i]], ordered.timestamp.tolist())
                        np.testing.assert_allclose([c[2:] for c in heatmap.customdata[i]],
                                                   ordered[['ground_truth_vm_pu', 'predicted_vm_pu', 'lower_95_vm_pu', 'upper_95_vm_pu']])
                    self.assertIn(str(count), figure.layout.title.text)
                    self.assertEqual(figure.layout.xaxis.type, 'linear')
                    if metric == 'Coverage Miss':
                        self.assertEqual((heatmap.zmin, heatmap.zmax), (0, 1))
                        self.assertEqual(list(heatmap.colorbar.ticktext), ['Inside PI (0)', 'Outside PI (1)'])

    def test_dst_clock_labels_preserved(self):
        spring = self.validate(self.fixture(87), 87)
        autumn = self.validate(self.fixture(304), 304)
        spring_times = spring[spring.bus.eq(self.buses[0])].time
        autumn_times = autumn[autumn.bus.eq(self.buses[0])].time
        self.assertFalse(spring_times.str.startswith('02:').any())
        for time in ['02:00', '02:15', '02:30', '02:45']:
            self.assertEqual(autumn_times.eq(time).sum(), 2)
        chart = self.helpers['annual_day_bus_timeseries_figure'](autumn, self.buses[0])
        self.assertEqual(len(set(chart.data[2].x)), 100)
        np.testing.assert_allclose(chart.data[2].y, autumn[autumn.bus.eq(self.buses[0])].ground_truth_vm_pu)

    def test_reject_incomplete_duplicate_and_unknown_grid(self):
        frame = self.fixture(304)
        cases = [frame.iloc[1:], frame[frame.bus.ne(self.buses[0])],
                 frame[frame.time_step.ne(frame.time_step.min())], pd.concat([frame, frame.iloc[:1]])]
        bad = frame.copy()
        bad.loc[0, 'time_step'] = 999999
        cases.append(bad)
        bad = frame.copy()
        bad.loc[0, 'bus'] = 999999
        cases.append(bad)
        for broken in cases:
            with self.subTest(shape=broken.shape):
                with self.assertRaises(ValueError):
                    self.validate(broken, 304)

    def test_reject_wrong_dates_times_and_metadata(self):
        replacements = dict(time='23:59', date='2016-01-01', timestamp='01.01.2016 00:00',
                            simbench_step=-1, day_of_year=1, step_in_day=-1, season='winter')
        for column, value in replacements.items():
            with self.subTest(column=column):
                frame = self.fixture(304)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.validate(frame, 304)
        with self.assertRaises(ValueError):
            self.validate(self.fixture(304), 303)

    def test_reject_fractional_identifiers(self):
        for column in ['bus', 'time_step']:
            frame = self.fixture(1).astype({column: float})
            frame.loc[0, column] += .5
            with self.assertRaises(ValueError):
                self.validate(frame, 1)

    def test_reject_bus_identity_and_role_mismatch(self):
        for column, value in [('simbench_name', 'wrong bus'), ('bus_role', 'wrong role')]:
            frame = self.fixture(1)
            frame[column] = value
            with self.assertRaises(ValueError):
                self.validate(frame, 1)
        cfg = copy.deepcopy(self.cfg)
        cfg['bus_identity'] = []
        with self.assertRaises(ValueError):
            self.helpers['validate_day_detail'](self.fixture(1), 1, cfg)

    def test_legacy_metadata_reconstructed_from_unique_step(self):
        frame = self.fixture(304).drop(columns=['simbench_step', 'timestamp', 'date', 'day_of_year', 'step_in_day'])
        validated = self.validate(frame, 304)
        self.assertEqual(validated.date.unique().tolist(), ['2016-10-30'])
        self.assertEqual(validated.step_in_day.nunique(), 100)

    def test_coverage_is_inclusive_and_not_csv_boolean_cast(self):
        frame = self.fixture(1)
        frame['covered_95'] = 'False'  # Truthiness of strings must not affect the plot.
        detail = self.validate(frame, 1)
        ordered = detail[detail.bus.eq(self.buses[0])]
        self.assertEqual(ordered.covered_95.tolist(), [True, True, True, False] * 24)

    def test_reject_invalid_voltage_and_saved_derived_values(self):
        for column, value in [('ground_truth_vm_pu', np.nan), ('sigma_vm_pu', -.1),
                              ('upper_95_vm_pu', 0), ('absolute_error_vm_pu', 123), ('pi_width_pu', 123)]:
            frame = self.fixture(1)
            frame[column] = value
            with self.assertRaises(ValueError):
                self.validate(frame, 1)

    def test_scope_and_snapshot_share_bus_mapping(self):
        detail = self.validate(self.fixture(304), 304)
        primary = self.helpers['filter_detail_scope'](detail, PRIMARY_SCOPE)
        self.assertEqual(set(primary.bus), set(self.buses[1:]))
        self.assertEqual(set(self.helpers['filter_detail_scope'](detail, ALL_SCOPE).bus), set(self.buses))
        step = detail.time_step.min()
        snapshot = self.helpers['annual_day_all_buses_figure'](detail, step)
        labels = self.helpers['frame_bus_label_lookup'](detail)
        self.assertEqual(set(snapshot.layout.xaxis.categoryarray), set(labels.values()))
        for trace in snapshot.data:
            for name, custom in zip(trace.x, trace.customdata):
                bus = next(b for b, label in labels.items() if label == name)
                row = detail[detail.bus.eq(bus) & detail.time_step.eq(step)].iloc[0]
                self.assertEqual(custom[1], row.ground_truth_vm_pu)

    def test_pipeline_csv_roundtrip_and_saved_metrics(self):
        for day in [87, 304]:
            frame = self.fixture(day)
            steps = sorted(frame.time_step.unique())
            def pivot(column):
                result = frame.pivot(index='time_step', columns='bus', values=column).reindex(index=steps, columns=self.buses)
                result.columns = [f'bus_{b}_vm_pu' for b in self.buses]
                return result
            metadata = frame.drop_duplicates('time_step').set_index('time_step').loc[steps].copy()
            metadata['day'] = day
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                save_daily_diagnostics(pivot('ground_truth_vm_pu'), pivot('predicted_vm_pu'), pivot('sigma_vm_pu'),
                                       metadata, self.buses, self.buses[:1], [day], dsse.season_for_day(day), root,
                                       self.cfg['bus_identity'])
                detail = self.validate(pd.read_csv(root / f'day_{day:03d}.csv.gz'), day)
                metrics = []
                for scope in [PRIMARY_SCOPE, ALL_SCOPE]:
                    part = self.helpers['filter_detail_scope'](detail, scope)
                    error = part.predicted_vm_pu - part.ground_truth_vm_pu
                    metrics.append(dict(day=day, evaluation_scope=scope, mae_pu=abs(error).mean(),
                                        rmse_pu=np.sqrt((error**2).mean()), coverage_95=part.covered_95.mean(),
                                        mean_interval_width_95_pu=part.pi_width_pu.mean()))
                pd.DataFrame(metrics).to_csv(root / 'daily_metrics.csv', index=False)
                self.helpers['verify_day_metrics'](detail, root, day)
                detail.loc[0, 'predicted_vm_pu'] += .1
                with self.assertRaises(ValueError):
                    self.helpers['verify_day_metrics'](detail, root, day)

    def test_streamlit_renders_all_metrics_on_both_dst_days(self):
        from streamlit.testing.v1 import AppTest
        names = {
            'dashboard_timeline', 'load_dashboard_network', 'validate_day_grid',
            'validate_day_detail', 'bus_identity_lookup', 'bus_order_from_frame',
            'frame_bus_label_lookup', 'annual_day_heatmap_figure', 'dashboard_plotly_chart',
        }
        source = (ROOT / 'seasonal/dashboard.py').read_text()
        definitions = [ast.get_source_segment(source, node) for node in ast.parse(source).body
                       if isinstance(node, ast.FunctionDef) and node.name in names]
        script = (
            'import sys, json, inspect, base64\n'
            'import numpy as np\nimport pandas as pd\nimport streamlit as st\n'
            'import plotly.graph_objects as go\n'
            f'sys.path.insert(0, {str(ROOT / "seasonal")!r})\n'
            'import dsse\nfrom dsse import season_for_day, load_network\n'
            + '\n\n'.join(definitions)
        )
        with tempfile.TemporaryDirectory() as directory:
            for day in [87, 304]:
                self.fixture(day).to_csv(Path(directory) / f'{day}.csv', index=False)
            script += f'\ncfg = json.loads({json.dumps(self.cfg)!r})\n'
            script += 'day = st.selectbox("Day", [87, 304])\n'
            script += f'detail = validate_day_detail(pd.read_csv({directory!r} + "/" + str(day) + ".csv"), day, cfg)\n'
            script += 'metric = st.radio("Metric", ["Absolute Error", "95% PI Width", "Coverage Miss"])\n'
            script += 'dashboard_plotly_chart(annual_day_heatmap_figure(detail, metric), width="stretch")\n'
            app = AppTest.from_string(script).run(timeout=60)
            self.assertFalse(app.exception)
            for day, count in [(87, 92), (304, 100)]:
                for metric in ['Absolute Error', '95% PI Width', 'Coverage Miss']:
                    with self.subTest(day=day, metric=metric):
                        app.selectbox[0].set_value(day)
                        app.radio[0].set_value(metric)
                        app.run(timeout=60)
                        self.assertFalse(app.exception)
                        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
                        trace = spec['data'][0]
                        self.assertEqual(len(trace['x']), count)
                        self.assertEqual(np.asarray(trace['z']).shape, (3, count))
                        self.assertEqual(np.asarray(trace['customdata']).shape, (3, count, 6))

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):
            self.helpers['annual_day_heatmap_figure'](self.validate(self.fixture(1), 1), 'typo')


if __name__ == '__main__':
    unittest.main()
