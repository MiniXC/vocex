import torch_xla.debug.metrics as met
from datetime import timedelta
import re
from rich import print
from time import time

# def print(*x):
#     pass

# class so we can use "with" statement to get metrics delta for a set of keys

"""
example metrics report string

Metric: CompileTime
  TotalSamples: 14
  Accumulator: 41s970ms754.699us
  ValueRate: 01s008ms052.059us / second
  Rate: 0.344467 / second
  Percentiles: 1%=006ms571.894us; 5%=006ms571.894us; 10%=023ms226.535us; 20%=028ms105.452us; 50%=03s362ms902.957us; 80%=06s507ms373.175us; 90%=08s602ms280.155us; 95%=09s014ms560.750us; 99%=09s014ms560.750us
Metric: DeviceLockWait
  TotalSamples: 33
  Accumulator: 156.682us
  ValueRate: 003.856us / second
  Rate: 0.812053 / second
  Percentiles: 1%=001.891us; 5%=002.381us; 10%=002.443us; 20%=002.633us; 50%=003.548us; 80%=005.099us; 90%=005.705us; 95%=020.288us; 99%=020.536us
"""

# parse time string into pandas timedelta object including milliseconds and microseconds
regex = re.compile(r'^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)?((?P<seconds>[\.\d]+?)s)?((?P<milliseconds>[\.\d]+?)ms)?((?P<microseconds>[\.\d]+?)us)?$')


def timeparse(time_str):
    """
    Parse a time string e.g. (2h13m) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 2h13m)
    :return datetime.timedelta: A datetime.timedelta object
    """
    parts = regex.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)

def report_to_dict(report):
    """Convert a metrics report string to a dict of metrics"""
    lines = report.split('\n')
    metrics = {}
    for line in lines:
        if line.startswith('Metric: '):
            metric = line[8:]
            metrics[metric] = {}
        elif line.startswith('  '):
            key, value = line[2:].split(': ')
            if key == 'Accumulator':
                # parse time
                try:
                    value = timeparse(value)
                except:
                    # just find the first number and assume it's a float (e.g. 41.1B)
                    value = float(re.search(r'\d+', value).group(0))
            elif key == 'TotalSamples':
                value = int(value)
            else:
                continue
            metrics[metric][key] = value
    return metrics

class MetricsDelta:
    def __init__(self, keys, name=None):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.start = report_to_dict(met.metrics_report())
        self.name = name
        self.start_time = time()
    def __enter__(self):
        if self.name:
            print("MetricsDelta: ", self.name)
        return self
    def __exit__(self, type, value, traceback):
        self.end = report_to_dict(met.metrics_report())
        self.delta = {}
        self.end_time = time()
        for key in self.keys:
            try:
                self.delta[key] = {
                    'TotalSamples': self.end[key]['TotalSamples'] - self.start[key]['TotalSamples'],
                    'Accumulator': self.end[key]['Accumulator'] - self.start[key]['Accumulator'],
                    'TotalSamples_total': self.end[key]['TotalSamples'],
                    'Accumulator_total': self.end[key]['Accumulator'],
                }
            except:
                print("MetricsDelta: key not found: ", key)
        print("MetricsDelta: ", self.delta)
        print("MetricsDelta: time elapsed: ", timedelta(seconds=self.end_time - self.start_time))
        print()