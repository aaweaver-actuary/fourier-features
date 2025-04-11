from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

    

@dataclass
class AirPlot:
    ax: plt.axes | None = None
    data: PlotData | list[PlotData] | None = None
    fig_params: FigParams | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = PlotData()
        if self.fig_params is None:
            self.fig_params = FigParams()

        if isinstance(self.data, PlotData):
            self.data = [self.data]

        if self.ax is None:
            fig, ax = plt.subplots(figsize=self.fig_params.figsize)
            self.ax = ax


    def plot(self):
        for d in self.data:
            d.ax_params(d.X, d.y, self.ax)
        plt.legend()
        plt.show()
        
@dataclass 
class PlotData:
    data_file: str = "data.csv"
    x_col: str = "Month"
    y_col: str = "Passengers"
    data: pd.DataFrame | None = None
    X: pd.Series | None = None
    y: pd.Series | None = None
    ax_params: AxParams | None = None


    def __post_init__(self):
        if self.data is None:
            self.read_data()

        self.X = self.data[self.x_col]
        self.y = self.data[self.y_col]

    def read_data(self) -> None:
        df = pd.read_csv(self.data_file)
        if self.x_col not in df.columns.tolist():
            raise KeyError(f"{self.x_col} must be set when initializing the data")
        if self.y_col not in df.columns.tolist():
            raise KeyError(f"{self.y_col} must be set when initializing the data")
        self.data = df
        
@dataclass 
class FigParams:
    title:str = "Monthly Airline Passengers, 1949-1960"
    xlabel: str = "Month"
    ylabel: str = "Number of airline passengers"
    figsize: tuple[int, int] = (15, 8)

    def __call__(self, ax):
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

@dataclass
class AxParams:
    line_color: str = 'black'
    line_width: float = 0.5
    fill_color: str | None = None
    marker: str | None = None
    is_line_plot: bool = True
    label: str | None = None

    def __post_init__(self):
        if self.fill_color is None or self.marker is None:
            self.is_line_plot = True
        else:
            self.is_line_plot = False

    def _params(self) -> dict:
        d = {
            'line_color': self.line_color,
            'line_width': self.line_width,
        }

        if self.fill_color is not None:
            d['fill_color'] = self.fill_color

        if self.marker is not None:
            d['marker'] = self.marker

        if self.label is not None:
            d['label'] = self.label

        return d


    def __call__(self, X: pd.Series, y: pd.Series, ax: plt.axes) -> plt.axes:
        if self.is_line_plot:
            ax.plot(X, y, **self._params())
        else:
            ax.scatter(X, y, **self._params())
