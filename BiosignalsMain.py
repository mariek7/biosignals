from SignalType import signal_types # SignalType.py
#from plot_from_file import plot_eeg_from_file # plot_from_file.py
from plot_from_file import plot_phases_from_file # plot_from_file.py
from realtime_plot_save_csv import realtime_plot_and_save # realtime_plot_save_csv.py

from dotenv import load_dotenv
import sys
import os
import json
import requests
import numpy as np
import pandas as pd
import pyqtgraph as pg
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
from datetime import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biosignals - EEG rest-stress-relax")
        #self.setGeometry(100, 100, 800, 600) # x, y, width, height
        self.setMinimumSize(1000, 980)
        self.current_phase = 'rest'  # default phase
        self.current_filename = ''
        
        self.recorded_files = {
            'rest': None,
            'stress': None,
            'relax': None
        }

        #self.setMaximumSize(800, 600)

        self.channel_to_plot = 'A1' # TODO multichannel


        self.plot_widget = FigureCanvas(Figure(figsize=(10, 5)))
        self.plot_widget_isVisible = True
        self.plot_widget.setVisible(self.plot_widget_isVisible)
        self.ax = self.plot_widget.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], lw=2)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        self.init_data()
        # print signal type and samlig rate
        self.signal = signal_types.get(os.getenv('signal_type', 'eeg'), signal_types['eeg'])
        self.sampling_rate = self.signal.sampling_rate

        #self.plot_widget = pg.PlotWidget()
        #self.stop_button = QtWidgets.QPushButton("Stop & save to file")
        self.change_mac_button = QtWidgets.QPushButton("Change MAC address")

        # channel choose
        h_channels = QtWidgets.QHBoxLayout()
        h_channels.addWidget(QtWidgets.QLabel("Choose channels:"))
        self.choose_A1 = QtWidgets.QCheckBox("A1")
        self.choose_A2 = QtWidgets.QCheckBox("A2")
        self.choose_A3 = QtWidgets.QCheckBox("A3")
        self.choose_A4 = QtWidgets.QCheckBox("A4")
        self.choose_A5 = QtWidgets.QCheckBox("A5")
        self.choose_A6 = QtWidgets.QCheckBox("A6")
        h_channels.addWidget(self.choose_A1)
        h_channels.addWidget(self.choose_A2)
        h_channels.addWidget(self.choose_A3)
        h_channels.addWidget(self.choose_A4)
        h_channels.addWidget(self.choose_A5)
        h_channels.addWidget(self.choose_A6)     

        # signal type choose
        h_signal_type = QtWidgets.QHBoxLayout()
        self.signals_combo_box = QtWidgets.QComboBox()
        self.signals_combo_box.addItems(list(signal_types.keys()))   
        #h_signal_type.addWidget(QtWidgets.QLabel("Choose signal type:"))  
        #h_signal_type.addWidget(self.signals_combo_box)  
        h_signal_type.addWidget(QtWidgets.QLabel(f"Signal type: eeg (sampling rate: {self.sampling_rate}, unit: {self.signal.unit})"))
           
        #valitse vaihe: lepo, stressi, rentoutus
        # choose phase
        h_phase = QtWidgets.QHBoxLayout()
        self.phases_combo_box = QtWidgets.QComboBox() 
        self.phases_combo_box.addItems(["rest", "stress", "relax"])
        self.current_phase = self.phases_combo_box.currentText()
        self.phases_combo_box.currentIndexChanged.connect(self.on_phase_changed)
        self.phases_combo_box.setToolTip("Choose phase for data acquisition: rest, stress, relax")
        self.hide_show_plot_widget_button = QtWidgets.QPushButton("Hide/Show live plot")
        self.hide_show_plot_widget_button.clicked.connect(self.hide_show_plot_widget)
        h_phase.addWidget(QtWidgets.QLabel("Choose phase:"))
        h_phase.addWidget(self.phases_combo_box)
        h_phase.addWidget(self.hide_show_plot_widget_button)
        #phase = self.phases_combo_box.currentText()

        #start data aquisition, stop and saved to file, text saved to file
        h_start_stop = QtWidgets.QHBoxLayout()    
        self.start_button = QtWidgets.QPushButton(f"Start acquisition ({self.current_phase})")
        self.stop_button = QtWidgets.QPushButton("Stop and save")
        self.start_button.clicked.connect(self.start_plotting)
        self.stop_button.clicked.connect(self.stop_plotting_and_save)
        h_start_stop.addWidget(self.start_button)
        h_start_stop.addWidget(self.stop_button)


        # text saved data to file xx
        h_saved_text = QtWidgets.QHBoxLayout()
        self.data_saved_text_box = QtWidgets.QTextEdit()
        self.data_saved_text_box.setReadOnly(True)
        self.data_saved_text_box.setFixedHeight(100)
        self.data_saved_text_box.setText(f"")
        h_saved_text.addWidget(self.data_saved_text_box)

        # analyze saved data 
        h_analyze = QtWidgets.QHBoxLayout()
        self.file_to_analyze = QtWidgets.QLineEdit()
        self.file_to_analyze.setFixedWidth(420)
        self.file_to_analyze.setPlaceholderText("Enter filename to analyze (e.g. 2025-05-23_09-59_eeg_rest)")
        self.analyze_button = QtWidgets.QPushButton("Analyze saved data")
        #self.analyze_button.setEnabled(False) # TODO: enable when data is saved
        self.analyze_button.clicked.connect(self.analyze_and_display_results)
        h_analyze.addWidget(self.file_to_analyze)
        h_analyze.addWidget(self.analyze_button)
        #h_analyze.addWidget(QtWidgets.QLabel("Analyze saved data"))

        # show analyze results
        monospace_font = QtGui.QFont("Courier New") # tai "Courier", "Monospace"
        monospace_font.setStyleHint(QtGui.QFont.Monospace)
        h_show_results = QtWidgets.QHBoxLayout() 
        self.results_text_box = QtWidgets.QTextEdit()
        self.results_text_box.setFont(monospace_font)
        self.results_text_box.setReadOnly(True) 
        self.results_text_box.setFixedHeight(200)
        self.results_text_box.setFixedWidth(400)
        self.results_text_box.setText("Results: \n")
        h_show_results.addWidget(self.results_text_box)
        # mock figure to results
        self.mock_result_figure = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap("") 
        #scaled_pixmap = pixmap.scaled(400,200, QtCore.Qt.KeepAspectRatio) #widthxheight
        self.mock_result_figure.setPixmap(pixmap)
        self.mock_result_figure.setScaledContents(True)
        h_show_results.addWidget(self.mock_result_figure)


        layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(self.text_box)
        layout.addWidget(QtWidgets.QLabel(f"MAC address: {self.mac_address} - channels: {self.channel_to_plot}"))
        layout.addWidget(self.change_mac_button)
        layout.addLayout(h_channels)
        layout.addLayout(h_signal_type)
        layout.addLayout(h_phase)
        layout.addWidget(self.plot_widget) # liveplot and results after analysis
        #layout.addWidget(self.checkbox)
        layout.addLayout(h_start_stop)       
        #layout.addWidget(self.save_button)
        layout.addLayout(h_saved_text)
        layout.addLayout(h_analyze)
        layout.addLayout(h_show_results)
        
        layout.addStretch()
        layout.addWidget(QtWidgets.QLabel("-"))
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_phase_changed(self):
        self.current_phase = self.phases_combo_box.currentText()
        self.start_button.setText(f"Start acquisition ({self.current_phase})")
        print(f"Selected phase: {self.current_phase}")

    def init_data(self):
        load_dotenv()
        self.date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.mac_address = os.getenv('MAC_ADDRESS')
        signal_type_key = os.getenv('signal_type', 'None')
        self.signal = signal_types.get(signal_type_key, signal_types['None'])
        self.transfer_func = self.signal.transfer_function
        self.sampling_rate = self.signal.sampling_rate

        self.dt = 1.0 / self.sampling_rate
        self.t = 0
        self.time_buffer = []
        self.data_buffer = []
        self.all_time = []
        self.all_data = []
        self.ax.set_ylim(self.signal.ylim)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(self.signal.unit)

    def hide_show_plot_widget(self):
        self.plot_widget.setVisible(not self.plot_widget_isVisible)
        self.plot_widget_isVisible = not self.plot_widget_isVisible

    def start_plotting(self):
        self.timer.start(300)

    def stop_plotting_and_save(self):
        self.timer.stop()
        filename = f'data_files/data_recording_{self.date_and_time}_{self.signal.name}_{self.current_phase}'
        self.recorded_files[self.current_phase] = filename

        df = pd.DataFrame({'Time (s)': self.all_time, f'{self.signal.name} ({self.signal.unit})': self.all_data})
        df.to_csv(f'{filename}.csv', index=False)
        self.data_saved_text_box.append(f"Data saved to file {filename}")

    def analyze_and_display_results(self):
        #print recorded files
        #print("Recorded files:", self.recorded_files)
        if not self.recorded_files.get("rest"):
            filename = self.file_to_analyze.text().strip()
            if not filename:
                self.results_text_box.setText("Please record the data or enter a filename to analyze.")
                return
            filename = f"data_files/{self.file_to_analyze.text().strip()}"
            
        else:
            filename = self.recorded_files.get("rest")

        base_filename = filename.rsplit("_", 1)[0]  # remove phase
        base_filename = base_filename.split("/")[-1]  # get last part of the path
        #filename = "2025-05-23_09-59_eeg_rest"
        #filename = "data_files/data_recording_2025-06-09_19-32_eeg_rest"
        #"data_files/data_recording_2025-06-09_19-32_eeg_stress"
        #"data_files/data_recording_2025-06-09_19-32_eeg_relax"
        #base_filename = "data_recording_2025-06-09_19-32_eeg"
        print(f"Base filename for analysis: {base_filename}")

        ratios = plot_phases_from_file(base_filename)
        ratios_text = format_ratios(ratios)

        self.plot_widget.setVisible(False)  # hide live plot
        pixmap = QtGui.QPixmap(f'plots/{base_filename}_phase_ratios.png')
        self.mock_result_figure.setPixmap(pixmap)
        self.results_text_box.setFixedHeight(550)
        self.results_text_box.setText(f"{ratios_text}")

        
    def update_plot(self):
        try:
            response = requests.get(f"http://localhost:8000/bitalino-get/?macAddress={self.mac_address}&samplingRate={self.sampling_rate}&recordingTime=1")
            all = json.loads(response.text)
            if "error" in all:
                print("API error:", all["error"])
                print("Detail:", all.get("detail", "No details"))
                return
            
            

            channel_data = np.array(all.get("data", [])[5])
            df = pd.DataFrame(channel_data)
            transfered_data = self.transfer_func(df)
            n_samples = len(transfered_data)
            times = np.arange(self.t, self.t + n_samples * self.dt, self.dt)
            self.t += n_samples * self.dt

            self.all_data.extend(transfered_data)
            self.all_time.extend(times)
            self.data_buffer.extend(transfered_data)
            self.time_buffer.extend(times)

            if len(self.time_buffer) > self.sampling_rate * 2:
                self.time_buffer = self.time_buffer[-self.sampling_rate*2:]
                self.data_buffer = self.data_buffer[-self.sampling_rate*2:]

            self.line.set_data(self.time_buffer, self.data_buffer)
            self.ax.relim()
            self.ax.autoscale_view()
            self.plot_widget.draw()
        except Exception as e:
            print("Error:", e)


# {:>10} oikealle tasattu, 10 merkin levyinen
# {:.2f} kaksi desimaalia
def format_ratios(ratios_and_power):
    text_ratios = "{:>11} {:>8} {:>8} {:>7}\n".format("Ratio", "Rest", "Stress", "Relax")
    text_ratios += "-" * 40 + "\n"
    for key in ['theta_beta', 'theta_alpha', 'alpha_beta']:
        values = ratios_and_power[key]
        rounded = ["{:.2f}".format(float(v)) for v in values]
        text_ratios += "{:>11} {:>8} {:>8} {:>7}\n".format(key, *rounded)

    text_powers = "\n"
    text_powers += "{:>11} {:>8} {:>8} {:>7}\n".format("Power", "Rest", "Stress", "Relax")
    text_powers += "-" * 40 + "\n"
    for key in ['theta_power', 'alpha_power', 'beta_power']:
        values = ratios_and_power[key]
        rounded = ["{:.2f}".format(float(v)) for v in values]
        text_powers += "{:>11} {:>8} {:>8} {:>7}\n".format(key, *rounded)

    total_power_rest = sum(ratios_and_power[key][0] for key in ['theta_power', 'alpha_power', 'beta_power'])
    total_power_stress = sum(ratios_and_power[key][1] for key in ['theta_power', 'alpha_power', 'beta_power'])
    total_power_relax = sum(ratios_and_power[key][2] for key in ['theta_power', 'alpha_power', 'beta_power'])

    relative_powers = {
        key: [
            (ratios_and_power[key][0] / total_power_rest) * 100,
            (ratios_and_power[key][1] / total_power_stress) * 100,
            (ratios_and_power[key][2] / total_power_relax) * 100
        ]
        for key in ['theta_power', 'alpha_power', 'beta_power']
    }
    text_relative_powers = "\n"
    text_relative_powers += "{:>14} {:>6} {:>7} {:>7}\n".format("Relative power", "Rest", "Stress", "Relax")
    text_relative_powers += "-" * 40 + "\n"
    for key, values in relative_powers.items():
        rounded = ["{:.2f}".format(float(v)) for v in values]
        text_relative_powers += "{:>15} {:>6} {:>7} {:>7}\n".format("rel_" + key, *rounded)

    interpretation = "\n---------------------\nThis is just some mock interpretation for now, perhaps I'll create some more useful later :D \n\n"
    if ratios_and_power['theta_beta'][1] < ratios_and_power['theta_beta'][0]:
        interpretation += "Theta/Beta ratio decreased during stress, indicating increased cognitive load.\n\n"
    elif ratios_and_power['theta_beta'][1] > ratios_and_power['theta_beta'][0]:
        interpretation += "Theta/Beta ratio increased during stress, suggesting reduced cognitive load.\n\n"
    if ratios_and_power['theta_alpha'][1] < ratios_and_power['theta_alpha'][0]:
        interpretation += "Theta/Alpha ratio decreased during stress, suggesting heightened alertness.\n\n"
    elif ratios_and_power['theta_alpha'][1] > ratios_and_power['theta_alpha'][0]:
        interpretation += "Theta/Alpha ratio increased during stress, indicating reduced alertness.\n\n"
    if ratios_and_power['alpha_beta'][1] < ratios_and_power['alpha_beta'][0]:
        interpretation += "Alpha/Beta ratio decreased during stress, indicating increased mental effort.\n\n"
    elif ratios_and_power['alpha_beta'][1] > ratios_and_power['alpha_beta'][0]:
        interpretation += "Alpha/Beta ratio increased during stress, suggesting reduced mental effort.\n\n"

    return text_ratios + text_powers + text_relative_powers + interpretation


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
