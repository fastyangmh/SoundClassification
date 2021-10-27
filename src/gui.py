# import
from src.project_parameters import ProjectParameters
from src.predict import Predict
import tkinter as tk
from tkinter import Tk, Button, Label, filedialog, messagebox
import numpy as np
from playsound import playsound
from src.utils import get_transform_from_file
import torchaudio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# class


class GUI:
    def __init__(self, project_parameters):
        self.project_parameters = project_parameters
        self.predict_object = Predict(project_parameters=project_parameters)
        self.transform = get_transform_from_file(
            filepath=project_parameters.transform_config_path)['predict']
        self.data_path = None

        # window
        self.window = Tk()
        self.window.geometry('{}x{}'.format(
            self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
        self.window.title('Demo GUI')

        # button
        self.load_audio_button = Button(
            self.window, text='load audio', fg='black', bg='white', command=self._load_audio)
        self.play_audio_button = Button(
            self.window, text='play audio', fg='black', bg='white', command=self._play_audio)
        self.recognize_button = Button(
            self.window, text='recognize', fg='black', bg='white', command=self._recognize)

        # label
        self.data_path_label = Label(self.window, text='', fg='black')
        self.probability_label = Label(self.window, text='', fg='black')
        self.result_label = Label(
            self.window, text='', fg='black', font=(None, 50))

        # matplotlib canvas
        # this is Tkinter default background-color
        facecolor = (0.9254760742, 0.9254760742, 0.9254760742)
        self.gallery_image_canvas = FigureCanvasTkAgg(
            Figure(figsize=(5, 5), facecolor=facecolor), master=self.window)

    def _load_audio(self):
        self.data_path = filedialog.askopenfilename(
            initialdir='./', title='Select audio file', filetypes=(('Audio Files', '.wav .ogg'),   ('All Files', '*.*')))
        waveform, sample_rate = torchaudio.load(filepath=self.data_path)
        if sample_rate != self.project_parameters.sample_rate:
            messagebox.showerror(title='Error!', message='please check the sample_rate and input sample_rate.\nthe sample_rate: {}\n the input sample_rate: {}'.format(
                sample_rate, self.project_parameters.sample_rate))
        if self.transform is not None:
            data = self.transform['audio'](waveform)
            if 'vision' in self.transform:
                data = self.transform['vision'](data)
        data = data.cpu().data.numpy()[0]
        data = data[::-1, :]
        self.gallery_image_canvas.figure.clear()
        subplot1 = self.gallery_image_canvas.figure.add_subplot(211)
        subplot1.plot(np.linspace(0, len(waveform[0]), len(
            waveform[0]))/sample_rate, waveform[0])
        subplot2 = self.gallery_image_canvas.figure.add_subplot(212)
        subplot2.imshow(data)
        subplot2.axis('off')
        self.gallery_image_canvas.figure.tight_layout()
        self.gallery_image_canvas.draw()
        self.data_path_label.config(
            text='image path: {}'.format(self.data_path))

    def _play_audio(self):
        if self.data_path is not None:
            playsound(sound=self.data_path, block=True)
        else:
            messagebox.showerror(
                title='Error!', message='please select an audio file!')

    def _recognize(self):
        if self.data_path is not None:
            probability = self.predict_object(data_path=self.data_path)
            self.probability_label.config(text=('probability:\n'+' {}:{},'*len(probability)).format(
                *np.concatenate(list(zip(self.project_parameters.classes, probability))))[:-1])
            self.result_label.config(
                text=self.project_parameters.classes[probability.argmax()])
        else:
            messagebox.showerror(
                title='Error!', message='please select an image!')

    def run(self):
        self.load_audio_button.pack(anchor=tk.NW)
        self.play_audio_button.pack(anchor=tk.NW)
        self.recognize_button.pack(anchor=tk.NW)
        self.data_path_label.pack(anchor=tk.N)
        self.gallery_image_canvas.get_tk_widget().pack(anchor=tk.N)
        self.probability_label.pack(anchor=tk.N)
        self.result_label.pack(anchor=tk.N)

        # run
        self.window.mainloop()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # GUI
    gui = GUI(project_parameters=project_parameters)
    gui.run()
