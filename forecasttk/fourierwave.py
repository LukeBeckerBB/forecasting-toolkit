__author__ = "Christoph Schauer, Alina Nizamutdinova"
__date__ = "2019-11-13"


import numpy
from pandas import Series, period_range
from matplotlib import pyplot


class FourierWave():
    """
    This class encapsulates several attributes and methods for applying a Fourier Transform to a 
    time series, visualizing its main frequencies, fitting a combination of cosine waves for 
    selected frequencies, and generating a forecast with it. 
    Requires as input and returns as output pandas series with PeriodIndex.
    """

    def __init__(self, y, freq):

        # Dictionary mapping 'freq' to num_periods
        freq_dc = {"M": 12, "D": 365.25}

        self.y = y
        self.num_periods = freq_dc[freq]
        self.num_years = int(len(self.y) / int(self.num_periods))
        
        self.fft_arr = None
        self.freq_arr = None
        self.amp_arr = None

        self.freq_idx_ls = None
        self.freq_amp_angle_ls = None

        self.score = None


    def transform(self):
        """
        Computes the Fourier Transform and the corresponding sample frequencies and amplitudes.
        """
        self.fft_arr = numpy.fft.fft(self.y)
        self.freq_arr = numpy.fft.fftfreq(len(self.y))
        self.amp_arr = numpy.abs(self.fft_arr)


    def plot_spectrum(self, max_frequency=6, max_magnitude=None, figsize=(15,5)):
        """
        Plots the positive frequency spectrum for a time series.
        """
        pyplot.figure(figsize=figsize)
        pyplot.stem(self.freq_arr*self.num_periods, self.amp_arr, use_line_collection=True)
        pyplot.xlabel("Frequency (1/year)")
        pyplot.ylabel("Amplitude")
        pyplot.title("Frequency Spectrum")
        pyplot.xlim(0, max_frequency)
        pyplot.ylim(0, max_magnitude)
        pyplot.show()


    def fit(self, frequency_indices):
        """
        Computes the frequencies, amplitudes, and angles for a list of frequency indices.
        """
        self.freq_idx_ls = frequency_indices
        self.freq_amp_angle_ls = []
        self.freq_amp_angle_dc = []

        # Loop over all frequency indices and store their (frequency, amplitude, angle) tuples
        for idx in self.freq_idx_ls:
            freq_i = self.freq_arr[idx]
            amp_i = numpy.abs(self.fft_arr[idx])
            angle_i = numpy.angle(self.fft_arr[idx])
            self.freq_amp_angle_ls.append((freq_i, amp_i, angle_i))


    def predict(self):
        """
        Computes the combination of cosine waves for all stored (frequency, amplitude, angle) 
        tuples with length 'len(y)'. Returns a series with PeriodIndex of the same length.
        """
        # Initialize wave with length 'len(y)', rounded down to full years
        wave_idx = numpy.linspace(0, self.num_years, len(self.y), endpoint=False)
        wave = numpy.zeros(len(wave_idx))

        # Loop over all (frequency, amplitude, angle) tuples and add up cosine waves
        for tpl in self.freq_amp_angle_ls:
            freq_i, amp_i, angle_i = tpl
            wave_i = 2 * amp_i / len(self.y)*numpy.cos(
                2 * numpy.pi * freq_i * self.num_periods * wave_idx+angle_i
                )
            wave += wave_i

        return Series(wave, index=self.y.index)


    def forecast(self, steps):
        """
        Continues the combination of cosine waves for the next 'steps' steps after the last period 
        in 'self.y.index'. Returns a series with PeriodIndex of the same length.
        """
        # Initialize wave with length 'len(y) + steps', rounded up to full years
        num_years_total = self.num_years + int(numpy.ceil(steps/self.num_periods))
        num_steps_total = len(self.y) + int(
            numpy.ceil(steps/self.num_periods
            )*self.num_periods)

        # Loop over all (frequency, amplitude, angle) tuples and add up cosine waves
        wave_idx = numpy.linspace(0, num_years_total, num_steps_total, endpoint=False)
        wave = numpy.zeros(len(wave_idx))
        for tpl in self.freq_amp_angle_ls:
            freq_i, amp_i, angle_i = tpl
            wave_i = 2 * amp_i / len(self.y) * numpy.cos(
                2 * numpy.pi * freq_i * self.num_periods * wave_idx+angle_i)
            wave += wave_i

        # Slice off the forecasted steps from the wave array
        y_fcst = wave[len(self.y):len(self.y) + steps]

        # Generate PeriodIndex for series with forecasted steps
        y_fcst_idx = period_range(self.y.index[-1] + 1, periods=steps, name=self.y.index.name)
        
        return Series(y_fcst, index=y_fcst_idx)
